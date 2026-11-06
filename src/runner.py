import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator

from src.settings import Settings
from src.store import Store
from src.usage import UsageTracker, add_usage

logger = logging.getLogger(__name__)

SEARCH_NODES = {"search_web": "web", "search_wikipedia": "wikipedia"}
HEARTBEAT_SECONDS = 15.0
DRAFT_FLUSH_CHARS = 80
DRAFT_FLUSH_SECONDS = 0.2
USAGE_EVERY_SECONDS = 1.0


class RunConflict(Exception):
    pass


class RunsBusy(Exception):
    pass


def _class_names(exc: BaseException) -> set[str]:
    # Match the whole class hierarchy: langchain-openai raises subclasses such as
    # OpenAIAuthenticationError, whose own name is not the one the openai package uses.
    return {cls.__name__ for cls in type(exc).__mro__}


def out_of_credit(exc: BaseException) -> bool:
    """The OpenAI account has no credit left. OpenAI reports this as a rate-limit error (429), but waiting
    does not help: the account has to be topped up."""
    return "RateLimitError" in _class_names(exc) and (
        getattr(exc, "code", None) == "insufficient_quota" or "insufficient_quota" in str(exc))


def friendly_error(exc: BaseException) -> dict:
    """Map an exception to a message that is safe to show. Details stay in the server log."""
    ref = uuid.uuid4().hex[:8]
    logger.error("Run failed [%s]", ref, exc_info=exc)

    names = _class_names(exc)
    code, message = "unknown", "Something went wrong while researching."
    if "AuthenticationError" in names:
        code, message = "auth", "The OpenAI API key was rejected."
    elif out_of_credit(exc):
        code, message = "quota", "The OpenAI account has no credit left."
    elif "RateLimitError" in names:
        code, message = "rate_limit", "The OpenAI API rate limit was reached."
    elif "NotFoundError" in names:
        code, message = "model", "The configured OpenAI model was not found."
    elif names & {"APITimeoutError", "ReadTimeout", "ConnectTimeout"}:
        code, message = "timeout", "The model took too long to respond."
    elif names & {"APIConnectionError", "ConnectError"}:
        code, message = "connection", "Could not reach the model provider."
    return {"code": code, "message": message, "ref": ref}


def sse(event_id: int, event: str, data: Any) -> str:
    return f"id: {event_id}\nevent: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p if isinstance(p, str) else str(p.get("text", "")) for p in content if isinstance(p, (str, dict)))
    return json.dumps(content, ensure_ascii=False)


class Run:
    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self.events: list[tuple[int, str, Any]] = []
        self.finished = False
        self.finished_at: float | None = None
        self.task: asyncio.Task | None = None
        self.cond = asyncio.Condition()

    async def emit(self, name: str, data: Any) -> None:
        async with self.cond:
            self.events.append((len(self.events) + 1, name, data))
            self.cond.notify_all()

    async def close(self) -> None:
        async with self.cond:
            self.finished = True
            self.finished_at = time.monotonic()
            self.cond.notify_all()


class RunManager:
    def __init__(self, graph, store: Store, settings: Settings):
        self.graph = graph
        self.store = store
        self.settings = settings
        self.runs: dict[str, Run] = {}
        self._closing = False

    def get(self, thread_id: str) -> Run | None:
        return self.runs.get(thread_id)

    def is_active(self, thread_id: str) -> bool:
        run = self.runs.get(thread_id)
        return bool(run and not run.finished)

    def active_count(self) -> int:
        return sum(1 for run in self.runs.values() if not run.finished)

    def _prune(self) -> None:
        cutoff = time.monotonic() - self.settings.run_ttl_seconds
        for thread_id in [t for t, r in self.runs.items() if r.finished and (r.finished_at or 0) < cutoff]:
            del self.runs[thread_id]

    def start(self, thread_id: str, graph_input: Any) -> Run:
        """Begin a run segment. Must be called from the event loop thread."""
        if self.is_active(thread_id):
            raise RunConflict(thread_id)
        if self.active_count() >= self.settings.max_concurrent_runs:
            raise RunsBusy()
        self._prune()

        run = Run(thread_id)
        self.runs[thread_id] = run
        self.store.update_run(thread_id, status="running", error=None)
        run.task = asyncio.create_task(self._execute(run, graph_input))
        return run

    async def cancel(self, thread_id: str) -> bool:
        run = self.runs.get(thread_id)
        if not run or run.finished or run.task is None:
            return False
        run.task.cancel()
        await asyncio.gather(run.task, return_exceptions=True)
        return True

    async def shutdown(self) -> None:
        self._closing = True
        tasks = [run.task for run in self.runs.values() if run.task and not run.task.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stream(self, run: Run, after: int = 0, heartbeat: float = HEARTBEAT_SECONDS) -> AsyncIterator[str]:
        index = max(0, after)
        while True:
            async with run.cond:
                if index >= len(run.events) and not run.finished:
                    try:
                        await asyncio.wait_for(run.cond.wait(), timeout=heartbeat)
                    except asyncio.TimeoutError:
                        pass
                batch = run.events[index:]
                finished = run.finished
            for event_id, name, data in batch:
                yield sse(event_id, name, data)
            index += len(batch)
            if finished and not batch:
                return
            if not batch:
                yield ": ping\n\n"

    async def _execute(self, run: Run, graph_input: Any) -> None:
        thread_id = run.thread_id
        previous = (self.store.get_run(thread_id) or {}).get("usage")
        tracker = UsageTracker(self.settings.price_input_per_1m, self.settings.price_output_per_1m)
        started = time.monotonic()
        config = {"configurable": {"thread_id": thread_id}, "callbacks": [tracker], "recursion_limit": 80}

        interview_ids: dict[str, int] = {}
        interview_meta: dict[str, dict] = {}
        draft = {"text": "", "last": time.monotonic()}
        last_usage = 0.0
        outcome: dict[str, Any] = {"status": "error"}

        async def flush_draft(force: bool = False) -> None:
            now = time.monotonic()
            if draft["text"] and (force or len(draft["text"]) >= DRAFT_FLUSH_CHARS or now - draft["last"] >= DRAFT_FLUSH_SECONDS):
                await run.emit("draft", {"delta": draft["text"]})
                draft["text"], draft["last"] = "", now

        def usage_now() -> dict:
            return add_usage(
                previous, tracker.snapshot(time.monotonic() - started),
                self.settings.price_input_per_1m, self.settings.price_output_per_1m,
            )

        async def ensure_interview(key: str) -> int:
            if key not in interview_ids:
                interview_ids[key] = len(interview_ids) + 1
                await run.emit("interview_start", {"interview": interview_ids[key], **interview_meta.get(key, {})})
            return interview_ids[key]

        try:
            async for namespace, mode, payload in self.graph.astream(
                graph_input, config, stream_mode=["updates", "custom"], subgraphs=True
            ):
                if mode == "custom":
                    if isinstance(payload, dict) and "draft" in payload and not namespace:
                        draft["text"] += payload["draft"]
                        await flush_draft()
                    elif isinstance(payload, dict) and "interview" in payload and namespace:
                        interview_meta[namespace[0]] = payload["interview"]
                        await ensure_interview(namespace[0])
                    continue

                await flush_draft(force=True)
                for node, update in payload.items():
                    update = update or {}
                    if node == "__interrupt__":
                        value = update[0].value
                        outcome = {"status": "awaiting_approval", "analysts": value.get("analysts", [])}
                        await run.emit("analysts", {"analysts": outcome["analysts"]})
                        continue

                    if not namespace:
                        await run.emit("stage", {"node": node})
                        continue

                    interview = await ensure_interview(namespace[0])
                    if node in SEARCH_NODES:
                        docs = update.get("context") or []
                        await run.emit("search", {
                            "interview": interview,
                            "source": SEARCH_NODES[node],
                            "found": len(docs),
                            "titles": [d.get("title", "") for d in docs[:3] if isinstance(d, dict)],
                        })
                    elif node in {"ask_question", "answer_question"}:
                        role = "interviewer" if node == "ask_question" else "expert"
                        for message in update.get("messages", []):
                            await run.emit("message", {"interview": interview, "role": role, "content": message_text(message)})
                    else:
                        await run.emit("interview_step", {"interview": interview, "node": node})

                if time.monotonic() - last_usage >= USAGE_EVERY_SECONDS:
                    last_usage = time.monotonic()
                    await run.emit("usage", usage_now())

            await flush_draft(force=True)
            state = await self.graph.aget_state(config)
            values = state.values or {}
            if values.get("final_report") and not state.next:
                outcome = {
                    "status": "done",
                    "report": values["final_report"],
                    "sources": values.get("final_sources") or [],
                    "warnings": values.get("warnings") or [],
                    "review_notes": values.get("review_notes") or [],
                }
            elif outcome["status"] != "awaiting_approval":
                outcome = {"status": "error", "error": {
                    "code": "unknown", "message": "The research ended unexpectedly.", "ref": uuid.uuid4().hex[:8]}}

        except asyncio.CancelledError:
            outcome = {"status": "interrupted" if self._closing else "cancelled"}
        except Exception as exc:
            outcome = {"status": "error", "error": friendly_error(exc)}

        # From here on the run is over; persist first, then tell subscribers.
        usage = usage_now()
        fields: dict[str, Any] = {"status": outcome["status"], "usage": usage, "error": outcome.get("error")}
        if outcome["status"] == "awaiting_approval":
            fields["analysts"] = outcome["analysts"]
        if outcome["status"] == "done":
            fields.update(report=outcome["report"], sources=outcome["sources"],
                          warnings=outcome["warnings"], review_notes=outcome["review_notes"])
        try:
            self.store.update_run(run.thread_id, **fields)
        except Exception:
            logger.exception("Could not save run %s", run.thread_id)

        try:
            await flush_draft(force=True)
            if outcome["status"] == "done":
                await run.emit("report", {
                    "final_report": outcome["report"], "sources": outcome["sources"],
                    "warnings": outcome["warnings"], "review_notes": outcome["review_notes"], "usage": usage,
                })
            elif outcome["status"] == "error":
                await run.emit("error", outcome["error"])
            elif outcome["status"] in {"cancelled", "interrupted"}:
                await run.emit("cancelled", {"status": outcome["status"]})
            await run.emit("usage", usage)
            await run.emit("done", {"status": outcome["status"]})
        finally:
            await run.close()
