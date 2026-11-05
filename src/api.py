import logging
import os
import re
import unicodedata
import uuid
from contextlib import AsyncExitStack, asynccontextmanager

import aiosqlite
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field

from src.agent import builder
from src.runner import RunConflict, RunManager, RunsBusy
from src.security import RateLimiter, SecurityHeadersMiddleware, check_api_key, client_key
from src.settings import ROOT_DIR, Settings, get_settings
from src.store import Store
from src.utils.languages import language_options, normalize_language

logger = logging.getLogger(__name__)

WEB_DIR = ROOT_DIR / "web"
THREAD_ID = re.compile(r"^[0-9a-f]{32}$")
RESUMABLE = {"cancelled", "interrupted", "error"}


class StartRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    topic: str = Field(min_length=3, max_length=300)
    max_analyst: int = Field(default=3, ge=1, le=6)
    language: str = Field(default="en", max_length=8)


class AnalystEdit(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=120)
    role: str = Field(min_length=1, max_length=200)
    affiliation: str = Field(default="", max_length=200)
    description: str = Field(default="", max_length=1500)


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    feedback: str = Field(default="", max_length=1000)
    analysts: list[AnalystEdit] | None = Field(default=None, min_length=1, max_length=6)


def _ascii_slug(text: str) -> str:
    text = text.translate({ord("ı"): "i", ord("İ"): "I", ord("ß"): "ss"})
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "report"


def _public_run(run: dict, active: bool) -> dict:
    return {
        "thread_id": run["thread_id"],
        "topic": run["topic"],
        "language": run["language"],
        "status": run["status"],
        "running": active,
        "settings": run["settings"],
        "analysts": run["analysts"],
        "report": run["report"],
        "sources": run["sources"] or [],
        "warnings": run["warnings"] or [],
        "review_notes": run["review_notes"] or [],
        "usage": run["usage"],
        "error": run["error"],
        "created_at": run["created_at"],
        "updated_at": run["updated_at"],
    }


def create_app(settings: Settings | None = None, checkpointer=None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = Store(settings.app_db)
        interrupted = store.mark_interrupted()
        if interrupted:
            logger.warning("%d run(s) were interrupted by the last shutdown", interrupted)

        async with AsyncExitStack() as stack:
            saver = checkpointer
            if saver is None:
                connection = await aiosqlite.connect(settings.checkpoint_db)
                stack.push_async_callback(connection.close)
                saver = AsyncSqliteSaver(
                    connection,
                    serde=JsonPlusSerializer(allowed_msgpack_modules=[("src.utils.objects", "Analyst")]),
                )
                await saver.setup()

            graph = builder.compile(checkpointer=saver)
            app.state.store = store
            app.state.checkpointer = saver
            app.state.graph = graph
            app.state.manager = RunManager(graph, store, settings)
            try:
                yield
            finally:
                await app.state.manager.shutdown()
                store.close()

    app = FastAPI(
        title="Research Agent",
        lifespan=lifespan,
        docs_url=None if settings.api_key else "/docs",
        redoc_url=None,
        openapi_url=None if settings.api_key else "/openapi.json",
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.state.settings = settings
    run_limiter = RateLimiter(settings.rate_limit_per_hour)

    def auth(request: Request) -> None:
        check_api_key(request, settings.api_key)

    def throttle(limiter: RateLimiter, request: Request) -> None:
        retry_after = limiter.check(client_key(request, settings.trust_proxy))
        if retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please try again later.",
                headers={"Retry-After": str(int(retry_after))},
            )

    def store_of(request: Request) -> Store:
        return request.app.state.store

    def manager_of(request: Request) -> RunManager:
        return request.app.state.manager

    def require_run(request: Request, thread_id: str) -> dict:
        run = store_of(request).get_run(thread_id) if THREAD_ID.match(thread_id) else None
        if run is None:
            raise HTTPException(status_code=404, detail="Research not found.")
        return run

    async def pending_steps(request: Request, thread_id: str) -> tuple:
        state = await request.app.state.graph.aget_state({"configurable": {"thread_id": thread_id}})
        return tuple(state.next or ())

    def begin(request: Request, thread_id: str, graph_input) -> None:
        try:
            manager_of(request).start(thread_id, graph_input)
        except RunConflict:
            raise HTTPException(status_code=409, detail="This research is already running.") from None
        except RunsBusy:
            raise HTTPException(
                status_code=429,
                detail="The server is busy with other researches. Please try again in a few minutes.",
                headers={"Retry-After": "30"},
            ) from None

    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "tavily": bool(os.getenv("TAVILY_API_KEY")),
            "auth": bool(settings.api_key),
            "languages": language_options(),
        }

    @app.get("/api/auth/check", dependencies=[Depends(auth)])
    def auth_check():
        return {"ok": True}

    @app.post("/api/research", status_code=201, dependencies=[Depends(auth)])
    async def start_research(body: StartRequest, request: Request):
        store = store_of(request)

        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not set on the server.")
        if not os.getenv("TAVILY_API_KEY"):
            raise HTTPException(status_code=503, detail="TAVILY_API_KEY is not set on the server.")

        throttle(run_limiter, request)

        language = normalize_language(body.language)
        thread_id = uuid.uuid4().hex
        record = {"max_analyst": body.max_analyst}
        store.create_run(thread_id, body.topic, language, record)
        try:
            begin(request, thread_id, {
                "topic": body.topic,
                "max_analyst": body.max_analyst,
                "language": language,
            })
        except HTTPException:
            store.delete_run(thread_id)
            raise
        return {"thread_id": thread_id}

    @app.get("/api/research", dependencies=[Depends(auth)])
    def list_research(request: Request):
        return {"runs": store_of(request).list_runs()}

    @app.get("/api/research/{thread_id}", dependencies=[Depends(auth)])
    def get_research(thread_id: str, request: Request):
        return _public_run(require_run(request, thread_id), manager_of(request).is_active(thread_id))

    @app.get("/api/research/{thread_id}/events", dependencies=[Depends(auth)])
    async def research_events(thread_id: str, request: Request, after: int = Query(default=0, ge=0)):
        require_run(request, thread_id)
        run = manager_of(request).get(thread_id)
        if run is None:
            raise HTTPException(status_code=404, detail="No live run for this research.")
        return StreamingResponse(
            manager_of(request).stream(run, after),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/research/{thread_id}/feedback", dependencies=[Depends(auth)])
    async def send_feedback(thread_id: str, body: FeedbackRequest, request: Request):
        run = require_run(request, thread_id)
        if run["status"] != "awaiting_approval" or not await pending_steps(request, thread_id):
            raise HTTPException(status_code=409, detail="This research is not waiting for approval.")
        if manager_of(request).is_active(thread_id):
            raise HTTPException(status_code=409, detail="This research is already running.")
        throttle(run_limiter, request)

        edited = [a.model_dump() for a in body.analysts] if body.analysts else None
        begin(request, thread_id, Command(resume={"feedback": body.feedback, "analysts": edited}))
        if edited:
            store_of(request).update_run(thread_id, analysts=edited)
        return {"ok": True}

    @app.post("/api/research/{thread_id}/resume", dependencies=[Depends(auth)])
    async def resume_research(thread_id: str, request: Request):
        run = require_run(request, thread_id)
        if run["status"] not in RESUMABLE or not await pending_steps(request, thread_id):
            raise HTTPException(status_code=409, detail="This research cannot be resumed.")
        throttle(run_limiter, request)
        begin(request, thread_id, None)
        return {"ok": True}

    @app.post("/api/research/{thread_id}/cancel", dependencies=[Depends(auth)])
    async def cancel_research(thread_id: str, request: Request):
        require_run(request, thread_id)
        return {"cancelled": await manager_of(request).cancel(thread_id)}

    @app.delete("/api/research/{thread_id}", status_code=204, dependencies=[Depends(auth)])
    async def delete_research(thread_id: str, request: Request):
        require_run(request, thread_id)
        await manager_of(request).cancel(thread_id)
        store_of(request).delete_run(thread_id)
        try:
            await request.app.state.checkpointer.adelete_thread(thread_id)
        except Exception:
            logger.warning("Could not delete checkpoints for %s", thread_id, exc_info=True)
        return Response(status_code=204)

    @app.get("/api/research/{thread_id}/export", dependencies=[Depends(auth)])
    def export_report(thread_id: str, request: Request):
        run = require_run(request, thread_id)
        if not run["report"]:
            raise HTTPException(status_code=409, detail="There is no finished report to export.")
        return Response(
            content=run["report"].encode("utf-8"),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{_ascii_slug(run["topic"])}.md"'},
        )

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app
