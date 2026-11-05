import re
import time
from typing import Any, Callable, Iterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda

from src.utils.objects import Analyst, Perpectives, ReviewIssue, ReviewResult, SearchQuery, SourcePicks


def system_text(messages: list[BaseMessage]) -> str:
    return "\n".join(str(m.content) for m in messages if m.type == "system")


def last_human(messages: list[BaseMessage]) -> str:
    humans = [str(m.content) for m in messages if isinstance(m, HumanMessage)]
    return humans[-1] if humans else ""


class ScriptedChat(BaseChatModel):
    review_issues: list = []
    calls: list = []
    fail_on: str = ""
    close_after: int = 0
    delay: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def respond(self, messages: list[BaseMessage]) -> str:
        if self.delay:
            time.sleep(self.delay)
        system = system_text(messages)
        kind = "other"

        if "interviewing an expert" in system:
            kind = "question"
            answers = sum(1 for m in messages if isinstance(m, AIMessage) and m.name == "expert")
            closing = (re.search(r'exactly this sentence: "(.*)"', system) or [0, "Thank you"])[1]
            text = closing if self.close_after and answers >= self.close_after else (
                f"Question {answers + 1}: how does it work in practice?"
            )
        elif "expert being interviewed" in system:
            kind = "answer"
            cites = " ".join(f"[{n}]" for n in (1, 2) if f"[{n}] " in system)
            text = f"The evidence points to a clear effect {cites}."
        elif "expert technical writer" in system:
            kind = "section"
            text = "## A fresh angle\n\n### Summary\n\nFirst claim [1]. Second claim [2]. Third claim [9]."
        elif "creating a report on this overall topic" in system:
            kind = "report"
            text = "## Insights\n\nThe memos agree [1][2]. Another point [3]."
        elif "finishing a report" in system:
            kind = "intro" if "introduction" in last_human(messages).lower() else "conclusion"
            heading = "# Big Title\n\n## Introduction" if kind == "intro" else "## Conclusion"
            text = f"{heading}\n\nSummary sentence [1]."
        elif "You are an editor" in system:
            kind = "revise"
            text = "## Insights\n\nThe memos agree [1]."
        else:
            text = "ok"

        self.calls.append(kind)
        if self.fail_on and self.fail_on == kind:
            raise RuntimeError(f"scripted failure in {kind}")
        return text

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        message = AIMessage(content=self.respond(messages), usage_metadata=self._usage())
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        text = self.respond(messages)
        words = re.findall(r"\S+\s*", text)
        for word in words:
            yield ChatGenerationChunk(message=AIMessageChunk(content=word))
        yield ChatGenerationChunk(message=AIMessageChunk(content="", usage_metadata=self._usage()))

    @staticmethod
    def _usage() -> dict:
        return {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}

    def with_structured_output(self, schema, **kwargs: Any):
        def build(messages):
            system = system_text(messages)
            if self.delay:
                time.sleep(self.delay)
            if schema is Perpectives:
                self.calls.append("analysts")
                if self.fail_on == "analysts":
                    raise RuntimeError("secret internal detail")
                count = int((re.search(r"exactly (\d+) analysts", system) or [0, 2])[1])
                feedback = (re.search(r"creation of the analysts:[ \t]*(.*)", system) or [0, ""])[1].strip()
                suffix = f" (revised: {feedback})" if feedback else ""
                return Perpectives(analyst=[
                    Analyst(
                        name=f"Analyst {i}", role=f"Role {i}", affiliation=f"Org {i}",
                        description=f"Focus {i}{suffix}",
                    )
                    for i in range(1, count + 1)
                ])
            if schema is SearchQuery:
                self.calls.append("query")
                question = last_human(messages) or str(messages[-1].content)
                return SearchQuery(search_query=f"query {question[:80]}", english_query=f"english {question[:80]}")
            if schema is SourcePicks:
                self.calls.append("select")
                return SourcePicks(best=[1, 2, 3, 4, 5])
            if schema is ReviewResult:
                self.calls.append("review")
                if self.fail_on == "review":
                    raise RuntimeError("scripted failure in review")
                return ReviewResult(issues=[ReviewIssue(**issue) for issue in self.review_issues])
            raise AssertionError(f"unexpected schema {schema}")

        return RunnableLambda(build)


def make_llm(**kwargs: Any) -> ScriptedChat:
    return ScriptedChat(calls=[], **kwargs)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:30]


def fake_tavily(query: str, search_depth: str = "basic", max_results: int = 3) -> list[dict]:
    return [
        {"url": "https://example.com/shared", "title": "Shared page", "content": "Shared content.", "kind": "web"},
        {"url": f"https://example.com/{slug(query)}", "title": f"Page for {query}", "content": f"About {query}.", "kind": "web"},
    ]


def fake_wikipedia(query: str, lang: str = "en", max_docs: int = 2, max_chars: int = 4000) -> list[dict]:
    return [{"url": f"https://{lang}.wikipedia.org/wiki/Topic", "title": "Topic", "content": "Encyclopedia text.", "kind": "wikipedia"}]


def empty_search(*args: Any, **kwargs: Any) -> list[dict]:
    return []


Responder = Callable[[list[BaseMessage]], str]
