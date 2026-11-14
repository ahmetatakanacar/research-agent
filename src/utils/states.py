import operator
from typing import Annotated, List, Optional

from langgraph.graph import MessagesState
from typing_extensions import NotRequired, TypedDict

from src.utils.objects import Analyst


class GenerateAnalystState(TypedDict):
    topic: str
    max_analyst: int
    language: NotRequired[str]
    human_analyst_feedback: NotRequired[Optional[str]]
    analysts: NotRequired[List[Analyst]]


class InterviewState(MessagesState):
    # `options` carries language and max_num_turns. They live under one
    # key the parent state does not have: a subgraph writes every key it shares with its parent
    # back, and parallel interviews would collide on plain keys.
    options: dict
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list
    search_query: str
    english_query: str
    warnings: Annotated[list, operator.add]


class ResearchGraphState(TypedDict):
    topic: str
    max_analyst: int
    language: NotRequired[str]
    max_num_turns: NotRequired[int]
    human_analyst_feedback: NotRequired[Optional[str]]
    analysts: List[Analyst]
    sections: Annotated[list, operator.add]
    warnings: Annotated[list, operator.add]
    merged_sections: List[str]
    sources: List[dict]
    introduction: str
    content: str
    conclusion: str
    review_notes: Annotated[List[dict], operator.add]  # the introduction and conclusion add theirs beside the body's
    final_report: str
    final_sources: List[dict]
