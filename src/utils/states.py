from typing_extensions import TypedDict, NotRequired, Annotated
from typing import Optional, List
from src.utils.objects import Analyst
from langgraph.graph import MessagesState
import operator

class GenerateAnalystState(TypedDict):
    topic: str
    max_analyst: int
    human_analyst_feedback: NotRequired[Optional[str]]
    analyst: NotRequired[List[Analyst]]

class InterviewState(MessagesState):
    max_num_turns: int
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list