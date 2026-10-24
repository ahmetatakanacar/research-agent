from typing_extensions import TypedDict, NotRequired
from typing import Optional, List
from objects import Analyst

class GenerateAnalystState(TypedDict):
    topic: str
    max_analyst: int
    human_analyst_feedback: NotRequired[Optional[str]]
    analyst: NotRequired[List[Analyst]]