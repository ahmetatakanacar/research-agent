import re
from typing import Literal

from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage
from langgraph.graph import END
from langgraph.types import Send

from src.settings import INTERVIEW_TURNS
from src.utils.languages import label, normalize_language
from src.utils.nodes import text_of
from src.utils.states import GenerateAnalystState, InterviewState, ResearchGraphState

load_dotenv()


def should_continue(state: GenerateAnalystState) -> Literal["create_analyst", END]:

    human_analyst_feedback = state.get("human_analyst_feedback", None)
    if human_analyst_feedback:
        return "create_analyst"

    return END


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.casefold()).strip()


def _is_closing(message, language: str) -> bool:
    spoken = _normalize(text_of(message))
    return any(
        _normalize(phrase) and _normalize(phrase) in spoken
        for phrase in (label(language, "closing"), label("en", "closing"))
    )


def route_after_question(state: InterviewState, name: str = "expert"):
    messages = state["messages"]
    answered = any(isinstance(m, AIMessage) and m.name == name for m in messages)
    language = normalize_language((state.get("options") or {}).get("language"))
    if answered and _is_closing(messages[-1], language):
        return "save_interview"

    return "generate_search_query"


def route_messages(state: InterviewState, name: str = "expert"):

    messages = state["messages"]
    max_num_turns = (state.get("options") or {}).get("max_num_turns", INTERVIEW_TURNS)

    num_respones = len([m for m in messages if isinstance(m, AIMessage) and m.name == name])

    if num_respones >= max_num_turns:
        return "save_interview"

    return "ask_question"


def initiate_all_interviews(state: ResearchGraphState):

    human_analyst_feedback = state.get("human_analyst_feedback")

    if human_analyst_feedback:
        return "create_analyst"

    topic = state["topic"]
    options = {
        "language": normalize_language(state.get("language")),
        "max_num_turns": state.get("max_num_turns", INTERVIEW_TURNS),
    }
    return [
        Send("conduct_interview", {
            "options": options,
            "analyst": analyst,
            "messages": [HumanMessage(content=f"so you said you were writing an article on {topic}")],
        }) for analyst in state["analysts"]
    ]
