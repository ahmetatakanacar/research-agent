from dotenv import load_dotenv
from src.utils.states import GenerateAnalystState, InterviewState
from typing import Literal
from langgraph.graph import END
from langchain.messages import AIMessage

load_dotenv()

def should_continue(state: GenerateAnalystState) -> Literal["create_analyst", END]:

    human_analyst_feedback = state.get("human_analyst_feedback", None)
    if human_analyst_feedback:
        return "create_analyst"

    return END

def route_messages(state: InterviewState, name: str="expert"):

    messages = state["messages"]
    max_num_turns = state.get("max_num_turns", 2)

    num_respones = len([m for m in messages if isinstance(m, AIMessage) and m.name == name])

    if num_respones >= max_num_turns:
        return "save_interview"

    return "ask_question"