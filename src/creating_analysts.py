from dotenv import load_dotenv
from langgraph.graph import START, StateGraph

from src.utils.edges import should_continue
from src.utils.nodes import create_analyst, human_feedback
from src.utils.states import GenerateAnalystState

load_dotenv()

builder = StateGraph(GenerateAnalystState)
builder.add_node("create_analyst", create_analyst)
builder.add_node("human_feedback", human_feedback)

builder.add_edge(START, "create_analyst")
builder.add_edge("create_analyst", "human_feedback")
builder.add_conditional_edges("human_feedback", should_continue)

graph = builder.compile()
