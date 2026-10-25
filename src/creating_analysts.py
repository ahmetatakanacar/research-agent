from src.utils.nodes import create_analyst
from langgraph.graph import START, END, StateGraph
from src.utils.states import GenerateAnalystState
from dotenv import load_dotenv

load_dotenv()

builder = StateGraph(GenerateAnalystState)
builder.add_node("create_analyst", create_analyst)

builder.add_edge(START, "create_analyst")
builder.add_edge("create_analyst", END)

graph = builder.compile()