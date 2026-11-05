from langgraph.graph import END, START, StateGraph

from src.utils.edges import route_after_question, route_messages
from src.utils.nodes import (
    generate_answer,
    generate_question,
    generate_search_query,
    save_interview,
    search_web,
    search_wikipedia,
    write_section,
)
from src.utils.states import InterviewState

SEARCH_NODES = ["search_web", "search_wikipedia"]

interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("generate_search_query", generate_search_query)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_wikipedia", search_wikipedia)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

interview_builder.add_edge(START, "ask_question")
interview_builder.add_conditional_edges("ask_question", route_after_question, ["generate_search_query", "save_interview"])
for node in SEARCH_NODES:
    interview_builder.add_edge("generate_search_query", node)
interview_builder.add_edge(SEARCH_NODES, "answer_question")
interview_builder.add_conditional_edges("answer_question", route_messages, ["ask_question", "save_interview"])
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

question_answer_graph = interview_builder.compile()
