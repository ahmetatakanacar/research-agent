from langgraph.graph import END, START, StateGraph

from src.answering_question import interview_builder
from src.utils.edges import initiate_all_interviews
from src.utils.nodes import (
    create_analyst,
    finalize_report,
    human_feedback,
    merge_sources,
    review_report,
    write_conclusion,
    write_introduction,
    write_report,
)
from src.utils.states import ResearchGraphState

builder = StateGraph(ResearchGraphState)
builder.add_node("create_analyst", create_analyst)
builder.add_node("human_feedback", human_feedback)
builder.add_node("conduct_interview", interview_builder.compile())
builder.add_node("merge_sources", merge_sources)
builder.add_node("write_report", write_report)
builder.add_node("write_introduction", write_introduction)
builder.add_node("write_conclusion", write_conclusion)
builder.add_node("review_report", review_report)
builder.add_node("finalize_report", finalize_report)

builder.add_edge(START, "create_analyst")
builder.add_edge("create_analyst", "human_feedback")
builder.add_conditional_edges("human_feedback", initiate_all_interviews, ["create_analyst", "conduct_interview"])
builder.add_edge("conduct_interview", "merge_sources")
builder.add_edge("merge_sources", "write_report")
builder.add_edge("merge_sources", "write_introduction")
builder.add_edge("merge_sources", "write_conclusion")
builder.add_edge("write_report", "review_report")
builder.add_edge(["review_report", "write_introduction", "write_conclusion"], "finalize_report")
builder.add_edge("finalize_report", END)

graph = builder.compile()
