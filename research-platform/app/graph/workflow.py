from langgraph.graph import StateGraph, START, END

from app.graph.state import ResearchState

from app.graph.nodes import (
    planner_node,
    retrieval_node,
    research_node,
    report_node
)

builder = StateGraph(ResearchState)

builder.add_node("planner", planner_node)

builder.add_node("retriever", retrieval_node)

builder.add_node("research", research_node)

builder.add_node("report", report_node)

builder.add_edge(START, "planner")

builder.add_edge("planner", "retriever")

builder.add_edge("retriever", "research")

builder.add_edge("research", "report")

builder.add_edge("report", END)

graph = builder.compile()