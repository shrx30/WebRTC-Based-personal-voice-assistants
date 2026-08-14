from app.graph.state import ResearchState
from app.llm.models import llm


def planner_node(state: ResearchState):

    print("Planner running...")

    return {}


def retrieval_node(state: ResearchState):

    print("Retrieving documents...")

    return {
        "retrieved_context": ""
    }


def research_node(state: ResearchState):

    prompt = f"""
Question:

{state['query']}

Context:

{state['retrieved_context']}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


def report_node(state: ResearchState):

    report = f"""
# Research Report

Question:
{state['query']}

Answer:

{state['answer']}
"""

    return {
        "report": report
    }