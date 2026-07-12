from typing import TypedDict


class ResearchState(TypedDict):
    query: str

    retrieved_context: str

    answer: str

    report: str