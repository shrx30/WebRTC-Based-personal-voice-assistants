from fastapi import APIRouter
from pydantic import BaseModel

from app.graph.workflow import graph


router = APIRouter()


class ResearchRequest(BaseModel):
    query: str


@router.get("/")
def root():
    return {"message": "Distributed Agent Platform API"}


@router.get("/health")
def health():
    return {"status": "healthy"}


@router.post("/research")
def research(request: ResearchRequest):

    result = graph.invoke(
        {
            "query": request.query
        }
    )

    return {
    "report": result["report"]
}