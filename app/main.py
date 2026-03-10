from fastapi import FastAPI

from app.config import settings
from app.models.schemas import QueryRequest, RagResponse
from app.rag.pipeline import HybridRAGPipeline


app = FastAPI(title=settings.project_name)
pipeline = HybridRAGPipeline()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=RagResponse)
def query_rag(request: QueryRequest) -> RagResponse:
    """
    Hybrid RAG endpoint.
    """
    return pipeline.answer(query=request.query, top_k=request.top_k)

