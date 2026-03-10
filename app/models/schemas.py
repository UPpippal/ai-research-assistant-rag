from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="User's natural language query")
    top_k: int = Field(5, description="Number of documents to retrieve")


class RetrievedDocument(BaseModel):
    id: str
    score: float
    source: str = Field(..., description="Path or logical source identifier")
    content: str
    meta: Optional[dict] = None


class RagResponse(BaseModel):
    query: str
    answer: str
    documents: List[RetrievedDocument]

