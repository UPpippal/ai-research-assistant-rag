from __future__ import annotations

from pathlib import Path
from typing import List

from app.config import settings
from app.models.schemas import RagResponse, RetrievedDocument
from app.retrievers.bm25_retriever import BM25Retriever, load_bm25_from_raw
from app.retrievers.vector_retriever import (
    EmbeddingModel,
    VectorRetriever,
    build_vector_store_from_raw,
)
from app.retrievers.hybrid_retriever import HybridRetriever


class SimpleLLMClient:
    """
    Placeholder LLM client.

    Replace `generate` with a real call to your LLM provider.
    """

    def generate(self, prompt: str) -> str:
        return (
            "This is a placeholder answer. "
            "Wire this up to your preferred LLM provider."
        )


class HybridRAGPipeline:
    def __init__(
        self,
        bm25: BM25Retriever | None = None,
        vector: VectorRetriever | None = None,
        embedding_model: EmbeddingModel | None = None,
        llm_client: SimpleLLMClient | None = None,
        raw_data_dir: Path | None = None,
    ) -> None:
        self.raw_data_dir = raw_data_dir or settings.raw_data_dir

        self.embedding_model = embedding_model or EmbeddingModel()
        self.bm25 = bm25 or load_bm25_from_raw(self.raw_data_dir)
        self.vector = vector or build_vector_store_from_raw(
            embedding_model=self.embedding_model,
            dir_path=self.raw_data_dir,
        )

        self.hybrid = HybridRetriever(
            bm25_retriever=self.bm25,
            vector_retriever=self.vector,
            embedding_model=self.embedding_model,
        )

        self.llm_client = llm_client or SimpleLLMClient()

    def _load_doc_text(self, doc_id: str) -> str:
        path = Path(doc_id)
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="ignore")
        return f"[Missing document content for {doc_id}]"

    def answer(self, query: str, top_k: int) -> RagResponse:
        hybrid_results = self.hybrid.retrieve(query=query, top_k=top_k)

        docs: List[RetrievedDocument] = []
        for doc_id, score in hybrid_results:
            content = self._load_doc_text(doc_id)
            docs.append(
                RetrievedDocument(
                    id=doc_id,
                    score=score,
                    source=doc_id,
                    content=content,
                    meta=None,
                )
            )

        context_snippets = "\n\n---\n\n".join(
            f"Source: {doc.source}\nScore: {doc.score:.4f}\n{doc.content[:1000]}"
            for doc in docs
        )

        prompt = (
            "You are a helpful assistant that answers questions based on the provided documents.\n\n"
            f"Question:\n{query}\n\n"
            "Relevant documents:\n"
            f"{context_snippets}\n\n"
            "Answer the question concisely. If the documents are not sufficient, say you don't know."
        )

        answer_text = self.llm_client.generate(prompt)

        return RagResponse(query=query, answer=answer_text, documents=docs)

