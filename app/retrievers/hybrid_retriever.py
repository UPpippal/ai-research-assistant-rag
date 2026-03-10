from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from app.config import settings
from app.retrievers.bm25_retriever import BM25Retriever
from app.retrievers.vector_retriever import EmbeddingModel, VectorRetriever


class HybridRetriever:
    """
    Combines BM25 and vector retrieval scores via a weighted sum.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: VectorRetriever,
        embedding_model: EmbeddingModel,
        bm25_weight: float | None = None,
        vector_weight: float | None = None,
    ) -> None:
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.embedding_model = embedding_model

        self.bm25_weight = bm25_weight or settings.bm25_weight
        self.vector_weight = vector_weight or settings.vector_weight

        total = self.bm25_weight + self.vector_weight
        if total <= 0:
            self.bm25_weight = 0.5
            self.vector_weight = 0.5
        else:
            self.bm25_weight /= total
            self.vector_weight /= total

    def retrieve(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        bm25_results = self.bm25.retrieve(query, top_k=settings.bm25_top_k)

        query_embedding = self.embedding_model.embed([query])[0]
        vector_results = self.vector.retrieve(
            query_embedding=query_embedding, top_k=settings.vector_top_k
        )

        score_map: Dict[str, float] = {}

        def accumulate(results: Iterable[Tuple[str, float]], weight: float) -> None:
            for doc_id, score in results:
                if doc_id not in score_map:
                    score_map[doc_id] = 0.0
                score_map[doc_id] += weight * score

        accumulate(bm25_results, self.bm25_weight)
        accumulate(vector_results, self.vector_weight)

        ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]

