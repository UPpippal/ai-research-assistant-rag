from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import faiss  # type: ignore
import numpy as np

from app.config import settings


class EmbeddingModel:
    """
    Minimal embedding interface.

    Replace the `embed` method with your actual model
    (e.g. OpenAI, local transformer, etc.).
    """

    def embed(self, texts: List[str]) -> np.ndarray:
        # TODO: Plug in a real embedding model.
        # For now, this is a deterministic dummy embedding.
        rng = np.random.default_rng(seed=0)
        return rng.normal(size=(len(texts), 384)).astype("float32")


@dataclass
class VectorStoreEntry:
    id: str
    vector: np.ndarray


class VectorRetriever:
    def __init__(
        self,
        entries: Iterable[VectorStoreEntry],
        dim: int,
    ) -> None:
        self.ids: List[str] = []
        vectors: List[np.ndarray] = []

        for entry in entries:
            self.ids.append(entry.id)
            vectors.append(entry.vector.astype("float32"))

        if vectors:
            self.index = faiss.IndexFlatIP(dim)
            matrix = np.vstack(vectors)
            faiss.normalize_L2(matrix)
            self.index.add(matrix)
        else:
            self.index = None

    def retrieve(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        if self.index is None or not self.ids:
            return []

        q = query_embedding.astype("float32")
        faiss.normalize_L2(q)
        if q.ndim == 1:
            q = q[None, :]

        scores, indices = self.index.search(q, top_k)
        scores = scores[0]
        indices = indices[0]

        results: List[Tuple[str, float]] = []
        for idx, score in zip(indices, scores):
            if idx < 0 or idx >= len(self.ids):
                continue
            results.append((self.ids[idx], float(score)))
        return results


def build_vector_store_from_raw(
    embedding_model: EmbeddingModel | None = None,
    dir_path: Path | None = None,
) -> VectorRetriever:
    base_dir = dir_path or settings.raw_data_dir
    embedding_model = embedding_model or EmbeddingModel()

    if not base_dir.exists():
        return VectorRetriever(entries=[], dim=384)

    texts: List[str] = []
    ids: List[str] = []

    for path in base_dir.rglob("*.txt"):
        ids.append(str(path))
        texts.append(path.read_text(encoding="utf-8", errors="ignore"))

    if not texts:
        return VectorRetriever(entries=[], dim=384)

    vectors = embedding_model.embed(texts)
    dim = vectors.shape[1]

    entries = [
        VectorStoreEntry(id=doc_id, vector=vec)
        for doc_id, vec in zip(ids, vectors)
    ]
    return VectorRetriever(entries=entries, dim=dim)

