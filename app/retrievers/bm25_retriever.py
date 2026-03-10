from pathlib import Path
from typing import Iterable, List, Tuple

from rank_bm25 import BM25Okapi

from app.config import settings


class BM25Retriever:
    """
    Simple BM25 retriever over local text files.

    This is intentionally minimal and keeps all data in memory.
    For production, you would likely swap this for Elasticsearch / OpenSearch.
    """

    def __init__(self, docs: Iterable[Tuple[str, str]]):
        """
        :param docs: iterable of (doc_id, text)
        """
        self.doc_ids: List[str] = []
        self.corpus_tokens: List[List[str]] = []

        for doc_id, text in docs:
            self.doc_ids.append(doc_id)
            tokens = self._tokenize(text)
            self.corpus_tokens.append(tokens)

        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()

    def retrieve(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if not self.bm25 or not self.doc_ids:
            return []

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        return [(self.doc_ids[i], float(scores[i])) for i in ranked_indices]


def load_bm25_from_raw(dir_path: Path | None = None) -> BM25Retriever:
    """
    Convenience loader that builds an in-memory BM25 index
    from all text files in a directory.
    """
    base_dir = dir_path or settings.raw_data_dir
    docs: List[tuple[str, str]] = []

    if not base_dir.exists():
        return BM25Retriever([])

    for path in base_dir.rglob("*.txt"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append((str(path), text))

    return BM25Retriever(docs)

