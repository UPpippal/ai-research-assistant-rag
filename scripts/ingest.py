from pathlib import Path

from app.config import settings
from app.retrievers.bm25_retriever import load_bm25_from_raw
from app.retrievers.vector_retriever import EmbeddingModel, build_vector_store_from_raw


def main() -> None:
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    settings.bm25_index_dir.mkdir(parents=True, exist_ok=True)
    settings.vector_store_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using raw data directory: {settings.raw_data_dir}")
    print("Scanning for .txt files...")

    txt_files = list(settings.raw_data_dir.rglob("*.txt"))
    if not txt_files:
        example_path = settings.raw_data_dir / "example.txt"
        example_path.write_text(
            "Hybrid Retrieval-Augmented Generation (RAG) combines vector search with keyword search.",
            encoding="utf-8",
        )
        print(f"No .txt files found. Created example file at: {example_path}")

    print("Building BM25 index in-memory...")
    _ = load_bm25_from_raw(settings.raw_data_dir)
    print("BM25 index built.")

    print("Building vector store in-memory...")
    _ = build_vector_store_from_raw(embedding_model=EmbeddingModel(), dir_path=settings.raw_data_dir)
    print("Vector store built.")

    print("Note: This script currently builds indices in-memory only.")
    print("Extend it to persist BM25 and vector indices under data/index and data/vector_store if needed.")


if __name__ == "__main__":
    main()

