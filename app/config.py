from pathlib import Path
from pydantic import BaseSettings


class Settings(BaseSettings):
    project_name: str = "Hybrid RAG Search Engine"

    data_dir: Path = Path("data")
    raw_data_dir: Path = data_dir / "raw"
    bm25_index_dir: Path = data_dir / "index"
    vector_store_dir: Path = data_dir / "vector_store"

    # Retrieval
    bm25_top_k: int = 10
    vector_top_k: int = 10

    # Hybrid weights (0.0 - 1.0)
    bm25_weight: float = 0.5
    vector_weight: float = 0.5

    class Config:
        env_file = ".env"


settings = Settings()

