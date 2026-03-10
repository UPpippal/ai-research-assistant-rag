import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOP_K = 4

VECTOR_DB_PATH = "vector_store/faiss_index"