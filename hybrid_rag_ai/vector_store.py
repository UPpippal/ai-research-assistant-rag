from langchain_community.vectorstores import FAISS
from hybrid_rag_ai.embeddings import load_embeddings


def create_vector_store(chunks):

    embeddings = load_embeddings()

    vector_db = FAISS.from_documents(chunks, embeddings)

    return vector_db