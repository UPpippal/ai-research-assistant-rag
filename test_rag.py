from hybrid_rag_ai.ingestion import load_documents, split_documents
from hybrid_rag_ai.vector_store import create_vector_store

docs = load_documents("data")
chunks = split_documents(docs)

db = create_vector_store(chunks)

results = db.similarity_search("What is RAG?")
print(results[0].page_content)