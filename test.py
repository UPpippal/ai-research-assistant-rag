from hybrid_rag_ai.ingestion import load_documents, split_documents

docs = load_documents("data")
chunks = split_documents(docs)

print(len(chunks))
print(chunks[0].page_content)