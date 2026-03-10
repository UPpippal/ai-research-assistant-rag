from hybrid_rag_ai.llm import load_llm
from hybrid_rag_ai.web_search import search_web


def ask_question(db, query):

    llm = load_llm()

    docs = db.similarity_search(query, k=3)

    doc_context = ""
    doc_sources = []

    for d in docs:
        doc_context += d.page_content + "\n"
        doc_sources.append(d.metadata.get("source", "document"))

    # web search
    web_context, web_sources = search_web(query)

    context = doc_context + web_context

    prompt = f"""
You are an AI research assistant.

Answer the question using the provided context.

If the context does not contain enough information,
use general knowledge but prefer the context.

Context:
{context}

Question:
{query}

Give a clear and structured answer.
"""

    response = llm.invoke(prompt)

    # remove duplicate sources
    sources = list(set(doc_sources + web_sources))

    return response.content, sources