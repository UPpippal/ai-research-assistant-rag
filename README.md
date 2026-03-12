
# AI Research Assistant (Hybrid RAG)

An AI-powered research assistant that allows users to upload research papers and ask questions about them.

The system combines **Document Retrieval (RAG)** with **Real-time Web Search** to generate accurate answers with citations.

---

## Features

- Upload multiple research papers (PDF / TXT)
- Ask questions about uploaded documents
- Hybrid RAG (Documents + Internet)
- Real-time web search using Tavily
- Source citations in answers
- ChatGPT-style chat interface
- Built using Streamlit

---

## Architecture

User Question  
↓  
Vector Search (FAISS)  
↓  
Web Search (Tavily)  
↓  
Context Merging  
↓  
Gemini LLM  
↓  
Answer + Sources  

---

## Tech Stack

- Python
- LangChain
- FAISS (Vector Database)
- Google Gemini API
- Tavily Web Search API
- Streamlit UI

---

## Project Structure

# ai-research-assistant-rag
