import os
import sys
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hybrid_rag_ai.ingestion import load_documents, split_documents
from hybrid_rag_ai.vector_store import create_vector_store
from hybrid_rag_ai.rag_pipeline import ask_question


st.title("AI Research Assistant")

os.makedirs("data", exist_ok=True)


uploaded_files = st.file_uploader(
    "Upload research papers",
    type=["txt", "pdf"],
    accept_multiple_files=True
)


@st.cache_resource
def build_database():

    docs = load_documents("data")
    chunks = split_documents(docs)
    db = create_vector_store(chunks)

    return db


# save uploaded files
if uploaded_files:

    for uploaded_file in uploaded_files:

        file_path = os.path.join("data", uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())


# show indexed documents
st.write("### Indexed Documents")

files = os.listdir("data")

if files:
    for file in files:
        st.write("-", file)

    db = build_database()

else:
    st.write("No documents uploaded yet.")


# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# show previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# user input
question = st.chat_input("Ask a question", key="chatbox")


if question and files:

    # show user message
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    # generate answer
    answer, sources = ask_question(db, question)

    response = answer + "\n\n**Sources:**\n"

    for s in sources:
        response += f"- {s}\n"

    # show AI response
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )