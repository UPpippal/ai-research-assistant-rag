from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


def load_documents(folder_path):

    documents = []

    for file in os.listdir(folder_path):

        file_path = os.path.join(folder_path, file)

        if file.endswith(".txt"):

            loader = TextLoader(file_path)

        elif file.endswith(".pdf"):

            loader = PyPDFLoader(file_path)

        else:
            continue

        documents.extend(loader.load())

    return documents


def split_documents(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    return splitter.split_documents(docs)