import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="my_collection",
    )
    return vectorstore


def load_vector_db():
    if os.path.exists("./chroma_db"):
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            collection_name="my_collection",
        )
        return vectorstore
    else:
        return None
