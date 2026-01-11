import streamlit as st
import os
from dotenv import load_dotenv


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


from src.loaders import *
from src.ingestion import *
from src.database import *

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def main():
    st.set_page_config(page_title="ProjectIQ 2026", layout="wide")
    
    # Session state for DB and History
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = load_vector_db()

    if "history" not in st.session_state:
        st.session_state.history = []

    # Sidebar
    with st.sidebar:
        if st.button("Sync Project Data"):
            docs = get_all_docs()
            chunks = chunk_docs(docs)
            st.session_state.vectorstore = create_vector_db(chunks)
            st.success("Database Updated!")

    # Chat Interface
    st.title("ProjectIQ Assistant")
    
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about your project...")

    if user_input:
        if not st.session_state.vectorstore:
            st.error("Please sync data first!")
            return

        st.session_state.history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
        
            chain = get_rag_chain(st.session_state.vectorstore)
            

            response = chain.invoke(user_input)
            st.markdown(response)

        st.session_state.history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()