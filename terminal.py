import os
from dotenv import load_dotenv
load_dotenv(override = True)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


from src.loaders import *
from src.ingestion import *
from src.database import *

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
    print("\n=== ProjectIQ (Terminal Mode) ===\n")

    vectorstore = load_vector_db()
    if not vectorstore:
        print("Vector DB not found")
        choice = input("Do you want to sync project data now? (y/n): ")
        if choice.lower() == "y":
            docs = get_all_docs()
            chunks = chunk_docs(docs)
            vectorstore = create_vector_db(chunks)
            print("Database synced successfully.\n")
        else:
            print("Exiting...")
            return
    chain = get_rag_chain(vectorstore)

    print("Ask questions about your project.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("you : ")
        if user_input.lower() in ["exit", "quit"]:
            print("Good Bye")
            break
        response = chain.invoke(user_input)
        print("\nProjectIQ:", response, "\n")

if __name__ == "__main__":
    main()