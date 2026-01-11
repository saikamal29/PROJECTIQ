from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.loaders import documents_list, load_excel, load_pdf, load_word
from pathlib import Path

# [".docx", ".pdf", ".xlsx"])


def get_all_docs():
    """
    Scans a folder for supported document types and loads them into Document objects.
    Args:
        folder_path (str): Path to the folder containing documents.
    Returns:
        list[Document]: A combined list of Document objects from all supported files in the folder.
    """
    folder_path = str(Path(__file__).parent.parent / "data")
    docs_dict = documents_list(folder_path)
    docs_doc_list = []
    for file_type in docs_dict.keys():
        if file_type == ".docx":
            for word_doc_path in docs_dict[file_type]:
                docs_doc_list.extend(load_word(word_doc_path))
        elif file_type == ".pdf":
            for file_path in docs_dict[file_type]:
                docs_doc_list.extend(load_pdf(file_path))
        elif file_type == ".xlsx":
            for file_path in docs_dict[file_type]:
                docs_doc_list.extend(load_excel(file_path))
    return docs_doc_list


def chunk_docs(docs_list):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    split_docs = text_splitter.split_documents(docs_list)
    return split_docs
