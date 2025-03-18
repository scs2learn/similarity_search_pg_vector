import requests
import os
import csv
import psycopg2
from dotenv import load_dotenv
import time

import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load .env file
load_dotenv()

def read_text_from_file(file_path):
    print("Inside read_text_from_file() ...")

    """
    Read the text from the text file

    Args:
        filepath (str) : path of the file thats processed

    Returns:
        text (str): text from the file
    
    Raises:
        ExceptionType: None
    
    """
    
    text = ""

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    return text

def get_chunks_as_documents(text, filepath, chunk_size=1000, chunk_overlap=100):
    print(f"inside get_chunks_as_documents() ...")

    """
    Convert the file text into chunks and store them as Documents 

    Args:
        text (str) : text from the file
        filepath (str) : path of the file thats processed
        chunk_size (int) : chunk size
        chunk_overlap (int) : chunk overlap

    Returns:
        text_as_documents (Document): Document containing the chunks and the metadata

    Raises:
        ExceptionType: None
    """
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    metadata = [{"chunk_number": index + 1, "filename": os.path.basename(filepath)} for index, _ in enumerate(chunks)]

    text_as_documents = text_splitter.create_documents(
        chunks,
        metadata
    )
    #print(text_as_documents[:2])

    return text_as_documents

def insert_data_to_pgvector(text_as_documents, document_text_embeddings):
    print("inside insert_data_to_pgvector() ...")

    """
    Insert the vector data into pgvector

    Args:
        text_as_documents (Document): Document containing the chunks and the metadata
        document_text_embeddings (list) : embeddings of the chunjks

    Returns:
        None

    Raises:
        ExceptionType: None
    """

    # Database connection details
    conn_info = {
        'dbname': 'postgres',
        'user': 'postgres',
        'password': 'password',
        'host': 'localhost',
        'port': '5432'
    }

    # Connect to the PostgreSQL database
    conn = psycopg2.connect(**conn_info)
    cursor = conn.cursor()

    # Insert data into the author_quotes table
    insert_query = """
    INSERT INTO sotu (filename, chunk_number, chunk_text, embedding)
    VALUES (%s, %s, %s, %s)
    """

    for document, embedding in zip(text_as_documents, document_text_embeddings):
        cursor.execute(insert_query, (document.metadata["filename"], document.metadata["chunk_number"], document.page_content, embedding))

    # Commit the changes and close the connection
    conn.commit()
    cursor.close()
    conn.close()

def main():
    text_from_file = read_text_from_file('..\\resources\\Trump_2018.txt')
    text_as_documents = get_chunks_as_documents(text_from_file, '..\\resources\\Trump_2018.txt')

    # for i, chunk in enumerate(text_chunks):
    #    print(f"Chunk {i+1}: {chunk}\n---")
    
    print(f"Number of chunks: {len(text_as_documents)}")
    
    embedding_model_name = "BAAI/bge-base-en-v1.5"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    document_text_embeddings = [embeddings.embed_query(document.page_content) for document in text_as_documents]
    #print(np.array(document_text_embeddings).shape)

    if len(text_as_documents) != len(document_text_embeddings):
        print("Mismatch between number of chunks and embeddings")
        return

    insert_data_to_pgvector(text_as_documents, document_text_embeddings)

if __name__ == "__main__":
    main()