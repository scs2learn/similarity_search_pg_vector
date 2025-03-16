import requests
import os
import csv
import psycopg2
from dotenv import load_dotenv
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_postgres import PGVector

# Load .env file
load_dotenv()

def read_csv_skip_header(file_path):
    print("Inside read_csv_skip_header() ...")
    """
    Read the csv file while skipping the header row

    Args:
        file_path (str): Filename with full path

    Returns:
        quotes (list), authors (list), categories (list): Returns these data

    Raises:
        ExceptionType: None
    """

    quotes = []
    authors = []
    categories = []

    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)

        # Skip the header row
        next(reader)
        
        for row in reader:
            if row:
                quotes.append(row[0])
                authors.append(row[1])
                categories.append(row[2])
    return quotes, authors, categories

def insert_data_to_pgvector(quotes, authors, categories, embeddings):
    print("inside insert_into_db() ...")

    """
    Insert the vector data into pgvector

    Args:
        quotes (str): quotes text
        authors (str) : author name
        categories (str) : category of the quotes
        embeddings : embeddings of the quote text

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
    INSERT INTO author_quotes (quotes, authors, categories, embedding)
    VALUES (%s, %s, %s, %s)
    """
    for quote, author, category, embedding in zip(quotes, authors, categories, embeddings):
        cursor.execute(insert_query, (quote, author, category, embedding))

    # Commit the changes and close the connection
    conn.commit()
    cursor.close()
    conn.close()

def main():
    quotes, authors, categories = read_csv_skip_header('..\\resources\\quotes_updated_5k.csv')

    # Initialize HuggingFace Embeddings
    embedding_model_name = "sentence-transformers/stsb-bert-large"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    quotes_embeddings = [embeddings.embed_query(quote) for quote in quotes]
    
    if len(quotes) != len(quotes_embeddings):
        print("Mismatch between number of quotes and embeddings")
        return
    
    insert_data_to_pgvector(quotes, authors, categories, quotes_embeddings)

if __name__ == "__main__":
    main()