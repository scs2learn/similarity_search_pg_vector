import requests
import sys
import os
import json
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load .env file
load_dotenv()

def get_vectorized_quotes(input_embedding):
    print(f"Inside get_vectorized_quotes() ...")
    
    """
    Brief summary of the function.

    Args:
        input_embedding (List): Contains the embeddings of the user input.

    Returns:
        str: Metadata for matching the embeddings

    Raises:
        ExceptionType: None
    """
    
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
   
    cursor.execute(f"SELECT * FROM author_quotes ORDER BY embedding <-> '{input_embedding}' LIMIT 2;")
    
    # Fetch the results
    quotes = cursor.fetchall()

    # Extract and print the quote text and author name
    retrieved_quotes = []
    for item in quotes:
        quote_text = item[1]
        author_name = item[2]
        quote_category = item[3]
        retrieved_quotes.append(f'"{quote_text}" - {author_name}')

    # Commit the transaction
    conn.commit()

    # Close the cursor
    cursor.close()

    print(f"Data from db : {retrieved_quotes}")

    return retrieved_quotes

def get_query_response(user_query, retrieved_quotes):
    print(f"Inside get_query_response() ...")
    """
    Retrieves the LLM response for the user query and the context data.

    Args:
        user_query (str): Contains the user input.
        retrieved_quotes (str) : contains the context from pgvector

    Returns:
        str: Metadata for matching the embeddings

    Raises:
        ExceptionType: None
    """

    hf_llm = HuggingFaceEndpoint(
        model="meta-llama/Llama-3.2-3B-Instruct",
        task="text-generation",
        temperature=0.6,
        max_new_tokens=512,
        top_p=0.9,
        huggingfacehub_api_token=os.getenv("HF_TOKEN")
    )

    # Prepare the input prompt
    prompt = (
        "You're a helpful assistant. You are helping people get motivated. Here are some quotes related to your input:\n"
        + "\n".join(retrieved_quotes)
        + "\nPlease provide a response related to the input and consider the above quotes."
        + f"\n\nUser Query: {user_query}"
    )

    # Generate the response directly
    llm_response = hf_llm.invoke(prompt)

    return llm_response

def main():
   if len(sys.argv) < 2:
       print("Usage: python app.py 'Your query goes here'")
       return
  
   user_input = sys.argv[1]
  
   # Get the embedding vector for the input text
   embedding_model_name = "sentence-transformers/stsb-bert-large"
   embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
   user_input_embeddings = embeddings.embed_query(user_input)

   if embeddings:
       # Get quotes similar to the input text
       retrieved_quotes = get_vectorized_quotes(user_input_embeddings)
       if retrieved_quotes:
           # Generate and print the response
           print(get_query_response(user_input, retrieved_quotes))
       else:
           print("No quotes retrieved from the database.")
   else:
       print("Failed to get embedding vector")

if __name__ == "__main__":
   main()
