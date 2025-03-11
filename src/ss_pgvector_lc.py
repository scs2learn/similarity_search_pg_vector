import os
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import PGVector
from langchain_postgres import PGVector

# Load text from a pdf file
def read_pdf_document(file_path):
    pdf_reader = PdfReader(file_path)
    text = ""

    for page in pdf_reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:  # Check if text is extracted successfully
            text += extracted_text + "\n"  # Append text of each page

    return text

# Step 1: Load Text from All PDF Files in a Folder
def load_text_from_pdfs(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            pdf_reader = PdfReader(file_path)
            text = ""

            for page in pdf_reader.pages:
                text += page.extract_text()  # Extract text from each page
            texts.append({"filename": filename, "text": text})
    return texts

def split_pdf_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    end = 0

    while end < len(text):
        end = start + chunk_size
        if end > len(text):
            end = len(text)
        chunks.append(text[start:end])
        start = end - overlap  # Overlap chunks

    return chunks

data_dir = "..\\resources"
pdf_texts = load_text_from_pdfs(data_dir)
# print(pdf_texts)

for pdf in pdf_texts:
    print(f"file '{pdf["filename"]}' has text length {len(pdf["text"])}")

# Step 2: Split Text into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n"]
)

documents = []
for pdf in pdf_texts:
    chunks = text_splitter.split_text(pdf["text"])
    for i, chunk in enumerate(chunks):
        documents.append({
            "filename": pdf["filename"],
            "chunk_number": i,
            "chunk_text": chunk
        })

# Step 3: Initialize HuggingFace Embeddings
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Step 4: Store Metadata and Embeddings in pgVector
connection_string = "postgresql+psycopg2://user:password@localhost:5432/postgres"

# Use the new PGVector implementation from langchain_postgres
vector_db = PGVector(
    connection=connection_string,
    collection_name="my_vector_docs",  # Specify the collection/table name
    embeddings=embeddings
)

# Insert the chunked text, metadata, and embeddings
chunk_texts = [doc["chunk_text"] for doc in documents]
chunk_metadatas = [{"filename": doc["filename"], "chunk_number": doc["chunk_number"]} for doc in documents]
chunk_embeddings = [embeddings.embed_query(text) for text in chunk_texts]

# Store the data in pgVector
vector_db.add_texts(texts=chunk_texts, metadatas=chunk_metadatas)

# Step 5: Query the Vector Data
retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Query example
query = "What are the key points discussed in the documents about Medicare?"
results = retriever.get_relevant_documents(query)

# Print the results
for result in results:
    print(f"Filename: {result.metadata['filename']}")
    print(f"Chunk Number: {result.metadata['chunk_number']}")
    print(f"Chunk Text: {result.page_content}\n")