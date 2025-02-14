import os
import csv
import subprocess
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import faiss
from openai import OpenAI
from langchain.schema import Document
from config import FAISS_INDEX_PATH


# Environment configuration
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Function to load PDF documents and extract metadata
def load_documents(pdf_folder):
    all_pages_with_metadata = []
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):  # Check for PDF files
            file_path = os.path.join(pdf_folder, file_name)
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            for page in pages:
                page.metadata = {"file_name": file_name}
            all_pages_with_metadata.extend(pages)
    return all_pages_with_metadata

# Function to load resume text from CSV and apply optimized chunking
def load_documents_from_csv(csv_file, chunk_size=1500, chunk_overlap=300):
    """Load resume text from CSV and split into optimized chunks."""
    all_documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    with open(csv_file, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            extracted_text = row["extracted_text"].strip()
            chunks = text_splitter.split_text(extracted_text)  # Split resume text

            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={"resume_id": row["resume_id"], "file_name": row["file_name"]}
                )
                all_documents.append(doc)

    return all_documents  # Return chunked documents

# Function to split text into chunks
def split_text(all_pages_with_metadata, chunk_size=4000, chunk_overlap=400):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_documents = []

    for page in all_pages_with_metadata:
        chunks = text_splitter.split_text(page.page_content)  # Split text into chunks

        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata=page.metadata  # Retain original file metadata
            )
            all_documents.append(doc)

    return all_documents  # Return list of `Document` objects

# Function to track token usage and save to CSV
def track_token_usage(texts, embedding_model, csv_filename="embedding_usage.csv"):
    client = OpenAI()
    usage_data = []
    for text in texts:
        response = client.embeddings.create(input=text, model="text-embedding-ada-002")
        token_usage = response.usage.total_tokens  # Get token usage
        usage_data.append([text, token_usage])
    with open(csv_filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Text", "Token Usage"])
        writer.writerows(usage_data)
    print(f"Token usage recorded in {csv_filename}")

# Function to create FAISS index
def create_faiss_index(all_documents):
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    texts = [doc.page_content for doc in all_documents]  # Fix here
    metadatas = [doc.metadata for doc in all_documents]  # Fix here
    
    track_token_usage(texts, embedding_model)  # Track API token usage
    
    faiss_index = FAISS.from_texts(texts, embedding=embedding_model, metadatas=metadatas)
    faiss_index.save_local(FAISS_INDEX_PATH)  # Save index for future use
    print("FAISS index saved successfully.")
    return FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)

# Example function to process PDF folder
def process_documents(pdf_folder):
    """Load PDFs, split text, and create FAISS index."""
    all_pages = load_documents(pdf_folder)  # Extract pages with metadata
    all_documents = split_text(all_pages)  # Convert to Document objects
    faiss_index = create_faiss_index(all_documents)  # Index using FAISS
    return faiss_index

# Process CSV and create FAISS index
def process_documents_from_csv(csv_file):
    """Process resumes from CSV, apply chunking, and create FAISS index."""
    all_documents = load_documents_from_csv(csv_file)  # Load & chunk text
    faiss_index = create_faiss_index(all_documents)  # Index using FAISS
    return faiss_index

# Verify the FAISS index by searching for a query
def test_faiss_index(faiss_index, query="Example query to test the FAISS index"):
    results = faiss_index.similarity_search_with_relevance_scores(query, k=5)
    for result in results:
        print(result)