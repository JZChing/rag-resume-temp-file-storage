#!/usr/bin/env python
# coding: utf-8

import subprocess

# Install required packages
subprocess.run(["pip", "install", "langchain", "langchain-experimental", "openai", 
                "presidio-analyzer", "presidio-anonymizer", "PyPDF2", "reportlab", 
                "pdfplumber", "fpdf", "faker", "pdf2image", "pytesseract"])

subprocess.run(["pip", "install", "PyMuPDF", "spacy"])
subprocess.run(["pip", "install", "pypdf", "langchain_openai", "langchain_community", "faiss-cpu", "--quiet"])
subprocess.run(["pip", "install", "streamlit"])

# Download the Spacy model
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"])



# Masking by redacting

# In[3]:


import os
import json
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from faker import Faker
import pdfplumber
from fpdf import FPDF
from concurrent.futures import ThreadPoolExecutor
from pdf2image import convert_from_path
import pytesseract

# Initialize components
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
faker = Faker()

# File mapping path
mapping_file = "mapping.json"

# Load existing file mapping or initialize a new one
if os.path.exists(mapping_file):
    with open(mapping_file, "r") as f:
        try:
            file_mapping = json.load(f)
        except json.JSONDecodeError:
            file_mapping = {}
else:
    file_mapping = {}

# Add a custom recognizer for addresses with more flexible patterns
address_recognizer = PatternRecognizer(
    supported_entity="ADDRESS",
    patterns=[
        Pattern(
            name="address_pattern",
            regex=r"\d{1,5}[\w\s,]*(Jln|Jalan|Kg|Kampung|Lorong|Taman|Bukit|Sungai|Bandar|Persiaran|Desa|Pangsapuri|Residensi)[\w\s,]*\d{0,6}[\w\s,]*",
            score=1.0
        )
    ]
)
analyzer.registry.add_recognizer(address_recognizer)

# Add a custom recognizer for phone numbers with broader regex
phone_number_recognizer = PatternRecognizer(
    supported_entity="PHONE_NUMBER",
    patterns=[
        Pattern(
            name="phone_number_pattern",
            regex=r"(\+?\d{1,3}[-.\s]?)?(1[02-9]|[2-9]\d{2})[-.\s]?\d{3}[-.\s]?\d{4}|\b\d{10,15}\b",
            score=1.0
        )
    ]
)
analyzer.registry.add_recognizer(phone_number_recognizer)

# Function to anonymize text
def anonymize_text(text):
    results = analyzer.analyze(
        text=text,
        entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION", "ADDRESS"],
        language="en",
    )

    operators = {
        "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[REDACTED EMAIL]"}),
        "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[REDACTED PHONE NO]"}),
        "LOCATION": OperatorConfig("replace", {"new_value": "[REDACTED ADDRESS]"}),
        "ADDRESS": OperatorConfig("replace", {"new_value": "[REDACTED ADDRESS]"}),
    }

    anonymized_result = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators,
    )
    return anonymized_result.text

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "".join(page.extract_text() + "\n" for page in pdf.pages if page.extract_text())
    return text

def extract_text_from_pdf_with_ocr(pdf_path):
    images = convert_from_path(pdf_path)
    return "".join(pytesseract.image_to_string(image) + "\n" for image in images)

# Function to write anonymized text back to a PDF
def write_text_to_pdf(text, output_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.add_font("DejaVu", "", r"C:\Users\hyper\Documents\RAG Resume\dejavu-fonts-ttf-2.37\dejavu-fonts-ttf-2.37\ttf\DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=12)

    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf.output(output_path, "F")

# Function to update the mapping.json file
def update_mapping(original_file, anonymized_file):
    file_mapping[original_file] = anonymized_file
    with open(mapping_file, "w") as f:
        json.dump(file_mapping, f, indent=4)

# Function to anonymize a single PDF and store mapping
def anonymize_pdf(file_name, input_directory, output_directory):
    input_path = os.path.join(input_directory, file_name)
    anonymized_file_name = f"anonymized_{file_name}"
    output_path = os.path.join(output_directory, anonymized_file_name)

    try:
        extracted_text = extract_text_from_pdf(input_path)
        if not extracted_text.strip():  # Use OCR if no text is found
            extracted_text = extract_text_from_pdf_with_ocr(input_path)

        anonymized_text = anonymize_text(extracted_text)
        write_text_to_pdf(anonymized_text, output_path)

        # Update the file mapping
        update_mapping(input_path, output_path)

        print(f"Successfully anonymized: {file_name}")
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# Main function to process all PDFs
def anonymize_pdfs_in_batch(input_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    pdf_files = [f for f in os.listdir(input_directory) if f.endswith(".pdf")]

    with ThreadPoolExecutor(max_workers=4) as executor:
        for file_name in pdf_files:
            executor.submit(anonymize_pdf, file_name, input_directory, output_directory)

    print("Batch anonymization complete.")

# Example usage
input_directory = r"C:\Users\hyper\Documents\RAG Resume\resumes"
output_directory = r"C:\Users\hyper\Documents\RAG Resume\anonymized_resumes"

#main function
if os.path.exists(input_directory):
    anonymize_pdfs_in_batch(input_directory, output_directory)
else:
    print(f"Input directory not found: {input_directory}")


# # RAG

# Set up OpenAI API **KEY**

# In[4]:


import os
os.environ['OPENAI_API_KEY']='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'


# Define LLM

# In[5]:


from langchain_openai import ChatOpenAI
llm= ChatOpenAI(model='gpt-4')


# Parse and load pdf documents

# In[6]:


import os
from langchain_community.document_loaders import PyPDFLoader

# Directory containing PDFs
pdf_folder = r"C:\Users\hyper\Documents\RAG Resume\anonymized_resumes"

# Initialize a list to store all pages with metadata
all_pages_with_metadata = []

# Iterate through all files in the folder
for file_name in os.listdir(pdf_folder):
    if file_name.endswith(".pdf"):  # Check for PDF files
        file_path = os.path.join(pdf_folder, file_name)

        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(file_path)

        # Load pages
        pages = loader.load()

        # Add metadata (e.g., file_name) to each page
        for page in pages:
            page.metadata = {"file_name": file_name}  # Add file name as metadata

        # Store pages with metadata
        all_pages_with_metadata.extend(pages)

# Now each page in `all_pages_with_metadata` has metadata linking it to the original PDF


# In[7]:


all_pages_with_metadata


# TEXT SPLITTING

# In[8]:


from langchain_text_splitters import CharacterTextSplitter
# Initialize the text splitter
text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=400)

# Initialize a list to store document chunks with metadata
all_chunks_with_metadata = []

# Split each page into smaller chunks
for page in all_pages_with_metadata:
    # Split the page content into chunks
    chunks = text_splitter.split_text(page.page_content)

    # Add metadata to each chunk
    for chunk in chunks:
        all_chunks_with_metadata.append({
            "text": chunk,
            "metadata": page.metadata  # Retain the original metadata
        })

# Now `all_chunks_with_metadata` contains the text chunks with their associated metadata

# Print an example of a chunk with metadata
all_chunks_with_metadata[:5] # Print the first 5 chunks for verification


# In[9]:


len(all_chunks_with_metadata)


# Load documents into knowledge base

# In[10]:


subprocess.run(["pip", "install", "-U", "langchain-openai"])


# In[11]:


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# In[12]:


import csv
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from openai import OpenAI

# Initialize the embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Extract texts and metadata from the chunks
texts = [chunk["text"] for chunk in all_chunks_with_metadata]
metadatas = [chunk["metadata"] for chunk in all_chunks_with_metadata]

# Function to track token usage and save to CSV
def track_token_usage(texts, embedding_model, csv_filename="embedding_usage.csv"):
    client = OpenAI()
    usage_data = []

    for text in texts:
        response = client.embeddings.create(input=text, model="text-embedding-ada-002")
        token_usage = response.usage.total_tokens  # Get token usage
        usage_data.append([text, token_usage])

    # Save token usage data to CSV
    with open(csv_filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Text", "Token Usage"])
        writer.writerows(usage_data)

    print(f"Token usage recorded in {csv_filename}")

# Track API token usage
track_token_usage(texts, embedding_model)

# Create the FAISS index from texts and embeddings
faiss_index = FAISS.from_texts(
    texts=texts,
    embedding=embedding_model,
    metadatas=metadatas
)

# Save the FAISS index to disk for later use (optional)
faiss_index.save_local("faiss_index")

faiss_index = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Verify the FAISS index by searching for a query
query = "Example query to test the FAISS index"
results = faiss_index.similarity_search(query, k=1)

# Print the top 5 results
for result in results:
    print(result)


# Similarity search with query

# In[13]:


question = "Give me the candidates having experience in Java, .Net, SQL programming languages"
docs = faiss_index.similarity_search_with_relevance_scores(
    question, k=5)


# In[14]:


docs


# In[23]:


import json
from langchain_community.llms import Ollama

llm = Ollama(model="deepseek-r1:8b")

# Function to load FAISS index
def load_faiss_index(index_path="faiss_index"):
    return FAISS.load_local(index_path, OpenAIEmbeddings(model="text-embedding-ada-002"), allow_dangerous_deserialization=True)

# Function to perform a similarity search and retrieve metadata
def query_faiss_index(faiss_index, query, top_k=4):
    results = faiss_index.similarity_search_with_relevance_scores(query, k=top_k)
    return results

import os

course_keywords = [
    "finance", "ai", "artificial intelligence", "human resource", "hr", "management",
    "internship", "business", "computer science", "software engineering", "cybersecurity", 
    "business administration", "marketing", "data science", "economics"
]

def extract_course_from_text(text):

    for keyword in course_keywords:
        if keyword in text.lower():
            return keyword.capitalize()  # Standardize capitalization
    return "Unknown"


def map_results_to_original_file(results, mapping_file="mapping.json"):
    with open(mapping_file, "r") as f:
        file_mapping = json.load(f)

    reversed_mapping = {os.path.basename(v): os.path.basename(k) for k, v in file_mapping.items()}

    mapped_results = []
    for result, score in results:
        metadata = result.metadata
        anonymized_filename = os.path.basename(metadata.get("file_name", "Unknown"))
        original_file = reversed_mapping.get(anonymized_filename, "Unknown")

        course = extract_course_from_text(result.page_content)  # Extract course info

        mapped_results.append({
            "text": result.page_content,
            "metadata": metadata,
            "original_file": original_file,
            "course": course,  # Store extracted course
            "relevance_score": score,
        })

    return mapped_results



def generate_llm_response(mapped_results, question):
    llm = Ollama(model="deepseek-r1:8b")

    question_lower = question.lower()
    
     # Detect "how many" queries
    if "how many" in question_lower:
        # Extract course name from query
        course_name = next((keyword for keyword in course_keywords if keyword in question_lower), None)
        
        if course_name:
            # Count unique candidates pursuing the specified course
            unique_candidates = {res["original_file"] for res in mapped_results if res["course"].lower() == course_name.lower()}
            count = len(unique_candidates)

            if count > 0:
                return f"There are **{count}** candidates pursuing **{course_name}**."
            else:
                return f"Sources list possible candidates pursuing **{course_name}**."
        else:
            return f"Sources list possible candidates pursuing **{course_name}**."

    # Greeting & Farewell Handling
    greetings = ["hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    farewells = ["bye", "goodbye", "see you", "take care", "good night"]

    # Keywords to detect relevant queries
    relevant_keywords = ["ai", "candidate", "experience", "skills", "education", "project", "role", "intern", "give",
                        "finance", "computer science", "hr", "it", "accounting", "business", "cgpa", "applicant", "anyone", "info", "human resource", "management", "who"]

    # Check for relevant keywords first
    if not any(keyword in question_lower for keyword in relevant_keywords):
        if any(greeting in question_lower for greeting in greetings):
            return "Hello! How can I assist you with candidate matching today?"
        
        if any(farewell in question_lower for farewell in farewells):
            return "Goodbye! Feel free to return if you need more assistance. Have a great day!"
        
        return "I'm here to help with candidate matching. Please ask about the candidates, their skills, experience, or roles."

    # 🔹 Step 1: Check if the query is a name search
    words = question.lower().split()
    name_match = next((word for word in words if len(word) > 2 and word[0].isupper()), None)

    if name_match:
        # 🔹 Filter mapped_results to include only candidates whose names match
        filtered_candidates = [res for res in mapped_results if name_match.lower() in res["text"].lower()]
        
        if not filtered_candidates:
            return f"No candidate found with the name **{name_match}**."
        
        sources = "\n".join([f"- {res['original_file']}" for res in filtered_candidates])
        context = "\n\n".join([res["text"] for res in filtered_candidates])
    
    else:
        # 🔹 If it's a general query, use all results
        sources = "\n".join([f"- {res['original_file']}" for res in mapped_results])
        context = "\n\n".join([res["text"] for res in mapped_results])

    # 🔹 Step 2: Generate Response with LLM
    prompt = f"""
    You are an AI assistant helping HR managers find the best candidates. Answer concisely using bullet points.
    Do not mixed up different candidates information.
    if user ask irrelevant question that you dont have answer based on the files, you can just say "I'm sorry, I don't have that information."
    
    **Query:** '{question}'

    **Example Response Format:**
    - **Candidate Name:** john 
    - **University:**  university of malaya
    - **Course:** computer science/accounting/human management/psychology/artificial intelligence and etc (if available)
    - **Experience:** (years & relevant roles)
    - **Education:** spm/bachelor degree/diploma and etc (if available)
    - **Result:** (CGPA or GPA) (if available)
    - **Skills:** (Key relevant skills only) eg. machine learning, accounting, financial planning
    - **Projects** (Mention briefly) eg, whatever system
    **this is just an example

    Skip irrelevant details and avoid redundant phrases.
    **Return only the final response in bullet points, without unnecessary explanations or reasoning.**  

    **Sources:**
    {sources}

    **Context:**
    {context}
    """
    return llm(prompt)


# Main Function
def augment_llm_with_vector_data(query, index_path="faiss_index", mapping_file="mapping.json"):
    # Load FAISS index
    faiss_index = load_faiss_index(index_path)

    # Perform similarity search
    results = query_faiss_index(faiss_index, query)

    # Map results back to original file
    mapped_results = map_results_to_original_file(results, mapping_file)

    # Generate response with LLM
    response = generate_llm_response(mapped_results, query)

    # Prepare the response to include both response and sources
    sources = [{"original_file": res['original_file'], "relevance_score": res['relevance_score']} for res in mapped_results]

    # Display response and sources
    print("LLM Response:")
    print(response)

    print("\nSources:")
    for res in mapped_results:
        print(f"- {res['original_file']} (Relevance Score: {res['relevance_score']})")

    return {
        "response": response,
        "sources": sources
    }


# Example usage
question = "how many finance candidates"
augment_llm_with_vector_data(question)


# DONE UNTIL HERE
