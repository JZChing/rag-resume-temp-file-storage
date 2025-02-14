from retriever import load_faiss_index, query_faiss_index, map_results_to_original_file
from llm_agent import generate_llm_response
from ingestdata import process_documents, process_documents_from_csv
from anonymize import anonymize_pdfs_in_batch
from config import PDF_INPUT_DIR as input_directory, PDF_OUTPUT_DIR as output_directory
import os
from convertpdf import process_uploaded_pdfs
from config import FAISS_INDEX_PATH

import spacy
import subprocess

# 🔹 Force Spacy to use 'en_core_web_sm' and prevent 'lg' installation
os.environ["SPACY_MODEL"] = "en_core_web_sm"
os.environ["SPACY_DATA"] = os.path.expanduser("~/.local/share/spacy/models/")

# Remove `en-core-web-lg` if it exists
lg_path = "/home/adminuser/venv/lib/python3.11/site-packages/en_core_web_lg"
if os.path.exists(lg_path):
    print("🛑 Removing en-core-web-lg to prevent errors...")
    subprocess.run(["rm", "-rf", lg_path])

try:
    # Load the small model
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

print("✅ Using 'en_core_web_sm' instead of 'en_core-web-lg'")

os.environ['OPENAI_API_KEY']='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'

def augment_llm_with_vector_data(query, index_path=FAISS_INDEX_PATH, mapping_file="mapping.json"):
    # 🔹 Check if FAISS index exists
    if not os.path.exists(FAISS_INDEX_PATH):
        return {"response": "⚠️ No resume data found. Please upload resumes first!", "sources": []}
    
    faiss_index = load_faiss_index(index_path)
    results = query_faiss_index(faiss_index, query)

    if not results:
        print("⚠️ No relevant results found in FAISS index!")
        return {"response": "I'm sorry, I don't have that information.", "sources": []}
    
    mapped_results = map_results_to_original_file(results, mapping_file)

    print(f"🔍 FAISS Retrieved {len(mapped_results)} results for query: {query}")

    response = generate_llm_response(mapped_results, query)
    sources = [{"original_file": res['original_file'], "relevance_score": res['relevance_score']} for res in mapped_results]
    print("LLM Response:")
    print(response)
    print("\nSources:")

    for res in mapped_results:
        print(f"- {res['original_file']} (Relevance Score: {res['relevance_score']})")
        print(f"Excerpt: {res['text'][:300]}...\n")  # Show a sample of retrieved text

    return {"response": response, "sources": sources}

def run_pipeline(input_directory, output_directory, csv_file):
    if os.path.exists(input_directory):
        anonymize_pdfs_in_batch(input_directory, output_directory)
    else:
        print(f"Input directory not found: {input_directory}")

    process_uploaded_pdfs(output_directory)
    
    process_documents_from_csv(csv_file)
"""
anonymize.py – Handles text and PDF anonymization.
convertpdf.py – Deals with PDF text extraction and OCR.
ingestdata.py – Loads and splits documents.
retriever.py – Implements FAISS vector retrieval.
llm_agent.py – Manages LLM interaction and response generation.
config.py – Stores configurations like API keys and file paths.
main.py – Serves as the main script to orchestrate everything.
"""