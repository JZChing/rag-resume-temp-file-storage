import json
import os
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from convertpdf import CSV_FILE
import pandas as pd
from config import FAISS_INDEX_PATH

def load_faiss_index(index_path=FAISS_INDEX_PATH):
    return FAISS.load_local(index_path, OpenAIEmbeddings(model="text-embedding-ada-002"), allow_dangerous_deserialization=True)

def query_faiss_index(faiss_index, query, top_k=3, relevance_threshold=0.6):
    results = faiss_index.similarity_search_with_relevance_scores(query, k=top_k)
    # Filter results based on relevance threshold
    filtered_results = [res for res in results if res[1] >= relevance_threshold]

    if not filtered_results:
        return None  # Return None if no results meet the threshold
    
    return filtered_results

course_keywords = [
    "finance", "ai", "artificial intelligence", "human resource", "hr", "management",
    "internship", "business", "computer science", "software engineering", "cybersecurity", 
    "business administration", "marketing", "data science", "economics"
]

def extract_course_from_text(text):
    for keyword in course_keywords:
        if keyword in text.lower():
            return keyword.capitalize()
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
        course = extract_course_from_text(result.page_content)
        mapped_results.append({
            "text": result.page_content,
            "metadata": metadata,
            "original_file": original_file,
            "course": course,
            "relevance_score": score,
        })
    return mapped_results

def count_candidates_from_university(university_name):
    try:
        # Load the CSV file
        df = pd.read_csv(CSV_FILE)

        # Ensure the extracted text column exists
        if "extracted_text" not in df.columns:
            return "Error: 'extracted text' column not found in CSV."

        # Convert university name and extracted text to lowercase for case-insensitive search
        university_name = university_name.lower()

        # Count rows where the university name appears in the extracted text
        count = df[df["extracted_text"].str.lower().str.contains(university_name, na=False)].shape[0]
       
        # Create a filtered DataFrame
        matched_rows = df[df["extracted_text"].str.lower().str.contains(university_name, na=False)]
        
        # Print matched rows for debugging
        print("Matched Rows:\n", matched_rows["extracted_text"])

        return count
    except Exception as e:
        return f"Error loading CSV: {e}"