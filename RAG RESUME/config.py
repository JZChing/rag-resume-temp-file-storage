OPENAI_API_KEY = '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'
import tempfile
import uuid

SESSION_ID = str(uuid.uuid4())[:8]  # Generate a unique session ID
"""
PDF_INPUT_DIR = f"uploaded_resumes_{SESSION_ID}"  # Unique folder for uploaded resumes
PDF_OUTPUT_DIR = f"processed_resumes_{SESSION_ID}"  # Unique folder for processed resumes
FAISS_INDEX_PATH = f"faiss_index_{SESSION_ID}"  # Unique FAISS index per session
CSV_FILE = f"resumes_data_{SESSION_ID}.csv"  # Unique CSV file per session
"""

# 🔹 Create a temporary base directory (automatically deleted on exit)
temp_dir = tempfile.TemporaryDirectory()

# 🔹 Define session-specific directories inside the temporary folder
PDF_INPUT_DIR = f"{temp_dir.name}/uploaded_resumes_{SESSION_ID}"
PDF_OUTPUT_DIR = f"{temp_dir.name}/processed_resumes_{SESSION_ID}"
FAISS_INDEX_PATH = f"{temp_dir.name}/faiss_index_{SESSION_ID}"
CSV_FILE = f"{temp_dir.name}/resumes_data_{SESSION_ID}.csv"