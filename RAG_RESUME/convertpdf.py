import os
import csv
import uuid  # For generating unique resume IDs
import pandas as pd
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from fpdf import FPDF
from config import CSV_FILE

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using pdfplumber, ensuring all pages are captured."""
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return ""

    extracted_text = []  # Use a list to store text from all pages

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text.append(text.strip())  # Strip any unnecessary whitespace

        full_text = "\n".join(extracted_text)  # Join all extracted text properly
        return full_text if full_text.strip() else ""  # Ensure empty string if no text is found

    except Exception as e:
        print(f"⚠️ Error extracting text from {pdf_path}: {e}")
        return ""


def extract_text_from_pdf_with_ocr(pdf_path):
    """Extract text from a scanned PDF using OCR."""
    try:
        images = convert_from_path(pdf_path)
        return "\n".join(pytesseract.image_to_string(image) for image in images)
    except Exception as e:
        print(f"⚠️ OCR extraction failed for {pdf_path}: {e}")
        return ""

def write_text_to_pdf(text, output_path):
    """Write extracted/anonymized text back to a new PDF file."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.add_font("DejaVu", "", r"C:\Users\hyper\Documents\RAG_RESUME\dejavu-fonts-ttf-2.37\ttf\DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=12)

    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf.output(output_path, "F")

def save_text_to_csv(resume_id, file_name, text):
    """Save extracted text to a CSV file with Resume ID."""
    file_exists = os.path.isfile(CSV_FILE)

    # Check if CSV exists and load existing data
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        
        # Check if this resume is already stored (based on file name or resume ID)
        if (df["file_name"] == file_name).any():
            print(f"⚠️ Skipping duplicate: {file_name} (already exists in CSV)")
            return  # Skip saving duplicate

    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["resume_id", "file_name", "extracted_text"])  # Header row
        
        writer.writerow([resume_id, file_name, text])

def process_uploaded_pdfs(pdf_folder):
    """Process uploaded PDFs and save extracted text to CSV."""
    pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"❌ No PDFs found in directory: {pdf_folder}")
        return None

    for pdf_path in pdf_files:
        if not os.path.exists(pdf_path):
            print(f"❌ Skipping missing file: {pdf_path}")
            continue

        resume_id = str(uuid.uuid4())[:8]  # Generate a short unique ID
        file_name = os.path.basename(pdf_path)

        try:
            extracted_text = extract_text_from_pdf(pdf_path)
            if not extracted_text.strip():  
                print(f"🔍 No text found in {file_name}, using OCR...")
                extracted_text = extract_text_from_pdf_with_ocr(pdf_path)

            save_text_to_csv(resume_id, file_name, extracted_text)
            print(f"✅ Processed {file_name} with ID {resume_id}")

        except Exception as e:
            print(f"⚠️ Error processing {file_name}: {e}")

    return CSV_FILE  # Return CSV path for next step
