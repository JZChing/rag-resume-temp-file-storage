import os
import json
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from faker import Faker
from concurrent.futures import ThreadPoolExecutor
from convertpdf import extract_text_from_pdf, extract_text_from_pdf_with_ocr, write_text_to_pdf

# Initialize components
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
faker = Faker()

# File mapping path
MAPPING_FILE = "mapping.json"

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

# Function to load file mapping
def load_mapping():
    """Loads the mapping.json file if it exists, otherwise returns an empty dictionary."""
    if os.path.exists(MAPPING_FILE):
        try:
            with open(MAPPING_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

# Function to update the mapping.json file
def update_mapping(original_file, anonymized_file):
    """Updates mapping.json with the anonymized file reference."""
    file_mapping = load_mapping()  # Load fresh mapping data

    file_mapping[original_file] = anonymized_file  # Add new mapping

    with open(MAPPING_FILE, "w") as f:
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
