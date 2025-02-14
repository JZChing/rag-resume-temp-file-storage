import time
import streamlit as st
import os
import shutil
from main import augment_llm_with_vector_data, run_pipeline
from config import PDF_INPUT_DIR as input_directory, PDF_OUTPUT_DIR as output_directory
from config import CSV_FILE, FAISS_INDEX_PATH

st.set_page_config(page_title="Candidate Matching Chatbot", page_icon="🤖", layout="wide")

# 🔹 Custom CSS to Add Sidebar Border
sidebar_border_css = """
<style>
    /* Add a border between sidebar and main content */
    [data-testid="stSidebar"] {
        border-right: 3px solid #30363D;
    }
</style>
"""

st.markdown(sidebar_border_css, unsafe_allow_html=True)

# 🔹 Ensure directories exist
os.makedirs(input_directory, exist_ok=True)
os.makedirs(output_directory, exist_ok=True)

# Sidebar for file upload
st.sidebar.header("📂 Upload Resumes")
uploaded_files = st.sidebar.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

# 🔹 Track processed files in session state
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

new_files = []  # To store newly uploaded file names

if uploaded_files:
    st.sidebar.write(f"📥 {len(uploaded_files)} files uploaded!")

    # Save files and check for new ones
    for uploaded_file in uploaded_files:
        file_path = os.path.join(input_directory, uploaded_file.name)

        if uploaded_file.name not in st.session_state.processed_files:
            new_files.append(uploaded_file.name)  # Mark file as new
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

    # Process only if there are new files
    if new_files:
        st.sidebar.success(f"✅ {len(new_files)} new files detected!")

        with st.spinner("Processing newly uploaded resumes... ⏳"):
            run_pipeline(input_directory, output_directory, CSV_FILE)

        # Mark the new files as processed
        st.session_state.processed_files.update(new_files)
        st.sidebar.success("📊 Newly uploaded resumes processed successfully!")
    else:
        st.sidebar.info("⚠️ No new files detected. Resumes already processed.")


# App title
st.title("🤖 Candidate Matching Chatbot")
st.write("Chat with the AI to find the best candidates for your job requirements!")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Define functions to check for greetings or non-relevant queries
def is_greeting(query):
    greetings = ['hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
    return any(greeting in query.lower() for greeting in greetings)

def is_farewell(query):
    farewells = ['bye', 'goodbye', 'see you', 'take care', 'good night']
    return any(farewell in query.lower() for farewell in farewells)

def is_relevant_question(query):
    relevant_keywords = ['ai', 'knowledge', 'cgpa', 'applicant', 'anyone', 'candidate', 'experience', 'skills', 'education', 'project', 'role', 'intern', 'finance', 'computer science', 'hr', 'it', 'accounting', 'business', 'info', 'human resource', 'management', 'who', 'give', 'summarize', 'summarise']
    return any(keyword in query.lower() for keyword in relevant_keywords)

# Chat container
st.subheader("💬 Chat")

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f"**You:** {chat['query']}")
    with st.chat_message("assistant"):
        st.markdown(f"**Bot:** {chat['response']}")

        # Display sources below the response only for relevant responses
        if chat['sources'] and is_relevant_question(chat['query']):
            sources_text = "📄 Sources:\n"
            for source in chat['sources']:
                sources_text += f"- `{source['original_file']}` (Relevance Score: `{source['relevance_score']}`)\n"
            st.markdown(sources_text)

# User input (chat-style input)
question = st.chat_input("Ask something...")

# If user submits a question
if question:
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(f"**You:** {question}")

    # Check if the question is a greeting or farewell
    if is_greeting(question):
        response_text = "Hello! How can I assist you with candidate matching today?"
        sources = []

    elif is_farewell(question):
        response_text = "Goodbye! Feel free to return if you need more assistance. Have a great day!"
        sources = []

    else:
        # Get response without rerunning document processing
        with st.spinner("Thinking... 🤔"):
            response_data = augment_llm_with_vector_data(question)

        # Extract response and sources
        response_text = response_data['response']
        sources = response_data['sources']

    # Display chatbot response
    with st.chat_message("assistant"):
        st.markdown(f"**Bot:** {response_text}")

        # Display sources below the response only for relevant responses
        if sources and is_relevant_question(question):
            sources_text = "📄 Sources:\n"
            for source in sources:
                sources_text += f"- `{source['original_file']}` (Relevance Score: `{source['relevance_score']}`)\n"
            st.markdown(sources_text)

    # Save chat history with sources
    st.session_state.chat_history.append({"query": question, "response": response_text, "sources": sources})



