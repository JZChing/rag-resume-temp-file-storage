import faiss
import pickle
import os
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "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"  # Set API key
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"]) 
# Load FAISS index
index_path = "faiss_index"
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

if os.path.exists(index_path):
    faiss_index = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    print(f"FAISS index loaded successfully with {faiss_index.index.ntotal} entries.")

    # Retrieve metadata
    stored_texts = faiss_index.docstore._dict  # Raw stored data
    for i, (doc_id, doc) in enumerate(stored_texts.items()):
        print(f"\n📌 **Entry {i+1}:**")
        print(f"🔹 Resume ID: {doc.metadata.get('resume_id', 'Unknown')}")
        print(f"🔹 File Name: {doc.metadata.get('file_name', 'Unknown')}")
        print(f"🔹 Text Preview: {doc.page_content[:300]}...")  # Show only a snippet
else:
    print("FAISS index file not found.")

