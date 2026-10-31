import json
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Initialize BGE-small and ChromaDB
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = Chroma(
    persist_directory="./crm_chroma_db", 
    embedding_function=embedding_model
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Retrieve context for the query
query = "What is the return shipping fee for a change of mind, and how long is the return window?"
retrieved_docs = retriever.invoke(query)
context = "\n".join([doc.page_content for doc in retrieved_docs])

# Standard System Prompt for LangGraph JSON Extraction
system_prompt = f"""You are a CRM data extraction assistant. Answer the user's query based on the provided context.
Return the extracted information strictly in JSON format.

Context:
{context}"""

# Payload for local Ollama API (Qwen-CRM)
payload = {
    "model": "qwen-crm",
    "prompt": f"{system_prompt}\n\nQuery: {query}",
    "stream": False,
    "format": "json" 
}

# Execute the pipeline
response = requests.post("http://localhost:11434/api/generate", json=payload)

if response.status_code == 200:
    print("Final JSON Output:")
    print(response.json()["response"])
else:
    print(f"Error connecting to Ollama: {response.status_code}")