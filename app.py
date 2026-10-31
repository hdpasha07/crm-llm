import streamlit as st
import requests
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

st.set_page_config(page_title="Local AI CRM", layout="centered")
st.title("Local AI CRM Extraction Engine")

# Cache the heavy embedding model so the app runs fast
@st.cache_resource
def load_retriever():
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_store = Chroma(persist_directory="./crm_chroma_db", embedding_function=embedding_model)
    return vector_store.as_retriever(search_kwargs={"k": 3})

retriever = load_retriever()

query = st.text_input("Enter CRM Query:", placeholder="e.g., What is the refund policy?")

if st.button("Extract JSON"):
    with st.spinner("Retrieving context and querying Qwen..."):
        # 1. Retrieve Context
        retrieved_docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        
        # 2. Build Prompt
        system_prompt = f"""You are a CRM data extraction assistant. Answer the user's query based on the provided context.
Return the extracted information strictly in JSON format.

Context:
{context}"""
        
        payload = {
            "model": "qwen-crm",
            "prompt": f"{system_prompt}\n\nQuery: {query}",
            "stream": False,
            "format": "json"
        }
        
        # 3. Fetch & Display
        try:
            response = requests.post("http://localhost:11434/api/generate", json=payload)
            if response.status_code == 200:
                json_output = json.loads(response.json()["response"])
                st.success("Extraction Complete")
                st.json(json_output)
            else:
                st.error(f"Ollama API Error: {response.status_code}")
        except Exception as e:
            st.error(f"Connection failed: {e}")