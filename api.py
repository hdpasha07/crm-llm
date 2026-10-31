from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

app = FastAPI(title="Local AI CRM API")

# Initialize the RAG components once at startup
print("Loading BGE-small and ChromaDB...")
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = Chroma(persist_directory="./crm_chroma_db", embedding_function=embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

class QueryRequest(BaseModel):
    query: str

@app.post("/extract")
def extract_crm_data(request: QueryRequest):
    # 1. Retrieve Context using BGE-small
    retrieved_docs = retriever.invoke(request.query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # 2. Build the strict prompt
    system_prompt = f"""You are a CRM data extraction assistant. Answer the user's query based on the provided context.
Return the extracted information strictly in JSON format.

Context:
{context}"""
    
    payload = {
        "model": "qwen-crm",
        "prompt": f"{system_prompt}\n\nQuery: {request.query}",
        "stream": False,
        "format": "json"
    }
    
    # 3. Query local Ollama Qwen model
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        if response.status_code == 200:
            return json.loads(response.json()["response"])
        else:
            raise HTTPException(status_code=500, detail="Ollama generation failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))