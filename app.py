import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import requests
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

app = FastAPI(title="Local AI CRM API (OpenAI Compatible)")

print(f"\n⚡ CURRENTLY EXECUTING FILE: {os.path.abspath(__file__)} ⚡\n")
print("Loading BGE-small and ChromaDB...")

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = Chroma(persist_directory="./crm_chroma_db", embedding_function=embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    model: str = "qwen-crm"
    
    class Config:
        extra = "ignore" 

@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    # 1. Isolate the user's latest query from the OpenAI message array
    user_query = ""
    for msg in reversed(request.messages):
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break
            
    if not user_query:
        raise HTTPException(status_code=400, detail="No user message found")

    # 2. Retrieve context via BGE-small
    retrieved_docs = retriever.invoke(user_query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # 3. Build the strict JSON extraction prompt
    system_prompt = f"""You are a CRM data extraction assistant. Answer the user's query based on the provided context.
Return the extracted information strictly in JSON format.

Context:
{context}"""
    
    payload = {
        "model": "qwen-crm",
        "prompt": f"{system_prompt}\n\nQuery: {user_query}",
        "stream": False,
        "format": "json"
    }
    
    # 4. Generate with local Ollama
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Ollama generation failed")
            
        raw_json_output = response.json()["response"]
        
        # 5. Return exact OpenAI-formatted response object with token usage
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": raw_json_output
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))