"""
document_loader.py
------------------
Parses, chunks, and embeds raw CRM documents into the persistent ChromaDB 
vector database (./crm_chroma_db) using LangChain and BAAI/bge-small-en-v1.5.

Matches the Chroma schema and embedding model expected by rag_pipeline.py.
"""

import os
import sys
from pathlib import Path
from typing import List

# Force UTF-8 encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ---------------------------------------------------------------------------
# Configuration & Default Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DOCS_DIR = BASE_DIR / "data" / "crm_documents"
DEFAULT_DB_DIR = BASE_DIR / "crm_chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_raw_documents(docs_dir: Path) -> List[Document]:
    """
    Recursively scans the directory and loads text/markdown/json files 
    into standard LangChain Document objects.
    """
    documents: List[Document] = []
    supported_extensions = {".md", ".txt", ".json", ".log"}

    if not docs_dir.exists():
        print(f"[Loader] Warning: Directory '{docs_dir}' does not exist.")
        return documents

    for file_path in docs_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
                if content:
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": str(file_path.relative_to(BASE_DIR)),
                                "filename": file_path.name,
                                "extension": file_path.suffix.lower(),
                            },
                        )
                    )
            except Exception as e:
                print(f"[Loader] Error reading '{file_path.name}': {e}")

    print(f"[Loader] Loaded {len(documents)} raw document(s) from '{docs_dir}'.")
    return documents


def chunk_documents(
    documents: List[Document], 
    chunk_size: int = CHUNK_SIZE, 
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Document]:
    """
    Splits loaded documents into semantically coherent chunks using 
    RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    
    # Assign unique deterministic IDs to each chunk based on source & sequence
    for idx, chunk in enumerate(chunks):
        source = chunk.metadata.get("filename", "doc")
        chunk.metadata["chunk_id"] = f"{source}_chunk_{idx}"

    print(f"[Splitter] Generated {len(chunks)} text chunk(s) (size={chunk_size}, overlap={chunk_overlap}).")
    return chunks


def ingest_to_chroma(
    chunks: List[Document], 
    persist_dir: Path = DEFAULT_DB_DIR,
    model_name: str = EMBEDDING_MODEL_NAME
) -> Chroma:
    """
    Embeds text chunks with BGE-small and persists them into ChromaDB.
    Ensures idempotency by checking existing document IDs.
    """
    persist_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Embeddings] Initializing HuggingFace embedding model: '{model_name}'...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # CPU inference for embedding is fast and preserves GPU VRAM for Ollama
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"[Chroma] Connecting to persistent vector database at '{persist_dir}'...")
    vector_store = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embedding_model,
    )

    if not chunks:
        print("[Chroma] No new chunks to ingest.")
        return vector_store

    chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]

    # Query existing IDs to avoid duplicate ingestion
    existing_data = vector_store.get(ids=chunk_ids)
    existing_ids = set(existing_data.get("ids", [])) if existing_data else set()

    new_chunks = [c for c in chunks if c.metadata["chunk_id"] not in existing_ids]
    new_ids = [c.metadata["chunk_id"] for c in new_chunks]

    if new_chunks:
        print(f"[Chroma] Ingesting {len(new_chunks)} new chunk(s) into ChromaDB...")
        vector_store.add_documents(documents=new_chunks, ids=new_ids)
        print(f"[Chroma] Successfully saved to '{persist_dir}'.")
    else:
        print("[Chroma] All chunks already exist in vector store. Skipping re-ingestion.")

    return vector_store


def main():
    print("=" * 65)
    print("  Local AI CRM -- Document Ingestion & Vector DB Builder")
    print("=" * 65)

    docs_dir = DEFAULT_DOCS_DIR
    persist_dir = DEFAULT_DB_DIR

    # Step 1: Load documents
    raw_docs = load_raw_documents(docs_dir)
    if not raw_docs:
        print(f"[Error] No supported documents found in '{docs_dir}'.")
        print("Please place .md, .txt, or .json files in that directory.")
        return

    # Step 2: Split into chunks
    chunks = chunk_documents(raw_docs)

    # Step 3: Embed & Ingest into Chroma
    vector_store = ingest_to_chroma(chunks, persist_dir=persist_dir)

    # Step 4: Verification test search
    print("\n--- Running Verification Test Query ---")
    test_query = "What is the status of the client's latest software deployment?"
    results = vector_store.similarity_search(test_query, k=2)

    print(f"Test Query: '{test_query}'")
    print(f"Retrieved Chunks: {len(results)}\n")
    for i, doc in enumerate(results, start=1):
        print(f"[{i}] Source: {doc.metadata.get('source')}")
        print(f"Content Preview: {doc.page_content[:200]}...\n")

    print("=" * 65)
    print("Ingestion complete! You can now run 'python rag_pipeline.py'.")
    print("=" * 65)


if __name__ == "__main__":
    main()

