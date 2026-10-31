"""
rag_pipeline.py
---------------
Retrieval-Augmented Generation (RAG) pipeline for the Local AI CRM system.

Public API
----------
  initialize_db()              -> Ingest dummy company policies into ChromaDB.
  get_policy_context(query)    -> Retrieve the most relevant policy for a query.

The ChromaDB collection "crm_policies" is persisted to the project's
vector_db/ directory so it survives process restarts.
"""

import os
import sys
from pathlib import Path

# Force UTF-8 output on Windows to avoid cp1252 codec errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR     = Path(__file__).resolve().parent   # …/Local_AI_CRM/src/
ROOT_DIR    = SRC_DIR.parent                    # …/Local_AI_CRM/
VECTOR_DB_DIR = ROOT_DIR / "vector_db"

COLLECTION_NAME  = "crm_policies"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# ---------------------------------------------------------------------------
# Dummy policy data
# ---------------------------------------------------------------------------
POLICIES: list[dict] = [
    {
        "id":       "policy_shipping",
        "category": "shipping",
        "policy": (
            "If a shipment is delayed by more than 3 days, offer a full refund of "
            "shipping costs and a 10% discount on the next order."
        ),
    },
    {
        "id":       "policy_refund",
        "category": "refund",
        "policy": (
            "Refunds must be requested within 30 days of purchase. Items must be in "
            "original packaging. Issue store credit for unboxed items."
        ),
    },
    {
        "id":       "policy_technical",
        "category": "technical",
        "policy": (
            "For hardware failure within 1 year, issue an immediate RMA for replacement. "
            "Software glitches should be escalated to Level 2 support."
        ),
    },
    {
        "id":       "policy_account",
        "category": "account",
        "policy": (
            "To reset a password or change billing details, the user must verify their "
            "identity via 2FA email code."
        ),
    },
]

# ---------------------------------------------------------------------------
# Module-level singletons (lazily initialised)
# ---------------------------------------------------------------------------
_embed_model: SentenceTransformer | None = None
_chroma_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def _get_embed_model() -> SentenceTransformer:
    """Return (and cache) the BGE embedding model."""
    global _embed_model
    if _embed_model is None:
        print(f"[RAG] Loading embedding model '{EMBED_MODEL_NAME}'...")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        print("[RAG] Embedding model loaded.")
    return _embed_model


def _get_chroma_client() -> chromadb.PersistentClient:
    """Return (and cache) a persistent ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[RAG] Connecting to ChromaDB at '{VECTOR_DB_DIR}'...")
        _chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        print("[RAG] ChromaDB client ready.")
    return _chroma_client


def _get_collection() -> chromadb.Collection:
    """Return (and cache) the ChromaDB collection, creating it if absent."""
    global _collection
    if _collection is None:
        client = _get_chroma_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine distance suits BGE
        )
    return _collection


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def initialize_db() -> None:
    """
    Ingest POLICIES into ChromaDB.

    Safe to call multiple times — existing entries matched by their stable
    string IDs are skipped, so no duplicates are created.
    """
    collection = _get_collection()
    model      = _get_embed_model()

    # Discover which policy IDs are already stored
    all_ids  = [p["id"] for p in POLICIES]
    existing = collection.get(ids=all_ids)
    existing_ids: set[str] = set(existing["ids"]) if existing["ids"] else set()

    new_policies = [p for p in POLICIES if p["id"] not in existing_ids]

    if not new_policies:
        print("[RAG] All policies already present in ChromaDB -- skipping ingestion.")
        return

    # Build composite documents: "Category: <cat>. Policy: <text>"
    documents: list[str] = [
        f"Category: {p['category']}. Policy: {p['policy']}"
        for p in new_policies
    ]

    print(f"[RAG] Generating embeddings for {len(new_policies)} new polic(ies)...")
    embeddings: list[list[float]] = model.encode(
        documents,
        normalize_embeddings=True,  # BGE recommends L2-normalisation for cosine sim
        show_progress_bar=False,
    ).tolist()

    collection.add(
        ids        = [p["id"]       for p in new_policies],
        documents  = documents,
        embeddings = embeddings,
        metadatas  = [
            {"category": p["category"], "policy": p["policy"]}
            for p in new_policies
        ],
    )
    print(f"[RAG] Ingested {len(new_policies)} polic(ies) into '{COLLECTION_NAME}'.")


def get_policy_context(query: str) -> str:
    """
    Retrieve the most relevant policy text for *query*.

    Parameters
    ----------
    query : str
        Free-text complaint or category term (e.g. "shipping delay").

    Returns
    -------
    str
        The raw policy string from the best-matching document, or a
        fallback message if the collection is empty.
    """
    collection = _get_collection()
    model      = _get_embed_model()

    query_embedding: list[float] = model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        include=["metadatas", "documents", "distances"],
    )

    if not results or not results.get("metadatas") or not results["metadatas"][0]:
        return "No matching policy found."

    top_metadata = results["metadatas"][0][0]
    top_distance = results["distances"][0][0]
    policy_text: str = top_metadata.get("policy", "No policy text available.")
    category:    str = top_metadata.get("category", "unknown")

    print(
        f"[RAG] Query: '{query}' -> matched category: '{category}' "
        f"(distance: {top_distance:.4f})"
    )
    return policy_text


# ---------------------------------------------------------------------------
# Testing block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    SEP = "=" * 60
    print(SEP)
    print("  CRM RAG Pipeline -- Initialisation & Smoke Test")
    print(SEP)

    # Step 1: Ingest all policies into ChromaDB
    initialize_db()

    # Step 2: Test retrieval across all four categories
    print()
    test_queries = [
        "shipping delay",
        "refund request",
        "technical support hardware failure",
        "account password reset",
    ]

    for q in test_queries:
        result = get_policy_context(q)
        print(f"  Query  : {q}")
        print(f"  Policy : {result}")
        print()

    # Verify vector_db directory was created
    db_files = list(VECTOR_DB_DIR.rglob("*"))
    print(f"[RAG] vector_db/ contains {len(db_files)} file(s).")
    for f in db_files:
        print(f"  {f.relative_to(ROOT_DIR)}")

    print()
    print(SEP)
    print("  Smoke test complete.")
    print(SEP)
