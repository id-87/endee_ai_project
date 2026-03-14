"""
endee_client.py
---------------
Thin wrapper around the Endee Python SDK.
Handles index creation and exposes upsert / query helpers.
"""

from endee import Endee, Precision
import os

ENDEE_HOST = os.getenv("ENDEE_HOST", "http://localhost:8080/api/v1")
ENDEE_AUTH_TOKEN = os.getenv("ENDEE_AUTH_TOKEN", "")  # Leave empty if auth disabled
INDEX_NAME = "rag_documents"
VECTOR_DIMENSION = 384  # Matches all-MiniLM-L6-v2


def get_client() -> Endee:
    client = Endee(ENDEE_AUTH_TOKEN) if ENDEE_AUTH_TOKEN else Endee()
    client.set_base_url(ENDEE_HOST)
    return client


def get_or_create_index():
    """Return the Endee index, creating it if it doesn't exist."""
    client = get_client()
    try:
        index = client.get_index(name=INDEX_NAME)
        return index
    except Exception:
        client.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIMENSION,
            space_type="cosine",
            precision=Precision.INT8
        )
        return client.get_index(name=INDEX_NAME)


def upsert_vectors(vectors: list[dict]):
    """
    Upsert a list of vector objects into Endee.
    Each dict must contain: id, vector, meta, filter.
    """
    index = get_or_create_index()
    index.upsert(vectors)


def search_vectors(query_vector: list[float], top_k: int = 5) -> list[dict]:
    """
    Query Endee for the top_k most similar vectors to query_vector.
    Returns a list of result dicts with id, similarity, and meta.
    """
    index = get_or_create_index()
    results = index.query(vector=query_vector, top_k=top_k, ef=128)
    return results
