from endee import Endee, Precision
import os

ENDEE_HOST = os.getenv("ENDEE_HOST", "http://localhost:8080/api/v1")
INDEX_NAME = "assignment"
VECTOR_DIMENSION = 384

# Create client and index ONCE at startup
_client = Endee()
_client.set_base_url(ENDEE_HOST)

try:
    _index = _client.get_index(INDEX_NAME)
except Exception:
    _client.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIMENSION,
        space_type="cosine",
        precision=Precision.INT8
    )
    _index = _client.get_index(name=INDEX_NAME)


def upsert_vectors(vectors):
    _index.upsert(vectors)


def search_vectors(query, top_k: int = 5):
    return _index.query(vector=query, top_k=top_k, ef=128)