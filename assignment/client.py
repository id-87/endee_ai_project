from endee import Endee, Precision
import os
 
ENDEE_HOST = os.getenv("ENDEE_HOST", "http://localhost:8080/api/v1")
INDEX_NAME="assignment"
 
# Create an index
client.create_index(
    name="my_index",
    dimension=384,
    space_type="cosine",
    precision=Precision.INT8
)
 
# Get the index
index = client.get_index(name="my_index")
 
# Add vectors
index.upsert([
    {
        "id": "doc1",
        "vector": [...],
        "meta": {"title": "First Document"}
    }
])
 
# Query
results = index.query(vector=[...], top_k=5)

def get_client():
    client = Endee()
    client.set_base_url(ENDEE_HOST)
    return client


def get_index():
    client=get_client()
    try:
        index=client.get_index(INDEX_NAME)
        return index
    except Exception:
        client.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIMENSION,
            space_type="cosine",
            precision=Precision.INT8
        )
        return client.get_index(name=INDEX_NAME)
    
def upsert_vectors(vectors):
    index=get_index()
    index.upsert(vectors)

def search_vectors(query,top_k:int=5):
    index=get_index()
    index.query(vector=query,top_k=top_k,ef=128)



