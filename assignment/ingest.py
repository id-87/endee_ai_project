import hashlib
import io
from sentence_transformers import SentenceTransformer
from endee_client import upsert_vectors

_embedder=SentenceTransformer("all-MiniLM-L6-v2")

CHUNK_SIZE=500
CHUNK_OVERLAP=100
def parse_text(contents,filename):
    if filename.endswith('.pdf'):
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(contents)) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as e:
            raise ValueError(f"failed to parse pdf:{e}")
    else:
        return contents.decode('utf-8',errors="ignore")


def _chunk_text(text):
    chunks=[]
    start=0
    while start<len(text):
        end=start+CHUNK_SIZE
        chunks.append(text[start:end].strip())
        start+=CHUNK_SIZE-CHUNK_OVERLAP
    return [c for c in chunks if len(c)>50]


def ingest_document(contents,filename):
    text=parse_text(contents,filename)
    if not text.strip():
        return {"status": "error", "detail": "No text could be extracted from the document.d"}
    chunks=_chunk_text(text)

    embeddings=_embedder.encode(chunks,show_progress_bar=False).tolist()

    vectors=[]

    for i,(chunk,embedding) in enumerate(zip(chunks,embeddings)):
        vectors.append({
            "id":_make_id(filename,i),
            "vector":embedding,
            "meta":{
                "filename":filename,
                "chunk_index":i,
                "text":chunk
            },
            "filter":{
                'filename':filename
            }
        })
    upsert_vectors(vectors)
    return {
        "status":"success",
        "filename":filename,
        "chunks_ingested":len(chunks)

    }
    

    