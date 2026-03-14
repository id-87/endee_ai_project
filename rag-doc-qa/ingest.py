"""
ingest.py
---------
Handles document ingestion:
  1. Parse PDF or TXT content
  2. Split into overlapping text chunks
  3. Embed each chunk using sentence-transformers
  4. Upsert vectors into Endee
"""

import hashlib
import io
from sentence_transformers import SentenceTransformer
from endee_client import upsert_vectors

# Load embedding model once at startup
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 100    # overlap between consecutive chunks


def _parse_text(contents: bytes, filename: str) -> str:
    """Extract raw text from PDF or TXT file bytes."""
    if filename.endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(contents)) as pdf:
                return "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {e}")
    else:
        return contents.decode("utf-8", errors="ignore")


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 50]  # drop tiny tail chunks


def _make_id(filename: str, chunk_index: int) -> str:
    """Generate a stable unique ID for each chunk."""
    raw = f"{filename}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def ingest_document(contents: bytes, filename: str) -> dict:
    """
    Full ingestion pipeline for a single document.
    Returns a summary dict of what was stored.
    """
    # Step 1: Parse
    text = _parse_text(contents, filename)
    if not text.strip():
        return {"status": "error", "detail": "No text could be extracted from the document."}

    # Step 2: Chunk
    chunks = _chunk_text(text)

    # Step 3: Embed all chunks in one batch (faster than one by one)
    embeddings = _embedder.encode(chunks, show_progress_bar=False).tolist()

    # Step 4: Build vector objects for Endee
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": _make_id(filename, i),
            "vector": embedding,
            "meta": {
                "filename": filename,
                "chunk_index": i,
                "text": chunk
            },
            "filter": {
                "filename": filename
            }
        })

    # Step 5: Upsert into Endee
    upsert_vectors(vectors)

    return {
        "status": "success",
        "filename": filename,
        "chunks_ingested": len(chunks)
    }
