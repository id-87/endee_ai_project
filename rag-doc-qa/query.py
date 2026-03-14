"""
query.py
--------
Handles the RAG query pipeline:
  1. Embed the user's question
  2. Search Endee for top-k similar chunks
  3. Build a prompt with retrieved context
  4. Call OpenAI to generate a grounded answer
"""

import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from endee_client import search_vectors

_embedder = SentenceTransformer("all-MiniLM-L6-v2")
_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly based on the provided context.
If the answer is not found in the context, say "I don't have enough information to answer that."
Do not make up information."""


def _build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return f"""Answer the following question using ONLY the context provided below.

Context:
{context}

Question: {question}

Answer:"""


def answer_question(question: str, top_k: int = 5) -> dict:
    """
    Full RAG pipeline: embed question → retrieve from Endee → generate answer.
    Returns a dict with question, answer, and source filenames.
    """
    # Step 1: Embed the question
    question_vector = _embedder.encode(question).tolist()

    # Step 2: Retrieve top-k chunks from Endee
    results = search_vectors(question_vector, top_k=top_k)

    if not results:
        return {
            "question": question,
            "answer": "No relevant documents found. Please ingest documents first.",
            "sources": []
        }

    # Step 3: Extract text chunks and source filenames from results
    context_chunks = []
    sources = set()
    for result in results:
        meta = result.get("meta", {})
        chunk_text = meta.get("text", "")
        filename = meta.get("filename", "unknown")
        if chunk_text:
            context_chunks.append(chunk_text)
            sources.add(filename)

    # Step 4: Generate answer with OpenAI
    prompt = _build_prompt(question, context_chunks)

    completion = _openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=512
    )

    answer = completion.choices[0].message.content.strip()

    return {
        "question": question,
        "answer": answer,
        "sources": list(sources)
    }
