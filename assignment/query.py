import os
from groq import Groq
from sentence_transformers import SentenceTransformer
from endee_client import search_vectors

_embedder=SentenceTransformer("all-MiniLM-L6-v2")
_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def build_prompt(question,context_chunks):
    context="\n\n---\n\n".join(context_chunks)
    return f"""Answer the following questions using ONLY the context provided below.

Context:{context}

Question:{question}

Answer:"""

def answer_question(question, top_k: int = 5):
    question_vector = _embedder.encode(question).tolist()
    results = search_vectors(question_vector, top_k=top_k)

    if not results:
        return {
            "question": question,
            "answer": "No relevant documents found. Please ingest documents first.",
            "sources": []
        }

    context_chunks = []
    sources = set()
    for result in results:
        meta = result.get("meta", {})
        chunk_text = meta.get("text", "")
        filename = meta.get("filename", "unknown")
        if chunk_text:
            context_chunks.append(chunk_text)
            sources.add(filename)

    if not context_chunks:
        return {
            "question": question,
            "answer": "Could not extract text from results.",
            "sources": []
        }

    prompt = build_prompt(question, context_chunks)
    completion = _groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "Answer using only the provided context. If the answer is not found, say 'I don't have enough information to answer that.'"},
        {"role": "user", "content": prompt}
    ]
)
    answer = completion.choices[0].message.content.strip()
    return {
        "question": question,
        "answer": answer,
        "sources": list(sources)
    }