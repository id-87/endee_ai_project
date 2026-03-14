# 📄 RAG Document Q&A API

A production-style **Retrieval Augmented Generation (RAG)** system built with FastAPI, the **Endee Vector Database**, and OpenAI GPT-3.5. Upload any PDF or text document and ask natural language questions — answers are grounded entirely in your documents.

---

## 🧩 Problem Statement

Traditional keyword search fails to understand *meaning*. If a document says "the agreement was terminated" and you search for "contract ended", keyword search misses it. This project solves that by converting documents into semantic vector embeddings, storing them in **Endee**, and retrieving the most relevant content at query time — then using an LLM to synthesize a precise, grounded answer.

---

## 🏗️ System Design

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────────┐
│  User PDF   │────▶│  /ingest API │────▶│  Text Chunker       │
│  or .txt    │     │  (FastAPI)   │     │  (500 char chunks)  │
└─────────────┘     └──────────────┘     └────────┬────────────┘
                                                   │
                                         ┌─────────▼────────────┐
                                         │  sentence-transformers│
                                         │  all-MiniLM-L6-v2    │
                                         │  (384-dim embeddings) │
                                         └─────────┬────────────┘
                                                   │
                                         ┌─────────▼────────────┐
                                         │   Endee Vector DB    │
                                         │   (cosine + INT8)    │
                                         └──────────────────────┘

┌─────────────┐     ┌──────────────┐     ┌─────────────────────┐
│  Question   │────▶│  /ask API    │────▶│  Embed question     │
│  (user)     │     │  (FastAPI)   │     │  → search Endee     │
└─────────────┘     └──────────────┘     └────────┬────────────┘
                                                   │ top-k chunks
                                         ┌─────────▼────────────┐
                                         │  OpenAI GPT-3.5      │
                                         │  (context + prompt)  │
                                         └─────────┬────────────┘
                                                   │
                                         ┌─────────▼────────────┐
                                         │  Grounded Answer     │
                                         │  + Source filenames  │
                                         └──────────────────────┘
```

---

## 🔍 How Endee is Used

Endee is the **core retrieval layer** of this project:

| Step | Endee Operation |
|------|----------------|
| Setup | `client.create_index(name, dimension=384, space_type="cosine", precision=Precision.INT8)` |
| Ingestion | `index.upsert([{id, vector, meta, filter}])` — stores each chunk as a vector with text metadata |
| Retrieval | `index.query(vector=query_embedding, top_k=5, ef=128)` — returns most semantically similar chunks |

**Why Endee?**
- High recall cosine similarity search with INT8 quantization (memory efficient)
- Metadata stored alongside vectors (`meta.text`) — no secondary DB needed
- Runs locally via Docker on a single node — zero cloud cost for development
- Python SDK makes integration clean and simple

---

## 📁 Project Structure

```
rag-doc-qa/
├── main.py              # FastAPI app — /ingest and /ask endpoints
├── ingest.py            # Parse → chunk → embed → upsert to Endee
├── query.py             # Embed question → search Endee → call OpenAI
├── endee_client.py      # Endee SDK wrapper (index creation, upsert, search)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml   # Runs both Endee server and the FastAPI app
└── README.md
```

---

## ⚙️ Setup & Execution

### Prerequisites
- Docker + Docker Compose v2
- Python 3.11+
- Gemini API key (free at https://aistudio.google.com/app/apikey)

### Option 1: Docker Compose (Recommended)

```bash
# Clone this repo
git clone https://github.com/<your-username>/rag-doc-qa
cd rag-doc-qa

# Add your Gemini key
export GEMINI_API_KEY=your-gemini-key

# Start Endee + the FastAPI app
docker compose up --build
```

The API will be live at `http://localhost:8000`.
Endee dashboard will be at `http://localhost:8080`.

---

### Option 2: Run Locally (without Docker for the app)

**Step 1 — Start Endee via Docker:**
```bash
docker run -d -p 8080:8080 \
  -v endee-data:/data \
  endeeio/endee-server:latest
```

**Step 2 — Install Python dependencies:**
```bash
pip install -r requirements.txt
```

**Step 3 — Set environment variables:**
```bash
export GEMINI_API_KEY=your-gemini-key
export ENDEE_HOST=http://localhost:8080/api/v1
```

**Step 4 — Run the API:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🚀 Usage

### 1. Ingest a document
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@your_document.pdf"
```

**Response:**
```json
{
  "status": "success",
  "filename": "your_document.pdf",
  "chunks_ingested": 42
}
```

### 2. Ask a question
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?", "top_k": 5}'
```

**Response:**
```json
{
  "question": "What is the refund policy?",
  "answer": "According to the document, refunds are processed within 7 business days...",
  "sources": ["your_document.pdf"]
}
```

### 3. Swagger UI
Visit `http://localhost:8000/docs` to explore and test all endpoints interactively.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI |
| Vector Database | **Endee** (endee-io/endee) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`, 384-dim) |
| LLM | Google Gemini 1.5 Flash (free tier) |
| PDF Parsing | pdfplumber |
| Containerization | Docker + Docker Compose |

---

## 📌 Notes

- Endee runs with **no authentication** in dev mode. For production, set `NDD_AUTH_TOKEN` in `docker-compose.yml` and `ENDEE_AUTH_TOKEN` env var.
- The embedding model (`all-MiniLM-L6-v2`) is downloaded automatically on first run (~90MB).
- To ingest multiple documents, simply call `/ingest` multiple times — all vectors are stored in the same Endee index.