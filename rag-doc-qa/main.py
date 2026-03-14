from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ingest import ingest_document
from query import answer_question

app = FastAPI(
    title="RAG Document Q&A API",
    description="Upload documents and ask questions powered by Endee Vector DB + OpenAI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


@app.get("/")
def root():
    return {"message": "RAG Doc Q&A API is running. Use /docs for Swagger UI."}


@app.post("/ingest", summary="Upload a PDF or text file for ingestion")
async def ingest(file: UploadFile = File(...)):
    """
    Upload a document (PDF or .txt). It will be chunked, embedded,
    and stored in Endee vector database.
    """
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")

    contents = await file.read()
    result = ingest_document(contents, file.filename)
    return result


@app.post("/ask", response_model=QuestionResponse, summary="Ask a question about ingested documents")
def ask(request: QuestionRequest):
    """
    Ask a natural language question. Endee retrieves the most relevant
    chunks, and OpenAI generates a grounded answer.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    response = answer_question(request.question, request.top_k)
    return response
