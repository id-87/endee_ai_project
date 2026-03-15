from fastapi import FastAPI,File,UploadFile,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ingest import ingest_document
from query import answer_question

app=FastAPI(
    title="RAGF QnA document",
    description="Upload documents and asked questions",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=["*"]
)
class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5
@app.get("/")
def root():
    return {'message':"RAG project running"}

@app.post('/ingest')
async def ingest(file:UploadFile=File(...)):
    if not file.filename.endswith((".pdf",".txt")):
        raise HTTPException(status_code=400,detail="Only pdf and txt files are supported")
    
    contents=await file.read()
    result=ingest_document(contents,file.filename)
    return result


@app.post('/ask')
def ask(request:QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400,detail="Question can not be empty")
    
    response=answer_question(request.question,request.top_k)
    return response


@app.get("/debug")
def debug():
    from endee_client import _index
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    vector = embedder.encode("refund policy").tolist()
    results = _index.query(vector=vector, top_k=5, ef=128)
    return {"count": len(results), "results": str(results)}