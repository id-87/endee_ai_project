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
    allow_origin=['*']
    allow_methods=['*']
    allow_headers=["*"]
)

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
def ask(request):
    if not request.question.strip():
        raise HTTPException(status_code=400,detail="Question can not be empty")
    
    response=answer_question(request.question,request.top_k)
    return response