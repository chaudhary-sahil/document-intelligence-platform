"""Optional FastAPI app (the Streamlit app doesn't need it).
Run from the project root:  uvicorn backend.main:app --port 8000"""
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from backend.config import DEFAULT_TOP_K
from backend.services import embedding_service, ingest_service, rag_service, vector_store
from backend.services.ingest_service import IngestError
from backend.services.llm_service import LLMError
from backend.services.system_info import system_info


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedding_service.warm_up()  # so the first upload isn't slow
    yield


app = FastAPI(title="Document Intelligence Platform", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20)
    source: Optional[str] = None  # None = search all documents


class InsightRequest(BaseModel):
    kind: Literal["summary", "risks", "key_dates", "obligations"]
    source: str


@app.get("/")
def health_check():
    return {"status": "running"}


@app.post("/upload")
def upload_and_index(file: UploadFile = File(...)):
    """Save a PDF, then extract -> chunk -> embed -> store, in one request.

    Plain `def` (not `async def`) on purpose: the work is slow and blocking, and
    FastAPI runs plain `def` endpoints in a thread pool so the server stays responsive.
    """
    try:
        return ingest_service.ingest_pdf(file.filename, file.file)
    except IngestError as exc:
        raise HTTPException(exc.status_code, str(exc))


@app.get("/documents")
def documents():
    return vector_store.list_documents()


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    try:
        return ingest_service.delete_document(filename)
    except IngestError as exc:
        raise HTTPException(exc.status_code, str(exc))


@app.get("/search")
def semantic_search(
    q: str = Query(min_length=1),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=20),
    source: Optional[str] = None,
):
    """Raw retrieval (no LLM). Useful for judging retrieval quality."""
    return {"query": q, "results": rag_service.retrieve(q, top_k, source)}


@app.post("/ask")
def ask(request: AskRequest):
    try:
        return rag_service.answer_question(request.question, request.top_k, request.source)
    except LLMError as exc:
        raise HTTPException(503, str(exc))


@app.post("/insights")
def insights(request: InsightRequest):
    try:
        return rag_service.generate_insight(request.kind, request.source)
    except LLMError as exc:
        raise HTTPException(503, str(exc))


@app.get("/db-info")
def db_info():
    return system_info()
