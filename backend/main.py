"""FastAPI app. Run from the project root:  uvicorn backend.main:app --port 8000"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from backend.config import DEFAULT_TOP_K, MAX_UPLOAD_MB, UPLOAD_DIR
from backend.services import embedding_service, rag_service, vector_store
from backend.services.chunk_service import chunk_pages
from backend.services.llm_service import LLMError
from backend.services.pdf_service import PDFError, extract_pages


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the embedding model once at startup so the first upload isn't slow.
    embedding_service.get_model()
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

    Declared with plain `def` (not `async def`) on purpose: this work is slow
    and blocking, and FastAPI runs plain `def` endpoints in a thread pool so
    the server stays responsive for other requests meanwhile.
    """
    filename = Path(file.filename or "").name  # strips any ../ path tricks
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are accepted.")

    final_path = UPLOAD_DIR / filename
    temp_path = UPLOAD_DIR / f".{filename}.part"

    # 1. Save to a temporary file, enforcing the size limit while streaming.
    limit = MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    too_big = False
    with open(temp_path, "wb") as out:
        while True:
            block = file.file.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            if size > limit:
                too_big = True
                break
            out.write(block)
    if too_big:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")

    # 2. Check it really is a PDF (not just named .pdf).
    with open(temp_path, "rb") as check:
        if check.read(5) != b"%PDF-":
            temp_path.unlink(missing_ok=True)
            raise HTTPException(400, "This file is not a valid PDF.")

    # 3. Run the pipeline. Only replace the old file/index if everything succeeds.
    try:
        pages, total_pages = extract_pages(temp_path)
        chunks = chunk_pages(pages)
        embeddings = embedding_service.embed_texts([c["text"] for c in chunks])
        vector_store.add_document(filename, chunks, embeddings)
    except PDFError as exc:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(422, str(exc))
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    temp_path.replace(final_path)
    return {
        "filename": filename,
        "pages": total_pages,
        "pages_with_text": len(pages),
        "chunks": len(chunks),
        "status": "indexed",
    }


@app.get("/documents")
def documents():
    return vector_store.list_documents()


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    filename = Path(filename).name
    removed = vector_store.delete_document(filename)
    pdf_path = UPLOAD_DIR / filename
    file_existed = pdf_path.exists()
    pdf_path.unlink(missing_ok=True)
    if removed == 0 and not file_existed:
        raise HTTPException(404, f"No document named '{filename}'.")
    return {"filename": filename, "chunks_removed": removed, "file_deleted": file_existed}


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
    return vector_store.stats()
