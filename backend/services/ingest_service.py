"""Upload pipeline shared by the FastAPI endpoint and the Streamlit app:
save -> validate -> extract -> chunk -> embed -> store."""
from pathlib import Path

from backend.config import MAX_UPLOAD_MB, PROTECTED_DOCS, SAMPLE_PDF, UPLOAD_DIR
from backend.services import embedding_service, pdf_service, vector_store
from backend.services.chunk_service import chunk_pages


class IngestError(Exception):
    """A problem the user can understand; carries an HTTP-style status code."""

    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


def _index(path, filename):
    try:
        pages, total_pages = pdf_service.extract_pages(path)
    except pdf_service.PDFError as exc:
        raise IngestError(str(exc), 422) from exc
    chunks = chunk_pages(pages)
    embeddings = embedding_service.embed_texts([c["text"] for c in chunks])
    vector_store.add_document(filename, chunks, embeddings)
    return {
        "filename": filename,
        "pages": total_pages,
        "pages_with_text": len(pages),
        "chunks": len(chunks),
        "status": "indexed",
    }


def ingest_pdf(filename, fileobj, allow_protected=False):
    """Save an uploaded PDF and index it, replacing any earlier version of it."""
    name = Path(filename or "").name  # strips any ../ path tricks
    if not name.lower().endswith(".pdf"):
        raise IngestError("Only .pdf files are accepted.", 400)
    if name in PROTECTED_DOCS and not allow_protected:
        raise IngestError(
            f"'{name}' is a built-in demo document and can't be replaced. "
            "Please rename your file.",
            400,
        )

    final_path = UPLOAD_DIR / name
    temp_path = UPLOAD_DIR / f".{name}.part"

    # 1. Save to a temporary file, enforcing the size limit while streaming.
    limit = MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    too_big = False
    with open(temp_path, "wb") as out:
        while True:
            block = fileobj.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            if size > limit:
                too_big = True
                break
            out.write(block)
    if too_big:
        temp_path.unlink(missing_ok=True)
        raise IngestError(f"File is larger than {MAX_UPLOAD_MB} MB.", 413)

    # 2. Check it really is a PDF (not just named .pdf).
    with open(temp_path, "rb") as check:
        if check.read(5) != b"%PDF-":
            temp_path.unlink(missing_ok=True)
            raise IngestError("This file is not a valid PDF.", 400)

    # 3. Index it. Only replace the old file if everything succeeds.
    try:
        summary = _index(temp_path, name)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    temp_path.replace(final_path)
    return summary


def delete_document(filename):
    name = Path(filename or "").name
    if name in PROTECTED_DOCS:
        raise IngestError(f"'{name}' is a built-in demo document and can't be deleted.", 403)
    removed = vector_store.delete_document(name)
    pdf_path = UPLOAD_DIR / name
    file_existed = pdf_path.exists()
    pdf_path.unlink(missing_ok=True)
    if removed == 0 and not file_existed:
        raise IngestError(f"No document named '{name}'.", 404)
    return {"filename": name, "chunks_removed": removed, "file_deleted": file_existed}


def ensure_sample_indexed():
    """Demo mode: make sure the sample contract is searchable. Returns True if it indexed it."""
    if not SAMPLE_PDF.exists():
        return False
    if SAMPLE_PDF.name in {d["filename"] for d in vector_store.list_documents()}:
        return False
    with open(SAMPLE_PDF, "rb") as handle:
        ingest_pdf(SAMPLE_PDF.name, handle, allow_protected=True)
    return True
