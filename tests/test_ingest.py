import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.services import ingest_service
from backend.services.ingest_service import IngestError
from backend.services.pdf_service import PDFError

PDF_BYTES = b"%PDF-1.4 fake content"
PAGES = [{"page": 1, "text": "Hello world. " * 5}]


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        patches = [
            patch.object(ingest_service, "UPLOAD_DIR", self.dir),
            patch.object(ingest_service.pdf_service, "extract_pages", return_value=(PAGES, 1)),
            patch.object(ingest_service.embedding_service, "embed_texts",
                         side_effect=lambda texts: [[0.1] * 3 for _ in texts]),
            patch.object(ingest_service.vector_store, "add_document"),
        ]
        self.mocks = [p.start() for p in patches]
        for p in patches:
            self.addCleanup(p.stop)
        self.add_document = self.mocks[3]

    def test_successful_upload_saves_file_and_indexes(self):
        result = ingest_service.ingest_pdf("doc.pdf", io.BytesIO(PDF_BYTES))
        self.assertEqual(result["status"], "indexed")
        self.assertTrue((self.dir / "doc.pdf").exists())
        self.assertFalse((self.dir / ".doc.pdf.part").exists())
        self.add_document.assert_called_once()

    def test_path_tricks_are_stripped(self):
        result = ingest_service.ingest_pdf("../../evil.pdf", io.BytesIO(PDF_BYTES))
        self.assertEqual(result["filename"], "evil.pdf")
        self.assertTrue((self.dir / "evil.pdf").exists())

    def test_non_pdf_extension_rejected(self):
        with self.assertRaises(IngestError) as ctx:
            ingest_service.ingest_pdf("notes.txt", io.BytesIO(b"hi"))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_fake_pdf_rejected_and_cleaned_up(self):
        with self.assertRaises(IngestError):
            ingest_service.ingest_pdf("x.pdf", io.BytesIO(b"not a pdf"))
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_too_large_rejected(self):
        with patch.object(ingest_service, "MAX_UPLOAD_MB", 0):
            with self.assertRaises(IngestError) as ctx:
                ingest_service.ingest_pdf("x.pdf", io.BytesIO(PDF_BYTES))
        self.assertEqual(ctx.exception.status_code, 413)
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_scanned_pdf_gives_422_and_keeps_old_file(self):
        (self.dir / "x.pdf").write_bytes(b"OLD")
        with patch.object(ingest_service.pdf_service, "extract_pages", side_effect=PDFError("scan")):
            with self.assertRaises(IngestError) as ctx:
                ingest_service.ingest_pdf("x.pdf", io.BytesIO(PDF_BYTES))
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertEqual((self.dir / "x.pdf").read_bytes(), b"OLD")

    def test_failure_during_indexing_cleans_temp_file(self):
        self.add_document.side_effect = RuntimeError("db down")
        with self.assertRaises(RuntimeError):
            ingest_service.ingest_pdf("x.pdf", io.BytesIO(PDF_BYTES))
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_protected_documents_cannot_be_replaced_or_deleted(self):
        with patch.object(ingest_service, "PROTECTED_DOCS", {"sample_contract.pdf"}):
            with self.assertRaises(IngestError):
                ingest_service.ingest_pdf("sample_contract.pdf", io.BytesIO(PDF_BYTES))
            with self.assertRaises(IngestError) as ctx:
                ingest_service.delete_document("sample_contract.pdf")
            self.assertEqual(ctx.exception.status_code, 403)
            ingest_service.ingest_pdf("sample_contract.pdf", io.BytesIO(PDF_BYTES), allow_protected=True)

    def test_delete_removes_index_and_file(self):
        (self.dir / "x.pdf").write_bytes(b"data")
        with patch.object(ingest_service.vector_store, "delete_document", return_value=3):
            result = ingest_service.delete_document("x.pdf")
        self.assertEqual(result["chunks_removed"], 3)
        self.assertFalse((self.dir / "x.pdf").exists())

    def test_delete_unknown_is_404(self):
        with patch.object(ingest_service.vector_store, "delete_document", return_value=0):
            with self.assertRaises(IngestError) as ctx:
                ingest_service.delete_document("ghost.pdf")
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
