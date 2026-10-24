import tempfile
import unittest
from pathlib import Path

try:
    import fitz
except ImportError:  # PyMuPDF not installed
    fitz = None


@unittest.skipIf(fitz is None, "PyMuPDF not installed")
class PDFServiceTests(unittest.TestCase):
    def _make_pdf(self, texts):
        doc = fitz.open()
        for text in texts:
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text)
        path = Path(tempfile.mkdtemp()) / "t.pdf"
        doc.save(path)
        doc.close()
        return path

    def test_pages_are_extracted_with_numbers(self):
        from backend.services.pdf_service import extract_pages

        pages, total = extract_pages(self._make_pdf(["Alpha text", "", "Gamma text"]))
        self.assertEqual(total, 3)
        self.assertEqual([p["page"] for p in pages], [1, 3])  # blank page skipped
        self.assertIn("Gamma", pages[1]["text"])

    def test_pdf_without_text_is_rejected(self):
        from backend.services.pdf_service import PDFError, extract_pages

        with self.assertRaises(PDFError):
            extract_pages(self._make_pdf([""]))

    def test_garbage_file_is_rejected(self):
        from backend.services.pdf_service import PDFError, extract_pages

        bad = Path(tempfile.mkdtemp()) / "bad.pdf"
        bad.write_bytes(b"not a pdf at all")
        with self.assertRaises(PDFError):
            extract_pages(bad)


if __name__ == "__main__":
    unittest.main()
