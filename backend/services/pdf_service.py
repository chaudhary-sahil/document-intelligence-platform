"""Step 1 of the pipeline: PDF -> text, kept page by page."""


class PDFError(ValueError):
    """Raised when a PDF can't be read or contains no extractable text."""


def extract_pages(pdf_path):
    """Return (pages, total_pages).

    pages is a list like [{"page": 1, "text": "..."}, ...] containing only
    pages that have text. Keeping the page number separate (instead of pasting
    "--- PAGE 3 ---" into the text) is what makes page-level citations possible.
    """
    import fitz  # PyMuPDF, imported here so this module can be imported without it

    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        raise PDFError(f"Could not open PDF: {exc}") from exc

    pages = []
    try:
        total_pages = len(document)
        for index in range(total_pages):
            text = document[index].get_text().strip()
            if text:
                pages.append({"page": index + 1, "text": text})
    finally:
        document.close()

    if not pages:
        raise PDFError(
            "No extractable text found. This PDF is probably a scan (images "
            "only). It needs OCR before it can be indexed."
        )
    return pages, total_pages
