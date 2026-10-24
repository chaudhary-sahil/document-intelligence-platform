"""Creates data/samples/sample_contract.pdf so you can test without hunting for a PDF.

Run from the project root:  python scripts/make_sample_pdf.py
"""
from pathlib import Path

import fitz  # PyMuPDF

PAGES = [
    (
        "SERVICE AGREEMENT",
        "This Service Agreement is made effective on 1 March 2026 between Northwind "
        "Analytics Ltd (the Provider) and Contoso Retail Inc (the Client).\n\n"
        "1. Scope of Services. The Provider shall deliver a monthly sales analytics "
        "dashboard and a quarterly strategy report to the Client throughout the term "
        "of this Agreement.\n\n"
        "2. Term. This Agreement runs for an initial period of 24 months from the "
        "effective date and renews automatically for successive 12 month periods unless "
        "either party gives notice of non-renewal at least 60 days before the end of "
        "the then-current term.",
    ),
    (
        "PAYMENT AND LIABILITY",
        "3. Fees and Payment. The Client shall pay the Provider a fixed fee of 8,000 USD "
        "per month. Invoices are issued on the first day of each month and must be paid "
        "within 45 days of the invoice date. Late payments incur interest of 1.5 percent "
        "per month.\n\n"
        "4. Limitation of Liability. The Provider's total liability under this Agreement "
        "shall not exceed the fees paid by the Client in the preceding 6 months. "
        "The Provider is not liable for indirect or consequential losses.\n\n"
        "5. Confidentiality. Each party must keep the other party's confidential "
        "information secret for 5 years after this Agreement ends.",
    ),
    (
        "TERMINATION",
        "6. Termination for Convenience. Either party may terminate this Agreement by "
        "giving 30 days written notice to the other party.\n\n"
        "7. Termination for Cause. Either party may terminate immediately by written "
        "notice if the other party materially breaches this Agreement and fails to "
        "remedy the breach within 15 days of being notified.\n\n"
        "8. Governing Law. This Agreement is governed by the laws of the State of "
        "New York. Disputes shall be resolved by binding arbitration in New York City.",
    ),
]

out = Path(__file__).resolve().parents[1] / "data" / "samples" / "sample_contract.pdf"
out.parent.mkdir(parents=True, exist_ok=True)

doc = fitz.open()
for title, body in PAGES:
    page = doc.new_page()  # A4
    page.insert_text((72, 80), title, fontsize=16)
    leftover = page.insert_textbox(fitz.Rect(72, 110, 523, 770), body, fontsize=11)
    if leftover < 0:
        raise RuntimeError("Text did not fit on the page; shorten the sample text.")
doc.save(out)
doc.close()
print(f"Created {out}")
