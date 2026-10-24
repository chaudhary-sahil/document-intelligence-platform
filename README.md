# Document Intelligence Platform

Upload PDFs (contracts, papers) → ask questions → get answers **with page citations**,
plus one-click Summary / Risks / Key dates / Obligations. Everything runs locally and free.

```
PDF → extract text per page → chunk (800 chars, 150 overlap) → embed (MiniLM, 384 numbers)
    → store in ChromaDB (with source + page)
Question → embed → top-K similar chunks → LLM (Ollama, Llama 3.2) answers ONLY from them → answer + citations
```

## 1. One-time setup (≈10 minutes, mostly downloads)

Needs Python 3.10+ (you used 3.12) and ~5 GB free disk.

```bash
cd document-intelligence-platform
python3 -m venv .venv                 # NOTE: .venv, not a venv inside a folder you share
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Install the local LLM runtime (Ollama), then download a model (~2 GB):

```bash
brew install ollama          # or download the app from https://ollama.com
ollama pull llama3.2         # the model the app uses by default
ollama list                  # confirm it appears
```

If you use the Ollama desktop app it is already serving. Otherwise run `ollama serve`
in its own terminal.

## 2. Run it (two terminals, both from the project root, venv activated)

Terminal 1 – backend:
```bash
uvicorn backend.main:app --port 8000
```
Wait for "Application startup complete" (the embedding model downloads on the first
run, ~90 MB, then loads). Swagger UI: http://127.0.0.1:8000/docs

Terminal 2 – frontend:
```bash
streamlit run frontend/app.py
```
Opens http://localhost:8501

## 3. Verify each stage (do these in order)

```bash
python scripts/make_sample_pdf.py          # creates data/samples/sample_contract.pdf
python -m unittest discover -s tests -t .  # all tests should pass (none skipped now)
```

| # | Action | Expected result |
|---|--------|-----------------|
| 1 | Open http://127.0.0.1:8000/db-info | `total_chunks: 0`, **`distance_metric: "cosine"`** |
| 2 | Streamlit sidebar → upload `data/samples/sample_contract.pdf` → **Process & index** | "Indexed sample_contract.pdf: 3 pages, ~5-8 chunks" |
| 3 | Upload the **same file again** | Chunk count in `/db-info` is unchanged (replace, not append) |
| 4 | **Raw search** tab: `how can the contract be terminated` | Top hit from page 3 with similarity ≈ 0.5+ |
| 5 | **Raw search**: `recipe for chocolate cake` | Low scores (≈ < 0.2) – irrelevant, as it should be |
| 6 | **Ask** tab: `How much notice is needed to terminate?` | "30 days [n]" with a source on page 3 |
| 7 | **Ask**: `What is the CEO's favourite colour?` | "I could not find this in the uploaded documents." |
| 8 | **Insights** → Key dates / Risks / Obligations / Summarize | Bullet lists with citations |
| 9 | Stop backend (Ctrl+C), start it again, open `/documents` | Your document is still listed (persistence) |
| 10 | Upload a second PDF, then ask with scope "All documents" | Sources from both files |
| 11 | 🗑 on a document | Disappears from list; `data/uploads/` file removed |

Stage 4 vs 5 is the quickest check that retrieval is healthy. If #1 does not say
`cosine`, delete the `vector_db/` folder and restart.

## 4. Project layout

```
backend/
  config.py                 all settings (paths, chunk size, model names, thresholds)
  main.py                   FastAPI endpoints
  services/
    pdf_service.py          PyMuPDF: PDF -> [{page, text}], rejects scans/corrupt files
    chunk_service.py        sentence-aware chunks, each tagged with its page
    embedding_service.py    sentence-transformers (lazy-loaded, normalized vectors)
    vector_store.py         ChromaDB: add/replace/delete/list/search (cosine)
    llm_service.py          Ollama HTTP client with friendly errors
    rag_service.py          prompts, relevance filter, Q&A and the 4 insights
frontend/app.py            Streamlit UI (Ask / Insights / Raw search)
scripts/make_sample_pdf.py  creates a test contract
tests/                      unit tests (chunking, PDF reading, RAG logic with fakes)
data/uploads/               your original PDFs      vector_db/   ChromaDB files
```

## 5. API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/upload` | Save + index a PDF (replaces an earlier version of the same filename) |
| GET | `/documents` | List indexed documents and chunk counts |
| DELETE | `/documents/{filename}` | Remove from index and disk |
| GET | `/search?q=…&top_k=5&source=file.pdf` | Retrieval only |
| POST | `/ask` | `{"question": "...", "top_k": 5, "source": null}` → answer + sources |
| POST | `/insights` | `{"kind": "risks", "source": "file.pdf"}` (summary, risks, key_dates, obligations) |
| GET | `/db-info` | Totals, distance metric, per-document counts |

## 6. Settings (environment variables)

`OLLAMA_MODEL` (default `llama3.2`; try `llama3.1:8b` for better answers if you have 16 GB RAM),
`MIN_RELEVANCE` (0.2), `CHUNK_SIZE` (800), `CHUNK_OVERLAP` (150), `MAX_UPLOAD_MB` (50),
`OLLAMA_URL`, `BACKEND_URL` (frontend → backend). Example:
`OLLAMA_MODEL=llama3.1:8b uvicorn backend.main:app --port 8000`

Changing chunk settings or the embedding model only affects documents indexed afterwards;
re-upload documents to re-index them.

## 7. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| Streamlit: "Cannot reach the backend" | Terminal 1 isn't running, or wrong port |
| Ask returns error "Cannot reach Ollama" | Open the Ollama app or run `ollama serve` |
| Error "Model 'llama3.2' is not installed" | `ollama pull llama3.2` |
| Upload says "No extractable text found" | Scanned PDF; needs OCR (see next steps) |
| First answer takes a minute | Ollama loading the model into memory; later ones are faster |
| Answers say "could not find" for things that are in the PDF | Check **Raw search** first. If the right chunk scores < 0.2, lower `MIN_RELEVANCE` or rephrase; if it isn't retrieved at all, adjust chunking |
| `ModuleNotFoundError: backend` | Run commands from the project root, not from inside `backend/` |
| Big PDF (hundreds of pages) takes minutes to index | Normal on CPU; the embedding step dominates |

## 8. Honest limits / next steps

* Summaries/Risks use the top 8 relevant chunks, not the whole document — good for
  contracts and short papers, an approximation for long books.
* Small local models can miss nuance; legal output is a reading aid, not legal advice.
* Scanned PDFs need OCR (Tesseract or `ocrmypdf`) before indexing.
* Later: re-ranking, hybrid keyword + vector search, streaming answers, auth, Docker,
  swapping Ollama for a hosted LLM by rewriting only `llm_service.generate`.
