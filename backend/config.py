"""Central settings. Every value can be overridden with an environment variable
(or, on Streamlit Community Cloud, with a secret of the same name)."""
import os
import tempfile
from pathlib import Path


def _flag(name, default="false"):
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _writable_dir(path):
    """Create `path` and make sure we can write to it; otherwise fall back to a temp dir.

    Some hosts mount the repository read-only. Falling back keeps the app
    running (data is then lost on restart, which is fine for a demo).
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.touch()
        probe.unlink()
        return path
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "document-intelligence-platform" / path.name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


# Absolute paths based on THIS file, so the app behaves the same no matter
# which folder you start it from.
BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = _writable_dir(BASE_DIR / "data" / "uploads")
VECTOR_DB_DIR = _writable_dir(BASE_DIR / "vector_db")
SAMPLE_PDF = BASE_DIR / "data" / "samples" / "sample_contract.pdf"

# --- Demo mode (public, shared deployment) ---
# Pre-loads the sample contract, protects it from deletion, lowers the upload
# limit and caps how many questions one visitor can ask.
DEMO_MODE = _flag("DEMO_MODE")
DEMO_MAX_QUESTIONS = int(os.getenv("DEMO_MAX_QUESTIONS", "30"))
PROTECTED_DOCS = {SAMPLE_PDF.name} if DEMO_MODE else set()

# --- Upload limits ---
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "5" if DEMO_MODE else "50"))

# --- Chunking (measured in characters) ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# --- Embeddings / vector DB ---
# "auto": use sentence-transformers (PyTorch) if it is installed, otherwise the
# lightweight ONNX build of the same all-MiniLM-L6-v2 model that ships with
# ChromaDB. Force one with "sentence-transformers" or "onnx".
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "auto").strip().lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "documents_cosine"

# --- Retrieval ---
DEFAULT_TOP_K = 5
INSIGHT_TOP_K = 8
# Cosine similarity below this = "not relevant enough to answer from".
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.2"))

# --- LLM ---
# "ollama": a model running on your own machine (default, free, private).
# "openai": any OpenAI-compatible API. Defaults point at Groq's free tier.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "300"))

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY", "")
# Reasoning models (e.g. gpt-oss) spend part of this budget "thinking" before they answer,
# so it must be generous or answers get cut off mid-sentence.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "3000"))
