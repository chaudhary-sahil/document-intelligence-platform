"""Central settings. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

# Absolute paths based on THIS file, so the app behaves the same no matter
# which folder you start it from. (Your old code used the relative path
# "vector_db", which silently creates a second database if you launch
# the server from a different directory.)
BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# --- Upload limits ---
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

# --- Chunking (measured in characters) ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# --- Embeddings / vector DB ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "documents_cosine"

# --- Retrieval ---
DEFAULT_TOP_K = 5
INSIGHT_TOP_K = 8
# Cosine similarity below this = "not relevant enough to answer from".
# 0.2 is a starting point; tune it after testing your own documents.
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.2"))

# --- LLM (Ollama runs models locally and exposes an HTTP API) ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "300"))
