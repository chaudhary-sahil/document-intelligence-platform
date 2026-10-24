"""Step 3: text -> vectors (384 numbers per text for all-MiniLM-L6-v2).

Two interchangeable backends run the SAME model:
  * sentence-transformers (PyTorch): what you use locally if it is installed.
  * ONNX (ships with ChromaDB, no PyTorch): small and light enough for free
    hosting tiers with ~1 GB of RAM.
"""
import importlib.util
from functools import lru_cache

from backend.config import EMBEDDING_BACKEND, EMBEDDING_MODEL


@lru_cache(maxsize=1)
def backend_name():
    if EMBEDDING_BACKEND in ("sentence-transformers", "onnx"):
        return EMBEDDING_BACKEND
    if importlib.util.find_spec("sentence_transformers") is not None:
        return "sentence-transformers"
    return "onnx"


@lru_cache(maxsize=1)
def get_model():
    if backend_name() == "sentence-transformers":
        # Imported here so that importing this file stays instant.
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(EMBEDDING_MODEL)

    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    return DefaultEmbeddingFunction()  # all-MiniLM-L6-v2 on onnxruntime


def warm_up():
    """Load the model now so the first real request isn't slow."""
    get_model()


def _as_list(vector):
    return vector.tolist() if hasattr(vector, "tolist") else [float(x) for x in vector]


def embed_texts(texts):
    model = get_model()
    if backend_name() == "sentence-transformers":
        # normalize_embeddings=True makes every vector length 1, so cosine
        # similarity is a plain dot product and scores are comparable.
        vectors = model.encode(
            texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()
    return [_as_list(v) for v in model(list(texts))]


def embed_query(query):
    return embed_texts([query])[0]
