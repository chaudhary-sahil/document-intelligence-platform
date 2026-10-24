"""Step 3: text -> vectors (lists of 384 numbers for all-MiniLM-L6-v2)."""
from functools import lru_cache

from backend.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_model():
    # Imported here so that importing this file is instant and tests that
    # don't need the model don't have to load PyTorch.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts):
    # normalize_embeddings=True makes every vector length 1, so cosine
    # similarity is a plain dot product and scores are comparable.
    vectors = get_model().encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_query(query):
    return embed_texts([query])[0]
