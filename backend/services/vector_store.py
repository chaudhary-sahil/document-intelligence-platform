"""Step 4: store and search vectors in ChromaDB (persisted on disk)."""
from collections import Counter
from functools import lru_cache

from backend.config import COLLECTION_NAME, VECTOR_DB_DIR

ADD_BATCH_SIZE = 500  # Chroma limits how many records one add() call may hold


@lru_cache(maxsize=1)
def get_collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    # "cosine" distance: distance = 1 - cosine_similarity, so we can report
    # an easy-to-read similarity score (1.0 = identical meaning).
    # Chroma's default is L2, which is why your old distances looked like 1.14-1.60.
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def delete_document(filename):
    """Remove every chunk of one document. Returns how many were removed."""
    collection = get_collection()
    existing = collection.get(where={"source": filename}, include=[])
    ids = existing["ids"]
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def add_document(filename, chunks, embeddings):
    """Index a document, REPLACING any earlier version of the same filename.

    Other documents are never touched, so many PDFs can live in one database.
    """
    collection = get_collection()
    delete_document(filename)

    for i in range(0, len(chunks), ADD_BATCH_SIZE):
        batch = chunks[i : i + ADD_BATCH_SIZE]
        collection.add(
            ids=[f"{filename}::chunk_{c['chunk_index']}" for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=embeddings[i : i + ADD_BATCH_SIZE],
            metadatas=[
                {
                    "source": filename,
                    "page": c["page"],
                    "chunk_index": c["chunk_index"],
                }
                for c in batch
            ],
        )


def list_documents():
    result = get_collection().get(include=["metadatas"])
    counts = Counter(meta["source"] for meta in result["metadatas"])
    return [{"filename": name, "chunks": n} for name, n in sorted(counts.items())]


def search(query_embedding, top_k=5, source=None):
    """Return the top_k most similar chunks (optionally from one document)."""
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, total),
        "include": ["documents", "metadatas", "distances"],
    }
    if source:
        kwargs["where"] = {"source": source}

    result = collection.query(**kwargs)

    hits = []
    for text, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append(
            {
                "text": text,
                "source": meta["source"],
                "page": meta["page"],
                "chunk_index": meta["chunk_index"],
                "score": round(1 - distance, 4),
            }
        )
    return hits


def stats():
    collection = get_collection()
    metric = (collection.metadata or {}).get("hnsw:space", "l2 (default)")
    return {
        "total_chunks": collection.count(),
        "distance_metric": metric,
        "documents": list_documents(),
    }
