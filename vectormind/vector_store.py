# vector_store.py
# Responsible for storing and querying embeddings using ChromaDB.

import chromadb  # type: ignore
from chromadb.api.types import Metadata, QueryResult
from pathlib import Path

COLLECTION_NAME = "vectormind"

_client = None
_collection = None


def _get_collection():
    global _client, _collection

    if _collection is None:
        # Use absolute path anchored to this repo (NOT current working directory)
        BASE_DIR = Path(__file__).resolve().parent.parent
        CHROMA_PATH = BASE_DIR / "chroma_db"

        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = _client.get_or_create_collection(name=COLLECTION_NAME)

    return _collection


def store_embeddings(embedded_chunks: list[dict]) -> None:
    """Store embedded chunks in the Chroma collection."""
    collection = _get_collection()

    ids = [f"{chunk['source']}_{chunk['chunk_id']}" for chunk in embedded_chunks]
    documents = [chunk["text"] for chunk in embedded_chunks]
    embeddings = [chunk["embedding"] for chunk in embedded_chunks]
    metadatas: list[Metadata] = [
        {"source": chunk["source"], "chunk_id": chunk["chunk_id"]}
        for chunk in embedded_chunks
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def query_similar(query_embedding: list[float], k: int = 3) -> QueryResult:
    """Return the k most similar chunks to the given query embedding."""
    collection = _get_collection()
    return collection.query(query_embeddings=[query_embedding], n_results=k)


def get_collection():
    """Public accessor used by debugging endpoints."""
    return _get_collection()


def list_sources() -> list[dict]:
    """Return all unique sources and their chunk counts."""
    collection = _get_collection()
    result = collection.get(include=["metadatas"])
    metadatas: list[dict] = result.get("metadatas") or []

    counts: dict[str, int] = {}
    for meta in metadatas:
        if meta and "source" in meta:
            src = str(meta["source"])
            counts[src] = counts.get(src, 0) + 1

    return [{"name": src, "chunks": count} for src, count in counts.items()]


def delete_source(source: str) -> int:
    """Delete all chunks belonging to source. Returns the number of chunks removed."""
    collection = _get_collection()

    result = collection.get(where={"source": source}, include=[])
    ids: list[str] = result.get("ids") or []

    if ids:
        collection.delete(ids=ids)

    return len(ids)