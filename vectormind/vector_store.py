# vector_store.py
# Responsible for storing and querying embeddings using ChromaDB.

import chromadb # type: ignore
from chromadb.api.types import Metadata, QueryResult

COLLECTION_NAME = "vectormind"

_client = None
_collection = None


def _get_collection():
    global _client, _collection

    if _collection is None:
        _client = chromadb.PersistentClient(path="./chroma_db")
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

    collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)


def query_similar(query_embedding: list[float], k: int = 3) -> QueryResult:
    """Return the k most similar chunks to the given query embedding."""
    collection = _get_collection()
    return collection.query(query_embeddings=[query_embedding], n_results=k)
