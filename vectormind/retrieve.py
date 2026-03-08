# retrieve.py
# Responsible for querying the vector store and returning relevant chunks.

import os

from openai import OpenAI
from chromadb.api.types import QueryResult

from vectormind.vector_store import query_similar

EMBEDDING_MODEL = "text-embedding-3-small"

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def retrieve(query: str, k: int = 3) -> QueryResult:
    """Embed a query string and return the k most similar chunks from the vector store."""
    client = _get_client()

    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    query_embedding: list[float] = response.data[0].embedding

    return query_similar(query_embedding, k=k)
