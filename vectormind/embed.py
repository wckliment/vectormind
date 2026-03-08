# embed.py
# Responsible for generating vector embeddings from text chunks.

import os

from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Add an 'embedding' field to each chunk by calling the OpenAI embeddings API."""
    client = _get_client()
    texts = [chunk["text"] for chunk in chunks]

    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)

    embedded = []
    for chunk, embedding_obj in zip(chunks, response.data):
        embedded.append({**chunk, "embedding": embedding_obj.embedding})

    return embedded
