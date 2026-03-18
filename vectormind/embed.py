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


BATCH_SIZE = 100


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Add an 'embedding' field to each chunk by calling the OpenAI embeddings API.

    Sends requests in batches of BATCH_SIZE to stay within the API token limit.
    Embedding order is preserved.
    """
    client = _get_client()
    texts = [chunk["text"] for chunk in chunks]

    all_embeddings: list = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend(response.data)

    return [{**chunk, "embedding": obj.embedding} for chunk, obj in zip(chunks, all_embeddings)]
