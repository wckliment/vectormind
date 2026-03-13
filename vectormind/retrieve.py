# retrieve.py
# Responsible for querying the vector store and returning relevant chunks.

import os
from typing import Optional

from openai import OpenAI
from rank_bm25 import BM25Okapi  # type: ignore
from chromadb.api.types import QueryResult

from vectormind.vector_store import query_similar, get_collection

EMBEDDING_MODEL = "text-embedding-3-small"
RERANK_MODEL = "gpt-4o-mini"
RERANK_TOP_K = 3

_client: Optional[OpenAI] = None

# BM25 index cache — rebuilt lazily, invalidated when corpus changes
_bm25_cache: Optional[tuple[BM25Okapi, list[str], list[dict]]] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def invalidate_bm25_cache() -> None:
    """Reset the cached BM25 index so it is rebuilt on the next request."""
    global _bm25_cache
    _bm25_cache = None


def build_bm25_index() -> tuple[BM25Okapi, list[str], list[dict]]:
    """Load all documents from the vector store and build a cached BM25 index."""
    global _bm25_cache
    if _bm25_cache is not None:
        return _bm25_cache

    collection = get_collection()
    result = collection.get()
    documents: list[str] = result.get("documents") or []
    metadatas: list[dict] = result.get("metadatas") or []

    doc_tokens = [doc.lower().split() for doc in documents]
    bm25_index = BM25Okapi(doc_tokens)

    _bm25_cache = (bm25_index, documents, metadatas)
    return _bm25_cache


def keyword_search(
    query: str,
    bm25_index: BM25Okapi,
    documents: list[str],
    metadatas: list[dict],
    k: int = 10,
) -> list[dict]:
    """Return the top-k documents ranked by BM25 score."""
    query_tokens = query.lower().split()
    scores: list[float] = list(bm25_index.get_scores(query_tokens))

    ranked = sorted(
        enumerate(scores), key=lambda x: x[1], reverse=True
    )[:k]

    max_score = ranked[0][1] if ranked and ranked[0][1] > 0 else 1.0

    return [
        {
            "document": documents[i],
            "metadata": metadatas[i],
            "score": score / max_score,  # normalize to [0, 1]
        }
        for i, score in ranked
        if score > 0
    ]


def rerank_chunks(query: str, documents: list[str]) -> list[str]:
    """Rerank all candidate chunks in a single LLM call and return the top 3."""
    client = _get_client()

    numbered = "\n\n".join(
        f"Document {i + 1}:\n{doc[:800]}"
        for i, doc in enumerate(documents)
    )
    prompt = (
        "You are ranking document chunks for relevance to a user query.\n\n"
        f"Query:\n{query}\n\n"
        f"Documents:\n{numbered}\n\n"
        "Return ONLY the document numbers of the three most relevant documents.\n"
        "Format: number, number, number\n"
        "Example: 2, 5, 1"
    )

    response = client.chat.completions.create(
        model=RERANK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content or ""

    indices: list[int] = []
    for token in raw.replace(",", " ").split():
        if token.isdigit():
            idx = int(token) - 1  # convert 1-based to 0-based
            if 0 <= idx < len(documents):
                indices.append(idx)
        if len(indices) == RERANK_TOP_K:
            break

    # Fall back to original order if parsing fails
    if not indices:
        indices = list(range(min(RERANK_TOP_K, len(documents))))

    return [documents[i] for i in indices]


def rewrite_query(query: str) -> str:
    """Rewrite a user query into a search-optimized form using the LLM."""
    client = _get_client()
    prompt = (
        "You are improving a user query for document retrieval.\n\n"
        f"Original query:\n{query}\n\n"
        "Rewrite the query so it contains the important keywords, "
        "technical terms, and concepts needed for searching documents.\n\n"
        "Return only the rewritten search query."
    )
    response = client.chat.completions.create(
        model=RERANK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return (response.choices[0].message.content or query).strip()


def retrieve(query: str, k: int = 10) -> QueryResult:
    """Embed a query string, merge vector and BM25 results, rerank, and return top-k."""
    client = _get_client()

    search_query = rewrite_query(query)

    # --- Vector search ---
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[search_query])
    query_embedding: list[float] = response.data[0].embedding
    vector_result = query_similar(query_embedding, k=k)

    vector_docs: list[str] = vector_result["documents"][0]
    vector_metadatas: list[dict] = vector_result["metadatas"][0]
    vector_distances: list[float] = vector_result["distances"][0]

    # Convert L2/cosine distance → similarity score in [0, 1]
    vector_scores: dict[str, float] = {
        doc: 1.0 / (1.0 + dist)
        for doc, dist in zip(vector_docs, vector_distances)
    }
    # --- BM25 keyword search ---
    bm25_index, all_docs, all_metadatas = build_bm25_index()
    kw_results = keyword_search(search_query, bm25_index, all_docs, all_metadatas, k=k)

    keyword_scores: dict[str, float] = {
        r["document"]: r["score"] for r in kw_results
    }

    # --- Merge & rerank (deduplicate by (text, source)) ---
    seen: set[tuple[str, str]] = set()
    unique_candidates: list[tuple[str, dict]] = []

    for doc, meta in zip(vector_docs, vector_metadatas):
        key = (doc, meta.get("source", ""))
        if key not in seen:
            seen.add(key)
            unique_candidates.append((doc, meta))

    for r in kw_results:
        key = (r["document"], r["metadata"].get("source", ""))
        if key not in seen:
            seen.add(key)
            unique_candidates.append((r["document"], r["metadata"]))

    scored: list[tuple[str, dict, float]] = []
    for doc, meta in unique_candidates:
        v_score = vector_scores.get(doc, 0.0)
        k_score = keyword_scores.get(doc, 0.0)
        combined = 0.7 * v_score + 0.3 * k_score
        scored.append((doc, meta, combined))

    scored.sort(key=lambda x: x[2], reverse=True)
    top_docs = scored[:k]

    combined_docs: list[str] = []
    combined_metadatas: list[dict] = []
    combined_distances: list[float] = []

    for doc, meta, combined_score in top_docs:
        combined_docs.append(doc)
        combined_metadatas.append(meta)
        # Convert combined similarity back to a distance-like value for API compat
        combined_distances.append(1.0 - combined_score)

    vector_result["documents"] = [combined_docs]
    vector_result["metadatas"] = [combined_metadatas]
    vector_result["distances"] = [combined_distances]

    return vector_result
