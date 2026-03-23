# retrieve.py
# Responsible for querying the vector store and returning relevant chunks.

import os
from typing import Optional

from openai import OpenAI
from rank_bm25 import BM25Okapi  # type: ignore
from chromadb.api.types import QueryResult

from vectormind.vector_store import get_collection

EMBEDDING_MODEL = "text-embedding-3-small"
RERANK_MODEL = "gpt-4o-mini"
RERANK_TOP_K = 3
MMR_LAMBDA = 0.7  # relevance vs. diversity tradeoff

_client: Optional[OpenAI] = None

# BM25 index cache: (index, documents, metadatas, embeddings)
# Rebuilt lazily; invalidated when new documents are ingested.
_bm25_cache: Optional[tuple[Optional[BM25Okapi], list[str], list[dict], list[list[float]]]] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

def invalidate_bm25_cache() -> None:
    """Reset the cached BM25 index so it is rebuilt on the next request."""
    global _bm25_cache
    _bm25_cache = None


def build_bm25_index() -> tuple[Optional[BM25Okapi], list[str], list[dict], list[list[float]]]:
    """
    Load all documents from the vector store and build a cached BM25 index.
    Also caches per-document embeddings so MMR can use them for diversity scoring.
    """
    global _bm25_cache
    if _bm25_cache is not None:
        return _bm25_cache

    collection = get_collection()
    result = collection.get(include=["embeddings", "documents", "metadatas"])

    documents: list[str] = result.get("documents") or []
    metadatas: list[dict] = result.get("metadatas") or []
    embeddings = result.get("embeddings")
    if embeddings is None:
        embeddings = []

    if len(documents) < 5:
        _bm25_cache = (None, documents, metadatas, embeddings)
        return _bm25_cache

    doc_tokens = [doc.lower().split() for doc in documents]
    bm25_index = BM25Okapi(doc_tokens)

    _bm25_cache = (bm25_index, documents, metadatas, embeddings)
    return _bm25_cache


# ---------------------------------------------------------------------------
# Keyword search
# ---------------------------------------------------------------------------

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

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

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


# ---------------------------------------------------------------------------
# MMR helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns 0 if either is zero-length."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def mmr_select(
    candidates: list[dict],
    query_embedding: list[float],
    k: int = 5,
    lambda_param: float = MMR_LAMBDA,
) -> list[dict]:
    """
    Select k diverse candidates using Maximal Marginal Relevance.

    At each step, pick the candidate that maximises:
        MMR = λ * relevance − (1 − λ) * max_sim_to_selected

    where relevance = candidate["score"] (pre-computed hybrid score)
    and   max_sim_to_selected = cosine similarity to the most similar
          already-selected document.

    Falls back gracefully when a candidate has no embedding.
    """
    if not candidates:
        return []

    selected: list[dict] = []
    remaining = list(candidates)

    while len(selected) < k and remaining:
        best_idx = -1
        best_score = float("-inf")

        for i, candidate in enumerate(remaining):
            relevance = candidate["score"]
            emb = candidate.get("embedding")
            if emb is None:
                emb = []

            if selected and emb is not None and len(emb) > 0:
                sim_to_selected = max(
                    cosine_similarity(emb, s["embedding"])
                    for s in selected
                    if s.get("embedding") is not None
                )
            else:
                sim_to_selected = 0.0

            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected


# ---------------------------------------------------------------------------
# LLM helpers (unchanged)
# ---------------------------------------------------------------------------

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
            idx = int(token) - 1
            if 0 <= idx < len(documents):
                indices.append(idx)
        if len(indices) == RERANK_TOP_K:
            break

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


# ---------------------------------------------------------------------------
# Main retrieval entry point
# ---------------------------------------------------------------------------

def retrieve(query: str, k: int = 10) -> QueryResult:
    """
    Full hybrid retrieval pipeline:

    1. Rewrite query
    2. Embed query
    3. Vector search  (with embeddings included for MMR)
    4. BM25 keyword search
    5. Merge + deduplicate by (text, source)
    6. Score = 0.7 * vector_sim + 0.3 * bm25
    7. MMR re-rank for diversity
    8. Return top-k in QueryResult format
    """
    client = _get_client()

    search_query = rewrite_query(query)

    # ------------------------------------------------------------------
    # 1. Embed the (rewritten) query
    # ------------------------------------------------------------------
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[search_query])
    query_embedding: list[float] = response.data[0].embedding

    # ------------------------------------------------------------------
    # 2. Vector search — include embeddings so MMR can use them
    # ------------------------------------------------------------------
    collection = get_collection()
    raw_vector: QueryResult = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["embeddings", "documents", "metadatas", "distances"],
    )

    vector_docs: list[str] = raw_vector["documents"][0]
    vector_metadatas: list[dict] = raw_vector["metadatas"][0]
    vector_distances: list[float] = raw_vector["distances"][0]
    vector_embeddings: list[list[float]] = (raw_vector.get("embeddings") or [[]])[0]

    # distance → similarity in [0, 1]
    vector_scores: dict[str, float] = {
        doc: 1.0 / (1.0 + dist)
        for doc, dist in zip(vector_docs, vector_distances)
    }

    # (doc_text, source) → embedding for fast lookup during MMR
    embedding_map: dict[tuple[str, str], list[float]] = {
        (doc, meta.get("source", "")): emb
        for doc, meta, emb in zip(vector_docs, vector_metadatas, vector_embeddings)
    }

    # ------------------------------------------------------------------
    # 3. BM25 keyword search (corpus embeddings cached alongside index)
    # ------------------------------------------------------------------
    bm25_index, all_docs, all_metadatas, all_embeddings = build_bm25_index()
    kw_results = []
    keyword_scores: dict[str, float] = {}

    if bm25_index is not None:
        kw_results = keyword_search(search_query, bm25_index, all_docs, all_metadatas, k=k)
        keyword_scores = {
            r["document"]: r["score"] for r in kw_results
        }

    # Extend embedding_map with corpus embeddings (fills in BM25-only candidates)
    for doc, meta, emb in zip(all_docs, all_metadatas, all_embeddings):
        key = (doc, meta.get("source", ""))
        if key not in embedding_map:
            embedding_map[key] = emb

    # ------------------------------------------------------------------
    # 4. Merge & deduplicate by (text, source)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 5. Compute hybrid scores and attach embeddings for MMR
    # ------------------------------------------------------------------
    candidates: list[dict] = []
    for doc, meta in unique_candidates:
        v_score = vector_scores.get(doc, 0.0)
        k_score = keyword_scores.get(doc, 0.0)
        combined = 0.7 * v_score + 0.3 * k_score
        key = (doc, meta.get("source", ""))
        candidates.append({
            "document": doc,
            "metadata": meta,
            "embedding": embedding_map.get(key, []),
            "score": combined,
        })

    # ------------------------------------------------------------------
    # 6. MMR re-rank — replaces naive sorted top-k
    # ------------------------------------------------------------------
    selected = mmr_select(candidates, query_embedding, k=k, lambda_param=MMR_LAMBDA)

    # ------------------------------------------------------------------
    # 7. Unpack into the QueryResult format expected by the rest of the pipeline
    # ------------------------------------------------------------------
    combined_docs: list[str] = []
    combined_metadatas: list[dict] = []
    combined_distances: list[float] = []

    for item in selected:
        combined_docs.append(item["document"])
        combined_metadatas.append(item["metadata"])
        # Convert similarity back to a distance-like value for API compatibility
        combined_distances.append(1.0 - item["score"])

    raw_vector["documents"] = [combined_docs]
    raw_vector["metadatas"] = [combined_metadatas]
    raw_vector["distances"] = [combined_distances]

    return raw_vector
