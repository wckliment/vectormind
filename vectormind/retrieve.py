# retrieve.py
# Responsible for querying the vector store and returning relevant chunks.

import os

from openai import OpenAI
from chromadb.api.types import QueryResult

from vectormind.vector_store import query_similar, get_collection

EMBEDDING_MODEL = "text-embedding-3-small"
RERANK_MODEL = "gpt-4o-mini"
RERANK_TOP_K = 3

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


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


def keyword_search(query: str, documents: list[str], k: int = 10) -> list[str]:
    """Rank documents by keyword overlap with the query and return the top k."""
    query_tokens = query.lower().split()
    scored = sorted(
        documents,
        key=lambda doc: sum(token in doc.lower() for token in query_tokens),
        reverse=True,
    )
    return scored[:k]


def retrieve(query: str, k: int = 10) -> QueryResult:
    """Embed a query string, merge vector and keyword results, and return candidates."""
    client = _get_client()

    search_query = rewrite_query(query)

    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[search_query])
    query_embedding: list[float] = response.data[0].embedding

    results = query_similar(query_embedding, k=k)

    vector_docs: list[str] = results["documents"][0]
    metadatas: list[dict] = results["metadatas"][0]
    distances: list[float] = results["distances"][0]

    # Build index from doc text → position for remapping parallel arrays
    doc_index: dict[str, int] = {doc: i for i, doc in enumerate(vector_docs)}

    # Fetch full corpus and build a metadata lookup for keyword-only results
    corpus = get_collection().get()
    all_docs: list[str] = corpus.get("documents") or []
    all_metadatas: list[dict] = corpus.get("metadatas") or []
    corpus_meta: dict[str, dict] = dict(zip(all_docs, all_metadatas))

    keyword_docs = keyword_search(search_query, all_docs, k=k)
    combined_docs: list[str] = list(dict.fromkeys(vector_docs + keyword_docs))

    combined_metadatas: list[dict] = []
    combined_distances: list[float] = []
    for doc in combined_docs:
        if doc in doc_index:
            combined_metadatas.append(metadatas[doc_index[doc]])
            combined_distances.append(distances[doc_index[doc]])
        else:
            combined_metadatas.append(corpus_meta.get(doc, {}))
            combined_distances.append(1.0)  # neutral distance for keyword-only results

    results["documents"] = [combined_docs]
    results["metadatas"] = [combined_metadatas]
    results["distances"] = [combined_distances]

    return results
