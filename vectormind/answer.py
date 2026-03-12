# answer.py
# Responsible for sending prompts to the LLM and returning grounded answers.

import os
import re

from openai import OpenAI

from vectormind.retrieve import retrieve, rerank_chunks

COMPRESS_TOP_N = 7


def compress_context(query: str, documents: list[str]) -> list[str]:
    """Extract the most query-relevant sentences across all documents."""
    query_tokens = set(query.lower().split())
    scored: list[tuple[int, str]] = []

    for doc in documents:
        for sentence in re.split(r"(?<=[.!?])\s+", doc):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) < 10:
                continue
            if sentence.strip() in {"*", "-", "•"}:
                continue
            score = len(query_tokens & set(sentence.lower().split()))
            scored.append((score, sentence))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:COMPRESS_TOP_N]]


def pack_context(sentences: list[str], max_sentences: int = 5) -> list[str]:
    """Remove duplicate and near-duplicate sentences, keeping the most informative ones."""
    sentences = list(dict.fromkeys(sentences))
    packed: list[str] = []

    for s in sentences:
        tokens = set(re.findall(r"\w+", s.lower()))
        duplicate = False
        for existing in packed:
            existing_tokens = set(re.findall(r"\w+", existing.lower()))
            intersection = tokens & existing_tokens
            union = tokens | existing_tokens
            if len(intersection) / max(len(union), 1) > 0.7:
                duplicate = True
                break
        if not duplicate:
            packed.append(s)
        if len(packed) >= max_sentences:
            break

    return packed


LLM_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = "You are a helpful assistant that answers questions using the provided context."
DISTANCE_THRESHOLD = 0.85

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def rewrite_query(query: str, history: list) -> str:
    """Rewrite the user's latest question into a standalone search query using conversation history."""
    client = _get_client()

    history_text = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in history
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite the user's latest question so it can be understood without "
                    "the conversation history. Only output the rewritten query, nothing else."
                ),
            },
            {
                "role": "user",
                "content": f"Conversation history:\n{history_text}\n\nLatest question: {query}",
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()


def answer_question(query: str, k: int = 10, history: list | None = None) -> dict:
    """Run the full RAG pipeline and return prepared documents for streaming."""
    retrieval_query = rewrite_query(query, history) if history else query
    results = retrieve(retrieval_query, k=k)

    distance: float = results.get("distances", [[float("inf")]])[0][0]
    if distance > DISTANCE_THRESHOLD:
        return {"documents": [], "sources": []}

    documents: list[str] = results.get("documents", [[]])[0]
    metadatas: list[dict] = results.get("metadatas", [[]])[0]
    sources: list[str] = list(dict.fromkeys(m["source"] for m in metadatas if "source" in m))

    documents = rerank_chunks(query, documents)
    documents = compress_context(query, documents)
    documents = pack_context(documents)

    return {
        "documents": documents,
        "sources": sources,
    }


def stream_answer(query: str, documents: list[str], history: list | None = None):
    """Stream the LLM response token-by-token using the prepared RAG context."""
    client = _get_client()

    context = "\n\n".join(
        f"Context {i + 1}:\n{doc}"
        for i, doc in enumerate(documents)
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        messages.extend(history)

    messages.append({
        "role": "user",
        "content": f"""Use the context below to answer the user's question.

Question:
{query}

Context:
{context}
"""
    })

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0,
        stream=True
    )

    for chunk in response:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content
