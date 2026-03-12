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


<<<<<<< Updated upstream
def answer_question(query: str, k: int = 10) -> dict:
    """Retrieve relevant chunks and return a grounded answer from the LLM."""
    results = retrieve(query, k=k)

    distance: float = results.get("distances", [[float("inf")]])[0][0]
    if distance > DISTANCE_THRESHOLD:
        return {"answer": "I do not know based on the available documents.", "sources": []}

    documents: list[str] = results.get("documents", [[]])[0]
    metadatas: list[dict] = results.get("metadatas", [[]])[0]
    sources: list[str] = list(dict.fromkeys(m["source"] for m in metadatas if "source" in m))

    documents = rerank_chunks(query, documents)
    documents = compress_context(query, documents)
    documents = pack_context(documents)

    context = "\n\n".join(f"Context {i + 1}:\n{doc}" for i, doc in enumerate(documents))
    user_message = f"""Use the context below to answer the user's question.

Context:
{context}

Question:
{query}

Instructions:
- Write a short explanation answering the question.
- If relevant, include key points as bullet points.
- Use clear natural language.
- Base the answer only on the provided context.
"""
=======
def answer_question(query: str, k: int = 3) -> dict:
    """Retrieve relevant chunks and return documents for streaming."""
    results = retrieve(query, k=k)

    documents: list[str] = results.get("documents", [[]])[0]
    metadatas: list[dict] = results.get("metadatas", [[]])[0]
    sources: list[str] = list(dict.fromkeys(
        m["source"] for m in metadatas if "source" in m
    ))

    return {
        "documents": documents,
        "sources": sources,
    }


def stream_answer(query: str, documents: list[str]):
    """
    Stream the LLM response token-by-token using the prepared RAG context.
    """
>>>>>>> Stashed changes

    client = _get_client()

    context = "\n\n".join(
        f"Context {i + 1}:\n{doc}"
        for i, doc in enumerate(documents)
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""Use the context below to answer the user's question.

Question:
{query}

Context:
{context}
"""
            }
        ],
        temperature=0,
        stream=True
    )

<<<<<<< Updated upstream
    return {"answer": response.choices[0].message.content or "", "sources": sources}
=======
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content
>>>>>>> Stashed changes
