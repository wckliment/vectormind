# answer.py
# Responsible for sending prompts to the LLM and returning grounded answers.

import os

from openai import OpenAI

from vectormind.retrieve import retrieve

LLM_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = "You are a helpful assistant that answers questions using the provided context."
DISTANCE_THRESHOLD = 0.85

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def answer_question(query: str, k: int = 3) -> dict:
    """Retrieve relevant chunks and return a grounded answer from the LLM."""
    results = retrieve(query, k=k)

    # DEBUG: inspect what the retriever returned
    print("\n--- RETRIEVAL DEBUG ---")
    print("Query:", query)
    print("Results:", results)
    print("-----------------------\n")

    distance: float = results.get("distances", [[float("inf")]])[0][0]
    if distance > DISTANCE_THRESHOLD:
        return {"answer": "I do not know based on the available documents.", "sources": []}

    documents: list[str] = results.get("documents", [[]])[0]
    metadatas: list[dict] = results.get("metadatas", [[]])[0]
    sources: list[str] = list(dict.fromkeys(m["source"] for m in metadatas if "source" in m))

    numbered_chunks = "\n\n".join(f"Context {i + 1}:\n{doc}" for i, doc in enumerate(documents))
    user_message = (
        "You must answer the question using ONLY the provided context. "
        "If the answer cannot be found in the context, say that you do not know.\n\n"
        f"{numbered_chunks}\n\n"
        f"Question:\n{query}\n\nAnswer:"
    )

    client = _get_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return {"answer": response.choices[0].message.content or "", "sources": sources}
