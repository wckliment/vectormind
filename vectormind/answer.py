# answer.py
# Responsible for sending prompts to the LLM and returning grounded answers.

import os

from openai import OpenAI

from vectormind.retrieve import retrieve

LLM_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = "You are a helpful assistant that answers questions using the provided context."

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def answer_question(query: str, k: int = 3) -> str:
    """Retrieve relevant chunks and return a grounded answer from the LLM."""
    results = retrieve(query, k=k)

    documents: list[str] = results.get("documents", [[]])[0]
    context = "\n\n".join(documents)

    user_message = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"

    client = _get_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content or ""
