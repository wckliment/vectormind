#!/usr/bin/env python3
"""
Retrieval evaluation script for VectorMind.

Usage:
    python evaluation/evaluate_retrieval.py --mode hybrid
    python evaluation/evaluate_retrieval.py --mode vector
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Allow imports from the repo root regardless of where the script is run from
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

BENCHMARK_PATH = Path(__file__).parent / "benchmark_questions.json"


def load_benchmark() -> list[dict]:
    with open(BENCHMARK_PATH) as f:
        return json.load(f)


def get_retrieved_sources(result: dict) -> list[str]:
    """Extract unique source filenames from a retrieval result."""
    metadatas: list[dict] = result.get("metadatas", [[]])[0]
    seen: set[str] = set()
    sources: list[str] = []
    for meta in metadatas:
        src = meta.get("source", "")
        if src and src not in seen:
            seen.add(src)
            sources.append(src)
    return sources


def run_hybrid(query: str, k: int) -> dict:
    from vectormind.retrieve import retrieve
    return retrieve(query, k=k)


def run_vector(query: str, k: int) -> dict:
    """Vector-only retrieval — embeds the query and calls ChromaDB directly."""
    import os
    from openai import OpenAI
    from vectormind.vector_store import query_similar

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query],
    )
    embedding: list[float] = response.data[0].embedding
    return query_similar(embedding, k=k)


def evaluate(mode: str, k: int) -> None:
    questions = load_benchmark()

    if mode not in ("hybrid", "vector"):
        print(f"Unknown mode: {mode!r}. Use --mode hybrid or --mode vector.")
        sys.exit(1)

    retrieve_fn = run_hybrid if mode == "hybrid" else run_vector

    print(f"Running evaluation (mode={mode}, top_k={k})...\n")

    correct = 0
    for item in questions:
        question: str = item["question"]
        expected: str = item["expected_source"]

        try:
            result = retrieve_fn(question, k)
            sources = get_retrieved_sources(result)
        except Exception as exc:
            sources = []
            print(f"  [ERROR] {exc}")

        passed = expected in sources
        if passed:
            correct += 1

        status = "PASS" if passed else "FAIL"
        print(f"Q: {question}")
        print(f"Expected: {expected}")
        print(f"Retrieved: {sources}")
        print(f"{status}\n")

    total = len(questions)
    accuracy = (correct / total * 100) if total else 0.0
    print(f"Total questions: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.0f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VectorMind retrieval quality.")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "vector"],
        default="hybrid",
        help="Retrieval mode to evaluate (default: hybrid)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of chunks to retrieve per question (default: 10)",
    )
    args = parser.parse_args()
    evaluate(args.mode, args.k)


if __name__ == "__main__":
    main()
