#!/usr/bin/env python3
"""
Batch ingestion script for the corpus_downloads/ directory.

Usage:
    python scripts/ingest_corpus.py
    python scripts/ingest_corpus.py --corpus path/to/other/dir

Skips documents already indexed in the vector store.
Uses the existing ingest_document() pipeline — no reimplementation of
chunking, embedding, or storage logic.
"""

import argparse
import sys
from pathlib import Path

# Allow imports from the repo root regardless of where the script is run from.
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from vectormind.ingest import ingest_document, UPLOAD_SUPPORTED_EXTENSIONS
from vectormind.vector_store import list_sources

CORPUS_DIR = Path(__file__).parent.parent / "corpus_downloads"


def main(corpus_dir: Path) -> None:
    if not corpus_dir.exists():
        print(f"Error: corpus directory not found: {corpus_dir}")
        sys.exit(1)

    already_indexed = {entry["name"] for entry in list_sources()}

    candidates = sorted(
        p for p in corpus_dir.iterdir()
        if p.is_file() and p.suffix.lower() in UPLOAD_SUPPORTED_EXTENSIONS
    )

    if not candidates:
        print(f"No supported files found in {corpus_dir}")
        return

    total = len(candidates)
    ingested = 0
    skipped = 0

    for path in candidates:
        if path.name in already_indexed:
            print(f"Skipping {path.name} (already indexed)")
            skipped += 1
            continue

        print(f"Ingesting {path.name}")
        try:
            chunks = ingest_document(path)
            print(f"  ✓ indexed ({chunks} chunks)")
            ingested += 1
        except Exception as exc:
            print(f"  ✗ failed: {exc}")

    print(f"\nDone. {ingested} ingested, {skipped} skipped, {total} total.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch ingest corpus_downloads/ into VectorMind.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=CORPUS_DIR,
        help=f"Directory to ingest (default: {CORPUS_DIR})",
    )
    args = parser.parse_args()
    main(args.corpus)
