from pathlib import Path

from vectormind.ingest import load_documents
from vectormind.chunk import chunk_documents
from vectormind.embed import embed_chunks
from vectormind.vector_store import store_embeddings

CORPUS_DIR = Path(__file__).parent.parent / "corpus_downloads"
DOCS_DIR   = Path(__file__).parent.parent / "data" / "docs"
KNOWLEDGE_DIR = Path(__file__).parent.parent / "data" / "knowledge"


def run_pipeline():
    print("Loading documents...")

    # TEMP: restrict ingestion to high-signal knowledge only
    seen: set[str] = set()
    docs: list[dict] = []

    directory = KNOWLEDGE_DIR

    if not directory.exists():
        print(f"[error] {directory} does not exist")
        return

    for doc in load_documents(directory):
        if doc["source"] not in seen:
            seen.add(doc["source"])
            docs.append(doc)

    print(f"Documents loaded: {len(docs)}")

    print("Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"Chunks created: {len(chunks)}")

    if chunks:
        print("\nExample chunk:")
        print({
            "source": chunks[0]["source"],
            "chunk_id": chunks[0]["chunk_id"],
            "text_preview": chunks[0]["text"][:120]
        })

    print("Generating embeddings...")
    embedded = embed_chunks(chunks)
    print(f"Embeddings created: {len(embedded)}")
    if embedded:
        print(f"Embedding length: {len(embedded[0]['embedding'])}")

    print("Storing embeddings...")
    store_embeddings(embedded)
    print("Embeddings stored.")


if __name__ == "__main__":
    run_pipeline()