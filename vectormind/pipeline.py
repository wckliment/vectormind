from vectormind.ingest import load_documents
from vectormind.chunk import chunk_documents


def run_pipeline():
    print("Loading documents...")
    docs = load_documents()
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


if __name__ == "__main__":
    run_pipeline()