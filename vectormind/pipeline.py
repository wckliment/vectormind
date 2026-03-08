from vectormind.ingest import load_documents
from vectormind.chunk import chunk_documents
from vectormind.embed import embed_chunks
from vectormind.vector_store import store_embeddings


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