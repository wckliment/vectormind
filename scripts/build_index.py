# build_index.py
# Loads documents, chunks them, embeds them, and stores them in the vector database.

from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import chromadb

from vectormind.chunking import chunk_text
from vectormind.document_loaders import load_documents
from vectormind.embed import embed_chunks
from vectormind.vector_store import store_embeddings, COLLECTION_NAME


DOCUMENTS_DIR = Path("documents")
CHROMA_PATH = "./chroma_db"


def clear_collection() -> None:
    """Clear the existing Chroma collection safely."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Cleared existing collection: {COLLECTION_NAME}")
    except Exception:
        print(f"No existing collection found for: {COLLECTION_NAME}")

    client.get_or_create_collection(name=COLLECTION_NAME)


def build_index() -> None:
    """Load documents, chunk them, embed them, and store embeddings."""
    clear_collection()

    docs = load_documents(DOCUMENTS_DIR)

    if not docs:
        print(f"No supported documents found in {DOCUMENTS_DIR}/")
        return

    total_chunks = 0

    for filename, text in docs:
        print(f"\nLoaded {filename}")

        chunks_text = chunk_text(text)
        print(f"Created {len(chunks_text)} chunks")

        chunks = [
            {"text": chunk, "source": filename, "chunk_id": i}
            for i, chunk in enumerate(chunks_text)
        ]

        embedded = embed_chunks(chunks)
        store_embeddings(embedded)

        total_chunks += len(embedded)

        print(f"Inserted {len(embedded)} embeddings")

    print(f"\nTotal embeddings inserted: {total_chunks}")


if __name__ == "__main__":
    build_index()