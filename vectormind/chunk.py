# chunk.py
# Responsible for splitting documents into smaller chunks for embedding.


def chunk_document(doc: dict, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Split a single document into overlapping character-based chunks."""
    text = doc["text"]
    source = doc["source"]
    chunks = []
    step = chunk_size - overlap
    chunk_id = 0

    for start in range(0, len(text), step):
        chunk_text = text[start : start + chunk_size]
        if not chunk_text.strip():
            continue
        chunks.append({"text": chunk_text, "source": source, "chunk_id": chunk_id})
        chunk_id += 1

    return chunks


def chunk_documents(
    documents: list[dict], chunk_size: int = 500, overlap: int = 50
) -> list[dict]:
    """Chunk all documents and return a flat list of chunks."""
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))
    return all_chunks
