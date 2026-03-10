# chunking.py
# Splits raw text into overlapping chunks for embedding and indexing.


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks of approximately chunk_size characters.

    Each chunk overlaps the previous one by overlap characters to preserve
    context across chunk boundaries.
    """
    chunks = []
    step = chunk_size - overlap
    start = 0

    while start < len(text):
        chunk = text[start : start + chunk_size]
        chunks.append(chunk)

        if start + chunk_size >= len(text):
            break

        start += step

    return chunks
