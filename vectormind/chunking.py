# chunking.py
# Paragraph-aware chunking with overlap at paragraph boundaries.

import re


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into chunks aligned to paragraph boundaries.

    Paragraphs are merged until the chunk approaches `chunk_size`.
    The last paragraph of each chunk is prepended to the next chunk
    to preserve contextual continuity.

    Oversized paragraphs are split by sentence boundaries.
    """
    text = re.sub(r" +", " ", text.strip())
    text = re.sub(r"(?m)^\s*\*\s+", "\n\n* ", text)
    text = re.sub(r"(?m)^\s*-\s+", "\n\n- ", text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    # Build raw chunks (list of paragraph lists)
    raw_chunks: list[list[str]] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        if len(para) > chunk_size:
            # Flush current chunk before handling oversized paragraph
            if current:
                raw_chunks.append(current)
                current = []
                current_len = 0
            # Split by sentence boundaries instead of raw characters
            sentences = re.split(r"(?<=[.!?])\s+", para)
            bucket: str = ""
            for sentence in sentences:
                added_len = len(sentence) + (1 if bucket else 0)
                if bucket and len(bucket) + added_len > chunk_size:
                    raw_chunks.append([bucket])
                    bucket = sentence
                else:
                    bucket = (bucket + " " + sentence).strip() if bucket else sentence
            if bucket:
                raw_chunks.append([bucket])
            continue

        added_len = len(para) + (2 if current else 0)
        if current_len + added_len > chunk_size:
            raw_chunks.append(current)
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += added_len

    if current:
        raw_chunks.append(current)

    # Apply paragraph-level overlap: prepend the last paragraph of the
    # previous chunk to the start of the next chunk.
    result: list[str] = []
    for i, chunk_paras in enumerate(raw_chunks):
        if i > 0:
            prev_last = raw_chunks[i - 1][-1]
            chunk_paras = [prev_last] + chunk_paras
        result.append("\n\n".join(chunk_paras))

    return result
