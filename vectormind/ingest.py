# ingest.py
# Responsible for loading raw documents from disk or other sources.

import re
from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent / "data" / "docs"
SUPPORTED_EXTENSIONS = {".txt", ".md"}
UPLOAD_SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# ---------------------------------------------------------------------------
# Corruption detection
# ---------------------------------------------------------------------------

def is_probably_corrupted(text: str) -> bool:
    """Return True if the text shows multiple signs of MDX/React serialized output."""
    indicators = [
        "static/chunks/",
        "__next_f.push",
        '"children":',
        '"props":',
        'self.__next',
    ]
    score = sum(1 for indicator in indicators if indicator in text)
    return score >= 2


# ---------------------------------------------------------------------------
# Cleaning functions
# ---------------------------------------------------------------------------

def clean_mdx_content(text: str) -> str:
    """Remove React/MDX node syntax from lightly-contaminated text."""
    cleaned = text

    cleaned = re.sub(r'\["\$".*?\]', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'\{[^}]*\}', '', cleaned)
    cleaned = re.sub(r'"+', '', cleaned)
    cleaned = re.sub(r',+', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # Safe fallback: if cleaning removes too much from a normal doc, keep original
    if len(cleaned) < 50:
        return text

    return cleaned


def aggressive_clean(text: str) -> str:
    """Aggressively strip Next.js / React serialization artifacts from heavily-corrupted text.

    Returns the cleaned string, or an empty string if insufficient text remains.
    Callers must check the return value and skip ingestion when empty.
    """
    cleaned = text

    cleaned = re.sub(r'static/chunks/[^"]+', '', cleaned)
    cleaned = re.sub(r'self\.__next_f\.push\(.*?\)', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'\{.*?\}', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'\[.*?\]', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'http[s]?://\S+', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned


def clean_text(text: str) -> str | None:
    """
    Apply the appropriate cleaning strategy based on contamination level.
    This is the single entry point called before every chunking operation.

    Returns cleaned text, or None if the document is corrupted and cannot be
    salvaged — indicating the caller should skip ingestion entirely.
    """
    if is_probably_corrupted(text):
        cleaned = aggressive_clean(text)
        if len(cleaned) < 100:
            return None  # corrupted and unrecoverable — skip this document
        return cleaned

    cleaned = clean_mdx_content(text)
    if len(cleaned) < 50:
        return text  # safe fallback for normal docs with little/no contamination
    return cleaned


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_documents(docs_dir: Path = DOCS_DIR) -> list[dict]:
    """Load all supported documents from docs_dir.

    Returns a list of dicts with keys:
        - text: cleaned document contents
        - source: filename

    Corrupted documents that cannot be cleaned are skipped entirely.
    """
    documents = []

    for path in sorted(docs_dir.iterdir()):
        if path.suffix not in SUPPORTED_EXTENSIONS:
            continue
        raw = path.read_text(encoding="utf-8")
        text = clean_text(raw)
        if text is None:
            print(f"[ingest] SKIP {path.name} — corrupted content, no usable text after cleaning")
            continue
        documents.append({"text": text, "source": path.name})

    return documents


def load_document(file_path: Path) -> str:
    """Read text content from a single .txt, .md, or .pdf file."""
    ext = file_path.suffix.lower()

    if ext in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8", errors="replace")

    if ext == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError:
            raise ImportError("pypdf is required for PDF ingestion: pip install pypdf")

        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    raise ValueError(f"Unsupported file type: {ext!r}")


def ingest_document(file_path: Path) -> int:
    """
    Full ingestion pipeline for a single uploaded document.
    Chunks, embeds, and stores the document in the vector DB.
    Returns the number of chunks added.
    """
    from vectormind.chunk import chunk_documents
    from vectormind.embed import embed_chunks
    from vectormind.vector_store import store_embeddings
    from vectormind.retrieve import invalidate_bm25_cache

    raw_text = load_document(file_path)
    text = clean_text(raw_text)  # cleaning applied BEFORE chunking

    if text is None:
        print(f"[ingest] SKIP {file_path.name} — corrupted content, no usable text after cleaning")
        return 0

    doc = {"text": text, "source": file_path.name}
    chunks = chunk_documents([doc], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    if not chunks:
        return 0

    embedded = embed_chunks(chunks)
    store_embeddings(embedded)
    invalidate_bm25_cache()

    return len(chunks)
