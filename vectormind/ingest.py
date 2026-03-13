# ingest.py
# Responsible for loading raw documents from disk or other sources.

from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent / "data" / "docs"
SUPPORTED_EXTENSIONS = {".txt", ".md"}
UPLOAD_SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_documents(docs_dir: Path = DOCS_DIR) -> list[dict]:
    """Load all supported documents from docs_dir.

    Returns a list of dicts with keys:
        - text: full document contents
        - source: filename
    """
    documents = []

    for path in sorted(docs_dir.iterdir()):
        if path.suffix not in SUPPORTED_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8")
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

    text = load_document(file_path)

    doc = {"text": text, "source": file_path.name}
    chunks = chunk_documents([doc], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    if not chunks:
        return 0

    embedded = embed_chunks(chunks)
    store_embeddings(embedded)
    invalidate_bm25_cache()

    return len(chunks)
