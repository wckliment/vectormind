# ingest.py
# Responsible for loading raw documents from disk or other sources.

from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent / "data" / "docs"
SUPPORTED_EXTENSIONS = {".txt", ".md"}


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
