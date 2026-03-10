# document_loaders.py
# Loads raw text from .txt, .md, and .pdf files for the indexing pipeline.

from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_document(path: Path) -> str:
    """Read a document and return its text content.

    Supports .txt, .md, and .pdf files.
    Raises ValueError for unsupported file types.
    """
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")

    if suffix == ".pdf":
        from pypdf import PdfReader  # lazy import

        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    raise ValueError(f"Unsupported document type: {path.suffix}")


def load_documents(directory: Path) -> list[tuple[str, str]]:
    """Load all supported documents from a directory.

    Returns:
        A list of (filename, text) tuples sorted by filename.
    """
    results: list[tuple[str, str]] = []

    if not directory.exists():
        return results

    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            text = load_document(path)
            results.append((path.name, text))

    return results