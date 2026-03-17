# server.py
# FastAPI application exposing the VectorMind RAG pipeline.

from dotenv import load_dotenv
load_dotenv()

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from vectormind.answer import answer_question, stream_answer
from vectormind.embed import embed_chunks
from vectormind.ingest import UPLOAD_SUPPORTED_EXTENSIONS, ingest_document
from vectormind.retrieve import retrieve, invalidate_bm25_cache
from vectormind.vector_store import get_collection, query_similar, list_sources, delete_source


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow local UI during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    history: list | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/documents")
def documents(limit: int = Query(default=10, ge=1)) -> dict:
    """Debug endpoint that returns a limited number of stored chunks."""
    collection = get_collection()
    result = collection.get(limit=limit)

    docs = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    ids = result.get("ids") or []

    return {
        "documents": docs,
        "metadatas": metadatas,
        "ids": ids,
    }


@app.get("/sources")
def sources() -> dict:
    """Return all unique document source names indexed in the vector store."""
    collection = get_collection()
    result = collection.get(include=["metadatas"])
    metadatas = result.get("metadatas") or []
    unique = list(dict.fromkeys(
        m["source"] for m in metadatas if m and "source" in m
    ))
    return {"sources": unique}


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest) -> dict:
    results = retrieve(request.query)

    documents: list[str] = results.get("documents", [[]])[0]
    metadata: list[dict] = results.get("metadatas", [[]])[0]
    distances: list[float] = results.get("distances", [[]])[0]

    sources = list(dict.fromkeys(
        m["source"] for m in metadata if "source" in m
    ))

    # Rich citations for UI deep-linking: one entry per unique (source, chunk_id) pair.
    seen: set[tuple] = set()
    citations: list[dict] = []
    for doc, meta in zip(documents, metadata):
        src = meta.get("source", "")
        chunk_id = int(meta.get("chunk_id", 0))
        key = (src, chunk_id)
        if key not in seen:
            seen.add(key)
            citations.append({"source": src, "chunk_id": chunk_id, "text": doc[:200]})

    return {
        "query": request.query,
        "documents": documents,
        "sources": sources,
        "citations": citations,
        "distances": distances,
    }


@app.post("/query")
def query(request: QueryRequest):
    """
    Streaming RAG endpoint.
    Runs the full pipeline and streams the LLM answer.
    """

    result = answer_question(request.query, history=request.history)
    documents = result.get("documents", [])
    labeled_chunks = result.get("labeled_chunks")

    if not documents:
        return StreamingResponse(
            iter(["I do not know based on the available documents."]),
            media_type="text/plain",
        )

    def generator():
        try:
            for token in stream_answer(
                request.query, documents, request.history, labeled_chunks=labeled_chunks
            ):
                yield token
        except Exception:
            yield "\n[Error: response generation failed]"

    return StreamingResponse(generator(), media_type="text/plain")


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    """Ingest an uploaded document into the vector store."""
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()

    if suffix not in UPLOAD_SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(UPLOAD_SUPPORTED_EXTENSIONS)}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        original_name = Path(filename).name
        tmp_path = tmp_path.rename(tmp_path.with_name(original_name))
        chunks_added = ingest_document(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return {"status": "success", "filename": file.filename, "chunks_added": chunks_added}


@app.get("/library")
def library() -> dict:
    """Return all indexed sources with their chunk counts."""
    return {"sources": list_sources()}


_CORPUS_DIRS = [
    Path(__file__).parent.parent / "corpus_downloads",
    Path(__file__).parent.parent / "data" / "docs",
]


@app.get("/documents/{source:path}")
def get_document(source: str):
    """Return the raw content of an indexed source document."""
    filename = Path(source).name  # strip any accidental path traversal
    suffix = Path(filename).suffix.lower()

    for corpus_dir in _CORPUS_DIRS:
        candidate = corpus_dir / filename
        if candidate.exists():
            if suffix == ".pdf":
                return FileResponse(
                    str(candidate),
                    media_type="application/pdf",
                    filename=filename,
                )
            content = candidate.read_text(encoding="utf-8", errors="replace")
            return {"source": filename, "content": content}

    raise HTTPException(status_code=404, detail=f"Document '{source}' not found.")


@app.delete("/documents/{source:path}")
def delete_document(source: str) -> dict:
    """Delete all chunks for a given source from the vector store."""
    removed = delete_source(source)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"Source '{source}' not found.")
    invalidate_bm25_cache()
    return {"status": "deleted", "source": source}


@app.post("/reindex")
def reindex() -> dict:
    """Invalidate the BM25 cache so it is rebuilt from current ChromaDB contents on next query."""
    invalidate_bm25_cache()
    return {"status": "ok", "message": "BM25 index invalidated and will rebuild on next retrieval."}


@app.post("/library-search")
def library_search(request: QueryRequest) -> dict:
    """Return matching document sources with a snippet from the best-matching chunk."""
    embedded = embed_chunks([{"text": request.query, "source": "", "chunk_id": 0}])
    embedding: list[float] = embedded[0]["embedding"]

    results = query_similar(embedding, k=20)

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]

    source_snippets: dict[str, str] = {}
    for doc, meta in zip(documents, metadatas):
        if not meta or "source" not in meta:
            continue
        src = str(meta["source"])
        if src not in source_snippets:
            snippet = doc.replace("\n", " ").strip()
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            source_snippets[src] = snippet

    return {
        "results": [
            {"source": src, "snippet": snippet}
            for src, snippet in source_snippets.items()
        ]
    }