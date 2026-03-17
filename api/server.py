# server.py
# FastAPI application exposing the VectorMind RAG pipeline.

from dotenv import load_dotenv
load_dotenv()

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from vectormind.answer import answer_question, stream_answer
from vectormind.embed import embed_chunks
from vectormind.ingest import UPLOAD_SUPPORTED_EXTENSIONS, ingest_document
from vectormind.retrieve import retrieve
from vectormind.vector_store import get_collection, query_similar


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

    return {
        "query": request.query,
        "documents": documents,
        "sources": sources,
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