# server.py
# FastAPI application exposing the VectorMind RAG pipeline.

from dotenv import load_dotenv
load_dotenv()


from fastapi import FastAPI
from pydantic import BaseModel
from vectormind.answer import answer_question
from vectormind.retrieve import retrieve

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest) -> dict:
    results = retrieve(request.query)
    documents: list[str] = results.get("documents", [[]])[0]
    metadata: list[dict] = results.get("metadatas", [[]])[0]
    distances: list[float] = results.get("distances", [[]])[0]
    sources = [m["source"] for m in metadata if "source" in m]
    return {"query": request.query, "documents": documents, "sources": sources, "distances": distances}


@app.post("/query")
def query(request: QueryRequest) -> dict:
    return answer_question(request.query)
