# server.py
# FastAPI application exposing the VectorMind RAG pipeline.

from dotenv import load_dotenv
load_dotenv()


from fastapi import FastAPI
from pydantic import BaseModel
from vectormind.answer import answer_question

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query(request: QueryRequest) -> dict[str, str]:
    result = answer_question(request.query)
    return {"answer": result}
