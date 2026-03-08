# VectorMind Architecture

VectorMind is a production-style Retrieval-Augmented Generation (RAG) system
for grounded question answering over documents.

## High-Level Flow

Documents
  ↓
Ingestion
  ↓
Chunking
  ↓
Embedding Generation
  ↓
Vector Index (Chroma)
  ↓
Retriever
  ↓
Context Builder
  ↓
LLM Response
  ↓
Guardrails
  ↓
Evaluation + Logging

## Components

### Ingestion (vectormind/ingest.py)
Loads raw documents from `data/docs/`.
Supports `.txt` and `.md` files.

Output:
- list of documents with text and source metadata.

### Chunking (vectormind/chunk.py)
Splits documents into smaller segments suitable for embedding.

Goals:
- preserve semantic meaning
- maintain manageable token sizes
- attach metadata (source, chunk_id)

### Embedding Generation (vectormind/embed.py)
Converts text chunks into vector embeddings using OpenAI embedding models.

### Vector Index (Chroma)
Stores embeddings and metadata for semantic retrieval.

Responsibilities:
- store vectors
- return top-k similar chunks
- maintain document metadata

### Retrieval (vectormind/retrieve.py)
Handles query embedding and vector similarity search.

Output:
- ranked document chunks
- similarity scores

### Context Builder (vectormind/prompt.py)
Assembles retrieved chunks into a prompt context for the LLM.

Responsibilities:
- enforce context window limits
- preserve source references

### Answer Generation (vectormind/answer.py)
Generates responses using retrieved context.

Key rule:
The model must answer **only from retrieved documents**.

### Guardrails (vectormind/guardrails.py)
Prevents unsupported or hallucinated responses.

Example behaviors:
- refuse if evidence is insufficient
- indicate uncertainty
- require supporting sources

### API Layer (api/server.py)
FastAPI service exposing the system.

Endpoints (planned):

POST /ask
POST /index
GET /health

### Evaluation (evaluation/)
Benchmark questions used to evaluate retrieval quality and response accuracy.

## Design Principles

VectorMind prioritizes:

- grounded answers
- transparent retrieval
- simple architecture
- modular pipeline components
- reproducibility

This project is intended to demonstrate practical AI engineering patterns,
not just a chatbot demo.