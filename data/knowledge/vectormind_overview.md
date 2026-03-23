# Vectormind Overview

## Overview

Vectormind is a retrieval system built on top of a vector database. It is designed to store, index, and retrieve documents using embeddings and similarity search.

Vectormind enables retrieval-augmented generation (RAG) by allowing systems to query a corpus of documents and return the most relevant information based on semantic similarity.

Vectormind is responsible for the retrieval layer in a larger AI system. It does not generate answers itself. Instead, it provides relevant context to downstream systems.

---

## Core Concepts

### Embeddings

Vectormind converts text into embeddings using an embedding model. An embedding is a numerical vector that represents the meaning of a piece of text.

Similar pieces of text have similar embeddings. This allows Vectormind to perform semantic search instead of keyword matching.

---

### Vector Database

Vectormind stores embeddings in a vector database. Each document is split into smaller chunks, and each chunk is embedded and stored.

Each stored item includes:
- the text chunk
- the embedding vector
- metadata such as source and chunk_id

The vector database enables fast similarity search over all stored chunks.

---

### Document Chunking

Before embedding, documents are split into smaller chunks. Chunking allows Vectormind to retrieve specific parts of documents instead of entire files.

Each chunk is:
- a portion of the original document
- associated with a source file
- assigned a unique chunk_id

Chunking improves retrieval precision and reduces noise.

---

## How Retrieval Works

1. A query is received as input.
2. The query is converted into an embedding using the same embedding model.
3. The embedding is compared against stored embeddings in the vector database.
4. The top-k most similar chunks are returned.

These returned chunks represent the most relevant information for the query.

Vectormind does not interpret or summarize the results. It only retrieves them.

---

## Role in the System

Vectormind is part of a larger system architecture:

- Axon: execution engine that plans and runs tasks
- Vectormind: retrieval layer that provides relevant documents
- Verdict: evaluation layer that scores system performance

The flow is:

User Input → Axon → Vectormind → Axon → Verdict

Vectormind provides the knowledge needed for Axon to generate grounded outputs.

---

## Key Characteristics

- Semantic search using embeddings
- Vector database for fast retrieval
- Chunk-based document indexing
- Metadata tracking (source, chunk_id)
- Retrieval-only (no generation)

Vectormind is designed to be deterministic, observable, and testable as part of an evaluation-driven AI system.

---

## Summary

Vectormind is a vector-based retrieval system that enables semantic search over a document corpus.

It works by:
- converting text into embeddings
- storing embeddings in a vector database
- retrieving the most similar chunks for a given query

Vectormind provides the foundation for retrieval-augmented AI systems by supplying relevant context to downstream components.