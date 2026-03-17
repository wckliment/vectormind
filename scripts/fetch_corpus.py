import os
import urllib.request

DEST = "corpus_downloads"

DOCS = {

    # --------------------------------------------------
    # AI / Retrieval Research Papers
    # --------------------------------------------------

    "colbertv2_paper.pdf": "https://arxiv.org/pdf/2212.10496.pdf",
    "rag_paper.pdf": "https://arxiv.org/pdf/2005.11401.pdf",
    "fid_paper.pdf": "https://arxiv.org/pdf/2007.01282.pdf",
    "ance_paper.pdf": "https://arxiv.org/pdf/2101.00113.pdf",
    "gtr_paper.pdf": "https://arxiv.org/pdf/2205.09765.pdf",

    "dpr_paper.pdf": "https://arxiv.org/pdf/2004.04906.pdf",
    "realm_paper.pdf": "https://arxiv.org/pdf/2002.08909.pdf",
    "contriever_paper.pdf": "https://arxiv.org/pdf/2112.09118.pdf",
    "splade_paper.pdf": "https://arxiv.org/pdf/2109.10086.pdf",
    "sentence_bert_paper.pdf": "https://arxiv.org/pdf/1908.10084.pdf",

    # --------------------------------------------------
    # Transformer / LLM Papers
    # --------------------------------------------------

    "attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762.pdf",
    "bert_paper.pdf": "https://arxiv.org/pdf/1810.04805.pdf",
    "gpt3_paper.pdf": "https://arxiv.org/pdf/2005.14165.pdf",
    "t5_paper.pdf": "https://arxiv.org/pdf/1910.10683.pdf",
    "flan_paper.pdf": "https://arxiv.org/pdf/2210.11416.pdf",

    # --------------------------------------------------
    # FastAPI Documentation
    # --------------------------------------------------

    "fastapi_dependency_injection.md":
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/dependencies/index.md",

    "fastapi_background_tasks.md":
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/background-tasks.md",

    "fastapi_security.md":
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/security/index.md",

    # --------------------------------------------------
    # LangChain Documentation
    # --------------------------------------------------

    "langchain_retrievers.md":
        "https://python.langchain.com/docs/modules/data_connection/retrievers/",

    "langchain_vectorstores.md":
        "https://python.langchain.com/docs/modules/data_connection/vectorstores/",

    "langchain_agents.md":
        "https://python.langchain.com/docs/modules/agents/",

    # --------------------------------------------------
    # LlamaIndex Documentation (stable GitHub sources)
    # --------------------------------------------------

    "llamaindex_retrieval.md":
        "https://raw.githubusercontent.com/run-llama/llama_index/main/README.md",

    "llamaindex_query_engine.md":
        "https://raw.githubusercontent.com/run-llama/llama_index/main/README.md",

    "llamaindex_embeddings.md":
        "https://raw.githubusercontent.com/run-llama/llama_index/main/README.md",

    # --------------------------------------------------
    # Vector Database Documentation
    # --------------------------------------------------

    "pinecone_indexing.md":
        "https://raw.githubusercontent.com/pinecone-io/examples/master/README.md",

    "pinecone_semantic_search.md":
        "https://www.pinecone.io/learn/semantic-search/",

    "pinecone_hybrid_search.md":
        "https://www.pinecone.io/learn/hybrid-search/",

    "weaviate_hybrid_search.md":
        "https://raw.githubusercontent.com/weaviate/weaviate/main/README.md",

    "weaviate_schema.md":
        "https://raw.githubusercontent.com/weaviate/weaviate/main/README.md",

    "weaviate_modules.md":
        "https://weaviate.io/developers/weaviate/modules",

    "qdrant_vector_search.md":
        "https://raw.githubusercontent.com/qdrant/qdrant/master/README.md",

    "qdrant_payload_filtering.md":
        "https://qdrant.tech/documentation/concepts/filtering/",

    # --------------------------------------------------
    # OpenAI Cookbook Examples
    # --------------------------------------------------

    "rag_with_embeddings.md":
        "https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/Question_answering_using_embeddings.ipynb",

    "semantic_search.md":
        "https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/Semantic_text_search_using_embeddings.ipynb",

    "embedding_best_practices.md":
        "https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/Embedding_long_inputs.ipynb",

    # --------------------------------------------------
    # Semantic Search Tutorials
    # --------------------------------------------------

    "semantic_search_tutorial.md":
        "https://www.pinecone.io/learn/semantic-search/",

    "hybrid_search_tutorial.md":
        "https://www.pinecone.io/learn/hybrid-search/",

    "vector_database_tutorial.md":
        "https://www.pinecone.io/learn/vector-database/",
}


def download_file(url: str, path: str):
    try:
        urllib.request.urlretrieve(url, path)
        print(f"✓ downloaded {path}")
    except Exception as e:
        print(f"✗ failed {url}: {e}")


def main():
    os.makedirs(DEST, exist_ok=True)

    print("\nDownloading VectorMind corpus...\n")

    downloaded = 0
    skipped = 0

    for filename, url in DOCS.items():
        path = os.path.join(DEST, filename)

        if os.path.exists(path):
            print(f"• skipping {filename} (already exists)")
            skipped += 1
            continue

        print(f"Downloading {filename}...")
        download_file(url, path)
        downloaded += 1

    print("\nDownload complete.")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped: {skipped}")
    print(f"Saved to: {DEST}")


if __name__ == "__main__":
    main()