import os
import urllib.request

DOCS = {
    # Retrieval / ML papers
    "dpr_paper.pdf": "https://arxiv.org/pdf/2004.04906.pdf",
    "realm_paper.pdf": "https://arxiv.org/pdf/2002.08909.pdf",
    "bert_paper.pdf": "https://arxiv.org/pdf/1810.04805.pdf",
    "transformer_paper.pdf": "https://arxiv.org/pdf/1706.03762.pdf",
    "contriever_paper.pdf": "https://arxiv.org/pdf/2112.09118.pdf",
    "splade_paper.pdf": "https://arxiv.org/pdf/2109.10086.pdf",
    "sentence_bert_paper.pdf": "https://arxiv.org/pdf/1908.10084.pdf",

    # Documentation
    "fastapi_readme.md": "https://raw.githubusercontent.com/tiangolo/fastapi/master/README.md",
    "langchain_readme.md": "https://raw.githubusercontent.com/langchain-ai/langchain/master/README.md",
    "openai_cookbook_embeddings.md": "https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/Embedding_long_inputs.ipynb",

    # Concept articles
    "semantic_search_intro.md": "https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/vector_databases/README.md",
    "vector_database_intro.md": "https://raw.githubusercontent.com/weaviate/weaviate/main/README.md",
}

DEST = "corpus_downloads"


def download_file(url: str, path: str):
    try:
        urllib.request.urlretrieve(url, path)
        print(f"✓ downloaded {path}")
    except Exception as e:
        print(f"✗ failed {url}: {e}")


def main():
    os.makedirs(DEST, exist_ok=True)

    print("Downloading corpus documents...\n")

    for filename, url in DOCS.items():
        path = os.path.join(DEST, filename)
        print(f"Downloading {filename}...")
        download_file(url, path)

    print("\nDownload complete.")
    print(f"Files saved to: {DEST}")


if __name__ == "__main__":
    main()