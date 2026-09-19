import os

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


PINECONE_API_KEY = _require("PINECONE_API_KEY")
OPENAI_API_KEY = _require("OPENAI_API_KEY")

PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "rag-docs")
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1")
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "docs")

# Pinecone-hosted embedding model (integrated inference). 1024 dims matches
# both llama-text-embed-v2 and multilingual-e5-large.
EMBEDDING_MODEL = os.environ.get("PINECONE_EMBEDDING_MODEL", "llama-text-embed-v2")
EMBEDDING_DIMENSION = int(os.environ.get("PINECONE_EMBEDDING_DIMENSION", "1024"))

# OpenAI model used for the generation step of the RAG chain.
GENERATION_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

DOCS_DIR = os.environ.get("DOCS_DIR", "docs")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))
TOP_K = int(os.environ.get("RAG_TOP_K", "5"))
