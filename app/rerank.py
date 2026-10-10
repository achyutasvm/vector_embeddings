import logging

from langchain_core.documents import Document
from pinecone import Pinecone

from app import config

logger = logging.getLogger(__name__)

_client: Pinecone | None = None


def _get_client() -> Pinecone:
    global _client
    if _client is None:
        _client = Pinecone(api_key=config.PINECONE_API_KEY)
    return _client


def rerank(question: str, docs: list[Document], top_n: int) -> list[Document]:
    """Reorder docs by relevance to the question and keep the best top_n.

    Each returned doc gets its reranker score in metadata["rerank_score"]. If
    the rerank call fails (quota, network, etc.), falls back to the vector
    store's original order so the question still gets answered.
    """
    if not docs:
        return docs
    try:
        result = _get_client().inference.rerank(
            model=config.RERANK_MODEL,
            query=question,
            documents=[{"id": str(i), "text": d.page_content} for i, d in enumerate(docs)],
            rank_fields=["text"],
            top_n=min(top_n, len(docs)),
            return_documents=False,
        )
    except Exception:
        logger.exception("Rerank failed; falling back to vector search order")
        return docs[:top_n]

    reranked = []
    for row in result.data:
        doc = docs[row.index]
        doc.metadata["rerank_score"] = float(row.score)
        reranked.append(doc)
    return reranked
