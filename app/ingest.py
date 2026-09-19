import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from app import config


def load_pdf(pdf_path: Path) -> list[Document]:
    """Load a single PDF into one Document per page."""
    return PyPDFLoader(str(pdf_path)).load()


def load_pages(docs_dir: Path) -> list[Document]:
    """Load every PDF in docs_dir into one Document per page."""
    pdf_paths = sorted(docs_dir.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDF files found in {docs_dir}")

    pages: list[Document] = []
    for pdf_path in pdf_paths:
        pages.extend(load_pdf(pdf_path))
    return pages


def chunk_by_page(pages: list[Document]) -> list[Document]:
    """Recursively split within each page, so chunks never cross a page boundary."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)

    counts: dict[tuple[str, int], int] = {}
    for chunk in chunks:
        source = Path(chunk.metadata.get("source", "doc")).stem
        page = chunk.metadata.get("page", 0)
        key = (source, page)
        counts[key] = counts.get(key, 0) + 1
        chunk.metadata["chunk_id"] = f"{source}-p{page}-{counts[key]}"
    return chunks


def ensure_index(pc: Pinecone) -> None:
    existing = {idx["name"] for idx in pc.list_indexes()}
    if config.PINECONE_INDEX_NAME in existing:
        return

    pc.create_index(
        name=config.PINECONE_INDEX_NAME,
        dimension=config.EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
    )
    while not pc.describe_index(config.PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)


def ingest_documents(pages: list[Document]) -> int:
    """Chunk pages, embed, and upsert into Pinecone. Returns the chunk count."""
    chunks = chunk_by_page(pages)

    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    ensure_index(pc)

    embeddings = PineconeEmbeddings(
        model=config.EMBEDDING_MODEL,
        pinecone_api_key=config.PINECONE_API_KEY,
    )

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=config.PINECONE_INDEX_NAME,
        namespace=config.PINECONE_NAMESPACE,
        ids=[c.metadata["chunk_id"] for c in chunks],
    )
    return len(chunks)


def run() -> None:
    pages = load_pages(Path(config.DOCS_DIR))
    print(f"Loaded {len(pages)} pages from {config.DOCS_DIR}")

    chunk_count = ingest_documents(pages)
    print(
        f"Upserted {chunk_count} chunks into index "
        f"'{config.PINECONE_INDEX_NAME}' (namespace '{config.PINECONE_NAMESPACE}')."
    )


if __name__ == "__main__":
    run()
