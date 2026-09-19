import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from app import config


def load_pages(docs_dir: Path) -> list[Document]:
    """Load every PDF in docs_dir into one Document per page."""
    pdf_paths = sorted(docs_dir.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDF files found in {docs_dir}")

    pages: list[Document] = []
    for pdf_path in pdf_paths:
        pages.extend(PyPDFLoader(str(pdf_path)).load())
    return pages


def chunk_by_page(pages: list[Document]) -> list[Document]:
    """Recursively split within each page, so chunks never cross a page boundary."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)

    counts: dict[int, int] = {}
    for chunk in chunks:
        source = Path(chunk.metadata.get("source", "doc")).stem
        page = chunk.metadata.get("page", 0)
        counts[page] = counts.get(page, 0) + 1
        chunk.metadata["chunk_id"] = f"{source}-p{page}-{counts[page]}"
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


def run() -> None:
    pages = load_pages(Path(config.DOCS_DIR))
    chunks = chunk_by_page(pages)
    print(f"Loaded {len(pages)} pages -> {len(chunks)} recursive chunks.")

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
    print(
        f"Upserted {len(chunks)} chunks into index "
        f"'{config.PINECONE_INDEX_NAME}' (namespace '{config.PINECONE_NAMESPACE}')."
    )


if __name__ == "__main__":
    run()
