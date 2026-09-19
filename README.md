# PDF RAG with Pinecone + LangChain

Retrieval-augmented generation over the PDF(s) in `docs/`.

- **Chunking**: each PDF page is loaded as its own `Document` (`PyPDFLoader`), then
  `RecursiveCharacterTextSplitter` splits within each page — so chunks respect
  page boundaries and every chunk keeps its `page` metadata.
- **Embeddings**: Pinecone's hosted inference model `llama-text-embed-v2`, via
  `langchain-pinecone`'s `PineconeEmbeddings`.
- **Vector store**: Pinecone serverless index, via `PineconeVectorStore`.
- **Generation**: OpenAI, via `langchain-openai`'s `ChatOpenAI` (default `gpt-4o-mini`).
  (Pinecone doesn't expose a standalone generation model outside its separate,
  auto-chunking Assistant product, so generation is done with an LLM here.)

## Architecture

```mermaid
flowchart TB
    PDF["docs/*.pdf"]

    subgraph ingest["Ingestion — python main.py ingest"]
        direction TB
        Loader["PyPDFLoader\n(one Document per page)"]
        Splitter["RecursiveCharacterTextSplitter\n(splits within each page)"]
        EmbedDocs["PineconeEmbeddings\nllama-text-embed-v2"]
        Loader --> Splitter --> EmbedDocs
    end

    Index[("Pinecone serverless index\n(PineconeVectorStore)")]

    subgraph query["Query — python main.py ask \"...\""]
        direction TB
        Question["Question"]
        EmbedQuery["PineconeEmbeddings\nllama-text-embed-v2"]
        Retrieve["Similarity search (top-k)"]
        Prompt["ChatPromptTemplate\n(context + question)"]
        LLM["ChatOpenAI\ngpt-4o-mini"]
        Answer["Answer + cited page numbers"]
        Question --> EmbedQuery --> Retrieve --> Prompt --> LLM --> Answer
    end

    PDF --> Loader
    EmbedDocs -- "upsert chunks" --> Index
    Index -- "top-k chunks" --> Retrieve
```

## Setup

```bash
python3 -m venv .venv   # Python 3.12 recommended (3.14 breaks a langchain-pinecone dep)
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then fill in PINECONE_API_KEY and OPENAI_API_KEY
```

## Usage

```bash
# Chunk docs/*.pdf and upsert into Pinecone
python main.py ingest

# Ask a question
python main.py ask "What are the main risk factors disclosed in this filing?"
```

## Config

All settings are environment variables (see `.env.example`): index name/cloud/region/
namespace, embedding model + dimension, generation model, chunk size/overlap, and
retrieval top-k.
