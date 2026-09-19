import argparse

from app import ingest, rag


def main() -> None:
    parser = argparse.ArgumentParser(description="Pinecone + LangChain RAG demo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "ingest",
        help="Load docs/*.pdf, chunk by page + recursively, embed, and upsert to Pinecone",
    )

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the ingested documents")
    ask_parser.add_argument("question", nargs="+")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest.run()
    elif args.command == "ask":
        question = " ".join(args.question)
        answer, docs = rag.ask(question)
        print("Answer:\n" + answer)
        print("\nSources:")
        for doc in docs:
            print(f" - page {rag.page_label(doc.metadata.get('page'))} ({doc.metadata.get('source')})")


if __name__ == "__main__":
    main()
