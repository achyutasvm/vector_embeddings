"""
STEP 1 of the evaluation: generate answers with the ORIGINAL RAG pipeline (gpt-4o-mini).

Reuses app/rag.py directly, so the questions go through exactly the same
retriever, prompt and model as your real app.

Outputs (in eval/output/):
  responses.txt   -> human-readable "prompt -> answer" log (what the assignment asks for)
  responses.json  -> same data in a structured form, used by evaluate_answers.py

Run from the repo root:
  python eval/generate_answers.py
"""

import json
import sys
from pathlib import Path

# Let Python find the "app" package when this file is run as eval/generate_answers.py
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv()

from app.rag import PROMPT, ask, format_docs  # noqa: E402  (import after sys.path fix)

# ---------------------------------------------------------------------------
# Your test questions. Keep one the PDF does NOT answer (the protein one),
# to check that the model says "I don't know" instead of making something up.
# ---------------------------------------------------------------------------
QUESTIONS = [
    "Where is Apple Headquarters?",
    "List all Apple Products",
    "Share Apple revenue by products and services",
    "Apple revenue for iphones in 2025",
    "What is the main source of protein rich food?",
    "How many employees are at Apple?",
]


def main():
    out_dir = REPO_ROOT / "eval" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with open(out_dir / "responses.txt", "w", encoding="utf-8") as log:
        for i, question in enumerate(QUESTIONS, start=1):
            # Same function the CLI and the API use: retrieve + generate
            answer, docs = ask(question)

            # The exact context string gpt-4o-mini saw (saved for the judges)
            context = format_docs(docs)

            # The full prompt as text, only for the log file
            prompt_text = PROMPT.format(context=context, question=question)

            records.append(
                {"id": i, "question": question, "context": context, "answer": answer}
            )

            log.write(f"===== Q{i} =====\n")
            log.write(f"PROMPT:\n{prompt_text}\n\n")
            log.write(f"ANSWER:\n{answer}\n\n")
            print(f"Q{i}: {question}\n -> {answer}\n")

    with open(out_dir / "responses.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(records)} responses to {out_dir}/")


if __name__ == "__main__":
    main()
