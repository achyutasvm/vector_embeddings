"""
STEP 2 of the evaluation: use OTHER models as judges ("LLM-as-a-judge").

Reads eval/output/responses.json (from generate_answers.py) and asks each
judge model: "given the question and the reference chunks, is this answer
correct or incorrect?"

Judges used:
  - Llama 3.3 70B via Groq   (needs GROQ_API_KEY, free tier at console.groq.com)
  - Gemini Flash via Google  (needs GOOGLE_API_KEY, from aistudio.google.com)

Outputs (in eval/output/):
  evaluation.json -> every verdict, plus the raw judge output
  evaluation.csv  -> one row per question, one column per judge

Run from the repo root:
  python eval/evaluate_answers.py
"""

import csv
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()

# The evaluation prompt from your assignment (with straight quotes).
EVAL_PROMPT = """You are given a question, an answer and reference text. You must determine whether the given answer correctly answers the question based on the reference text. Here is the data:
[BEGIN DATA]
************
[Question]: {question}
************
[Reference]: {context}
************
[Answer]: {sampled_answer}
[END DATA]
Your response must be a single word, either "correct" or "incorrect", and should not contain any text or characters aside from that word. "correct" means that the question is correctly and fully answered by the answer. "incorrect" means that the question is not correctly or only partially answered by the answer"""

# Model names change over time: check the provider's current model list
# if one of these returns a "model not found" error.
JUDGES = {
    # llama-3.3-70b-versatile was retired from Groq; gpt-oss-120b is the
    # closest capability class still hosted there as of 2026-09.
    "gpt-oss-120b": ChatGroq(model="openai/gpt-oss-120b", temperature=0),
    # gemini-2.5-flash is retired for new users; Google's own 404 names
    # gemini-3.6-flash as the replacement.
    "gemini-flash": ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0),
}


def extract_text(content) -> str:
    """judge.invoke(...).content is a plain string for most models, but
    gemini-3.6-flash returns a list of content blocks (each a dict with a
    "type"/"text" pair, plus an opaque "extras" signature) instead. Pull out
    just the text either way."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def normalize(raw: str) -> str:
    """Models sometimes add a period, quotes or capitals. Map them to one label.
    Check 'incorrect' FIRST, because the word 'incorrect' contains 'correct'."""
    text = raw.strip().lower().strip(".\"'` \n")
    if text.startswith("incorrect"):
        return "incorrect"
    if text.startswith("correct"):
        return "correct"
    return "invalid"  # judge didn't follow the one-word rule


def main():
    out_dir = Path("eval/output")
    records = json.loads((out_dir / "responses.json").read_text(encoding="utf-8"))

    for rec in records:
        prompt = EVAL_PROMPT.format(
            question=rec["question"],
            context=rec["context"],
            sampled_answer=rec["answer"],
        )
        rec["verdicts"] = {}
        rec["raw_judge_output"] = {}
        for name, judge in JUDGES.items():
            raw = extract_text(judge.invoke(prompt).content)
            rec["raw_judge_output"][name] = raw
            rec["verdicts"][name] = normalize(raw)
        print(f"Q{rec['id']}: {rec['verdicts']}")

    # Save full results
    with open(out_dir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # Save a simple table
    judge_names = list(JUDGES)
    with open(out_dir / "evaluation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "answer", *judge_names, "judges_agree"])
        for rec in records:
            v = [rec["verdicts"][n] for n in judge_names]
            writer.writerow(
                [rec["id"], rec["question"], rec["answer"], *v, len(set(v)) == 1]
            )

    # Summary
    print("\n===== SUMMARY =====")
    total = len(records)
    for name in judge_names:
        n_correct = sum(r["verdicts"][name] == "correct" for r in records)
        print(f"{name}: {n_correct}/{total} correct ({n_correct / total:.0%})")
    agree = sum(len(set(r["verdicts"].values())) == 1 for r in records)
    print(f"Judges agreed on {agree}/{total} questions")


if __name__ == "__main__":
    main()
