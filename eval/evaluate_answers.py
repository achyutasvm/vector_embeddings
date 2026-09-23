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
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()

MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 5  # doubles each attempt: 5s, 10s, 20s, 40s


def invoke_with_retry(judge, prompt: str):
    """Provider APIs occasionally return a transient error (Google's
    gemini-3.6-flash 503 'high demand' in particular) worth retrying with
    backoff. A 429 quota/rate-limit error is different: each retry is
    itself another request against the same capped quota, so retrying it
    just burns through what's left faster — fail fast on those instead."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return judge.invoke(prompt)
        except Exception as exc:
            message = str(exc)
            if "429" in message or "RESOURCE_EXHAUSTED" in message or "rate_limit" in message.lower():
                raise
            if attempt == MAX_RETRIES:
                raise
            wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(f"    (attempt {attempt} failed: {exc}; retrying in {wait}s)")
            time.sleep(wait)

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


def save_results(records: list[dict], out_dir: Path) -> None:
    """Write evaluation.json/csv from whatever verdicts exist so far, so a
    later failure doesn't discard already-completed judging."""
    judged = [r for r in records if "verdicts" in r]

    with open(out_dir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(judged, f, indent=2, ensure_ascii=False)

    judge_names = list(JUDGES)
    with open(out_dir / "evaluation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "answer", *judge_names, "judges_agree"])
        for rec in judged:
            v = [rec["verdicts"].get(n, "") for n in judge_names]
            writer.writerow(
                [rec["id"], rec["question"], rec["answer"], *v, len(set(v)) == 1]
            )


def main():
    out_dir = Path("eval/output")
    records = json.loads((out_dir / "responses.json").read_text(encoding="utf-8"))

    # Resume support: judge quotas (Gemini's free tier especially) are scarce
    # enough that re-querying a judge that already answered a question is
    # wasteful. Carry forward any verdicts from a previous, interrupted run.
    eval_path = out_dir / "evaluation.json"
    previous = {}
    if eval_path.exists():
        for rec in json.loads(eval_path.read_text(encoding="utf-8")):
            previous[rec["id"]] = rec

    try:
        for rec in records:
            prior = previous.get(rec["id"])
            rec["verdicts"] = dict(prior["verdicts"]) if prior else {}
            rec["raw_judge_output"] = dict(prior["raw_judge_output"]) if prior else {}

            missing = [name for name in JUDGES if name not in rec["verdicts"]]
            if not missing:
                print(f"Q{rec['id']}: {rec['verdicts']} (already judged, skipped)")
                continue

            prompt = EVAL_PROMPT.format(
                question=rec["question"],
                context=rec["context"],
                sampled_answer=rec["answer"],
            )
            for name in missing:
                raw = extract_text(invoke_with_retry(JUDGES[name], prompt).content)
                rec["raw_judge_output"][name] = raw
                rec["verdicts"][name] = normalize(raw)
            print(f"Q{rec['id']}: {rec['verdicts']}")
            save_results(records, out_dir)  # incremental, so progress isn't lost
    finally:
        save_results(records, out_dir)

    # Summary (only over questions that actually got judged)
    judged = [r for r in records if "verdicts" in r]
    print("\n===== SUMMARY =====")
    total = len(judged)
    if total < len(records):
        print(f"({total}/{len(records)} questions judged; run again to finish the rest)")
    for name in JUDGES:
        n_correct = sum(r["verdicts"].get(name) == "correct" for r in judged)
        print(f"{name}: {n_correct}/{total} correct ({n_correct / total:.0%})" if total else f"{name}: 0/0")
    agree = sum(len(set(r["verdicts"].values())) == 1 for r in judged)
    if total:
        print(f"Judges agreed on {agree}/{total} questions")


if __name__ == "__main__":
    main()
