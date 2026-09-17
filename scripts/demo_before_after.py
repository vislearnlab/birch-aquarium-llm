"""Before/after demo: fully generic (no-RAG) LLM answer vs. the real birch-ask
RAG pipeline (retrieval + kid-safety system prompt + safety gate + institution
scrubbing + numeric-grounding check), for real questions asked by real kids in
pilot testing (data/pilot/*_backup.jsonl).

Usage:
    .venv/bin/python scripts/demo_before_after.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from src import config, serve

QUESTIONS = [
    {"question": "How fast can they slide on their bellies? Do they slide?", "animal": "gentoo penguin", "age": 9, "participant": "museumAI06"},
    {"question": "why is it called loggerhead?", "animal": "loggerhead sea turtle", "age": 9, "participant": "museum010"},
    {"question": "Can their babies stay alive under the snow?", "animal": "African penguin", "age": 7, "participant": "museumAI09"},
]


def generic_answer(question: str) -> str:
    """Fully generic call: no retrieved context, no kid-safety system prompt,
    no post-processing -- just the raw model answering the bare question."""
    resp = requests.post(
        f"{config.OLLAMA_HOST}/api/chat",
        json={
            "model": config.OLLAMA_MODEL,
            "messages": [{"role": "user", "content": question}],
            "stream": False,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "options": {"temperature": config.TEMPERATURE},
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "").strip()


def main():
    results = []
    for item in QUESTIONS:
        print(f"\n=== {item['question']!r} (age {item['age']}, re: {item['animal']}) ===")
        print("running generic (no-RAG)...")
        before = generic_answer(item["question"])
        print("running full RAG pipeline...")
        after = serve.ask(item["question"], animal=item["animal"])
        results.append({**item, "before": before, "after": after})
        print(f"BEFORE: {before}")
        print(f"AFTER:  {after['answer']}")

    Path("/tmp/before_after_demo.json").write_text(json.dumps(results, indent=2))
    print("\nsaved -> /tmp/before_after_demo.json")


if __name__ == "__main__":
    main()
