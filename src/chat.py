"""Interactive RAG chat over the Birch Aquarium index, powered by a local Ollama model."""
from __future__ import annotations

import json

import requests

from . import config, ingest

SYSTEM_PROMPT = (
    "You are a friendly tour guide at Birch Aquarium at Scripps in La Jolla, "
    "California. You're talking to kids ages 6 to 10 who are curious about the "
    "ocean and sea creatures.\n\n"
    "How to talk:\n"
    "- Use simple words a kid would understand. Short sentences. Be warm and "
    "  excited about the ocean!\n"
    "- If you use a tricky science word (like 'invertebrate' or 'bioluminescence'), "
    "  explain it right after in plain words.\n"
    "- Keep answers short — 2 to 4 sentences usually. Don't lecture.\n"
    "- Ask a fun follow-up question sometimes to keep the conversation going.\n\n"
    "What you know:\n"
    "- For questions about Birch Aquarium specifically (exhibits, hours, animals "
    "  on display, programs, location), use the <context> below. Add a tiny link "
    "  in brackets at the end if you used a source, like [aquarium.ucsd.edu/exhibits].\n"
    "- For general ocean and sea-creature questions, you can use your own ocean "
    "  knowledge — you don't need a source for those. Just be sure it's true.\n"
    "- If you don't know, say 'Hmm, I'm not sure!' — never make stuff up."
)


def _format_context(hits: list[tuple[float, str, str]]) -> str:
    parts = []
    for i, (score, chunk, source) in enumerate(hits, 1):
        parts.append(f"[{i}] source: {source}\n{chunk}")
    return "\n\n---\n\n".join(parts)


def answer(question: str) -> str:
    hits = ingest.search(question, k=config.TOP_K)
    context = _format_context(hits)

    user_msg = f"<context>\n{context}\n</context>\n\nQuestion: {question}"
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "stream": True,
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
    }

    text_parts: list[str] = []
    with requests.post(
        f"{config.OLLAMA_HOST}/api/chat", json=payload, stream=True, timeout=300
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            chunk = event.get("message", {}).get("content", "")
            if chunk:
                print(chunk, end="", flush=True)
                text_parts.append(chunk)
            if event.get("done"):
                break
    print()
    return "".join(text_parts)


def repl(voice_mode: bool = False) -> None:
    mode_note = "voice mode" if voice_mode else "text mode"
    print(f"Birch Aquarium chat ({config.OLLAMA_MODEL}, {mode_note}).")
    print("Ctrl-C or empty input to exit.\n")

    if voice_mode:
        from . import voice  # lazy import — only load Whisper if used

    while True:
        try:
            if voice_mode:
                print("you>", end=" ", flush=True)
                q = voice.listen().strip()
                if not q:
                    print()
                    return
                print(f"\rfrom mic> {q}                    ")
            else:
                q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            return
        print("\nguide> ", end="", flush=True)
        try:
            answer(q)
        except requests.RequestException as e:
            print(f"\n[ollama error: {e}]")
        print()
