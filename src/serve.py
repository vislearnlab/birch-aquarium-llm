"""HTTP server exposing the Birch RAG pipeline to the browser experiment.

Stdlib only — no web framework, matching the repo's "no extra deps" setup.

Endpoints
    GET  /health          -> {ok, model, index_chunks, ollama}
    POST /ask             -> {answer, sources, latency_ms, ...}

The experiment's Node server proxies to this; see birch-ask/README.md.

Run:  python scripts/run.py serve  [--port 8077]
"""
from __future__ import annotations

import base64
import io
import json
import random
import re
import time
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import requests

from . import config, datalog, ingest, safety


@lru_cache(maxsize=1)
def _demo_page() -> str:
    """Self-contained browser chat page, served same-origin at GET / and /demo."""
    return (Path(__file__).parent / "demo.html").read_text(encoding="utf-8")

# Kid-facing prompt. Deliberately diverges from chat.SYSTEM_PROMPT in two ways:
#   1. NO follow-up questions — in the yoked-facts condition the experimenter
#      reads these aloud, and a trailing question would hand the control group
#      the very conversational pull the manipulation is supposed to withhold.
#   2. No bracketed source links — the experimenter reads this text verbatim.
SYSTEM_PROMPT = (
    "You are a friendly ocean expert answering a question from a child aged 7 to 10 "
    "who is looking at a photo of a sea creature.\n\n"
    "How to answer:\n"
    "- Use simple words a child understands. Short sentences.\n"
    "- Be warm and excited about the ocean.\n"
    "- If you use a tricky science word, explain it right after in plain words.\n"
    "- Answer in 2 to 4 sentences. Do not lecture.\n"
    "- Your answer will be READ ALOUD by an adult exactly as written. So write "
    "only the answer itself — no headings, no bullet points, no stage "
    "directions, no bracketed links, no emoji.\n"
    "- Do NOT ask the child any questions. Do not end with a question. "
    "Just give the answer and stop.\n\n"
    "VERY IMPORTANT — talk about the ANIMAL, never about any particular aquarium:\n"
    "- The reference material below comes partly from an aquarium website. Use the "
    "FACTS in it, but never mention the aquarium, its name, its staff, its exhibits, "
    "its tanks, or its website.\n"
    "- Never say 'we', 'us', 'our', 'here', 'our aquarists', 'in our tank', or "
    "'come visit'. You are not speaking on behalf of any institution.\n"
    "- Never claim this animal lives at, or is on display at, any particular place.\n"
    "- Write about the animal as it is in the world: what it does, eats, looks like, "
    "where it lives in the wild.\n\n"
    "What you know:\n"
    "- Use the <context> below for facts about this animal.\n"
    "- For general ocean and sea-creature questions you may also use your own ocean "
    "knowledge. Be sure it is true.\n"
    "- If you do not know, say 'Hmm, I'm not sure!' — never make anything up."
)

# Shown to the experimenter, appended to the answer, when the numeric-grounding
# check below fires. Three variants so it doesn't read as the same rote line
# repeated across a session. Authored text, not model output, so it's exempt
# from (and doesn't need) the institution-scrubbing applied to generated
# answers — the whole point here is to name Birch Aquarium as a real resource.
UNCERTAIN_CAVEATS = [
    "Zorpie isn't quite sure about the answer to that one! That sounds like a "
    "perfect question for someone who works at Birch Aquarium — you could ask "
    "them today, or the next time you visit.",
    "Hmm, that's a tricky one — Zorpie doesn't know for sure! The folks at "
    "Birch Aquarium would probably love that question. Maybe ask one of them "
    "today, or next time you're here!",
    "Zorpie's still learning about that one! That would be a great question "
    "for an expert at Birch Aquarium — ask if you see one today, or on your "
    "next visit.",
]

# Belt-and-braces: strip a trailing question even if the model ignores the prompt.
_TRAILING_Q = re.compile(r"(?:(?<=[.!])|^)\s*[^.!?]*\?\s*$")
# Pulls specific numbers out of text for the numeric-grounding check in ask().
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def strip_follow_up(text: str) -> str:
    """Remove a trailing follow-up question. Keeps text that is entirely one question."""
    out = _TRAILING_Q.sub("", text.strip()).strip()
    return out if out else text.strip()


def build_context(hits):
    return "\n\n---\n\n".join(
        f"[{i}] source: {src}\n{chunk}" for i, (score, chunk, src) in enumerate(hits, 1)
    )


def numbers_in(text: str) -> set[str]:
    return set(_NUM_RE.findall(text))


def ask(
    question: str,
    animal: str | None = None,
    top_k: int | None = None,
    previous_question: str | None = None,
    previous_answer: str | None = None,
) -> dict:
    """Retrieve context and generate one kid-facing answer. Returns a JSON-able dict.

    Each call is otherwise stateless — the caller (birch-ask) hands back the
    immediately preceding question/answer for the current animal, if any, so
    a follow-up like "what kind of foods?" has something to anchor to instead
    of reading as a fresh, unrelated question. Folded into both the retrieval
    query (the previous question often carries the actual topic; the current
    one alone may not) and the chat history (as a real prior turn, not text
    glued into the current one).
    """
    t0 = time.time()
    query = " ".join(p for p in (animal, previous_question, question) if p)
    hits = ingest.search(query, k=top_k or config.TOP_K)
    t_retrieve = time.time()

    context_text = build_context(hits)
    user_msg = f"<context>\n{context_text}\n</context>\n\nQuestion: {question}"
    if animal:
        user_msg = f"The child is looking at a photo of: {animal}.\n\n" + user_msg

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if previous_question and previous_answer:
        messages.append({"role": "user", "content": previous_question})
        messages.append({"role": "assistant", "content": previous_answer})
    messages.append({"role": "user", "content": user_msg})

    resp = requests.post(
        f"{config.OLLAMA_HOST}/api/chat",
        json={
            "model": config.OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "options": {"temperature": config.TEMPERATURE},
        },
        timeout=180,
    )
    resp.raise_for_status()
    raw = resp.json().get("message", {}).get("content", "").strip()
    answer = strip_follow_up(raw)
    answer, institution_stripped = safety.strip_institution(answer)
    follow_up_stripped = raw != strip_follow_up(raw) or raw != answer

    # Output gate: even a benign question can produce something a child shouldn't hear.
    out_verdict = safety.check_answer(answer)
    if not out_verdict.ok:
        return {
            "answer": safety.REFUSAL,
            "answer_raw": raw,
            "blocked": True,
            "blocked_stage": "answer",
            "blocked_category": out_verdict.category,
            "sources": [], "model": config.OLLAMA_MODEL, "animal": animal,
            "question": question, "retrieve_ms": int((t_retrieve - t0) * 1000),
            "latency_ms": int((time.time() - t0) * 1000),
        }

    # Numeric-grounding check: if the answer states a specific number that
    # doesn't appear anywhere in the retrieved context, the model likely
    # filled a gap with its own general knowledge rather than reading it off
    # a real source. Manual A/B testing (see project notes) found this catches
    # confidently-fabricated numbers with zero false positives on well-
    # grounded answers, though it can't catch non-numeric fabrications — a
    # known, accepted gap for now. Appends a kid-facing redirect to Birch
    # Aquarium staff rather than letting a made-up number stand unqualified.
    unmatched_numbers = sorted(numbers_in(answer) - numbers_in(context_text))
    uncertain = bool(unmatched_numbers)
    if uncertain:
        answer = f"{answer.rstrip()} {random.choice(UNCERTAIN_CAVEATS)}"

    return {
        "answer": answer,
        "answer_raw": raw,
        "blocked": False,
        "uncertain": uncertain,
        "unmatched_numbers": unmatched_numbers,
        "follow_up_stripped": follow_up_stripped,
        "institution_stripped": institution_stripped,
        "sources": [
            {"score": round(s, 4), "source": src, "preview": chunk[:200]}
            for s, chunk, src in hits
        ],
        "model": config.OLLAMA_MODEL,
        "animal": animal,
        "question": question,
        "retrieve_ms": int((t_retrieve - t0) * 1000),
        "latency_ms": int((time.time() - t0) * 1000),
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter default logging
        print(f"[serve] {fmt % args}")

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, code: int = 200):
        body = html.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/demo"):
            self._send_html(_demo_page())
            return
        if path != "/health":
            self._send(404, {"error": "not found"})
            return
        try:
            _, chunks, _ = ingest.load_index()
            n = len(chunks)
        except SystemExit as e:
            self._send(503, {"ok": False, "error": str(e)})
            return
        try:
            r = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=5)
            ollama = "up" if r.ok else f"http {r.status_code}"
        except requests.RequestException as e:
            ollama = f"down ({e.__class__.__name__})"
        self._send(200, {
            "ok": ollama == "up",
            "model": config.OLLAMA_MODEL,
            "index_chunks": n,
            "ollama": ollama,
        })

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/ask", "/transcribe"):
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError) as e:
            self._send(400, {"error": f"bad JSON: {e}"})
            return

        if path == "/transcribe":
            b64 = body.get("audio") or ""
            if not b64:
                self._send(400, {"error": "missing 'audio' (base64)"})
                return
            try:
                result = transcribe_audio(base64.b64decode(b64))
            except Exception as e:
                print(f"[serve] transcribe failed: {e}")
                self._send(500, {"error": f"{e.__class__.__name__}: {e}"})
                return
            print(f"[serve] transcribed {result['bytes']}B -> {result['text']!r}")
            self._send(200, result)
            return

        t_recv = time.time()
        question = (body.get("question") or "").strip()
        if not question:
            self._send(400, {"error": "missing 'question'"})
            return

        # Input gate. Blocked questions never reach the model; the experimenter gets
        # the protocol's own remedy ("let me check with the experimenter") and the
        # attempt is still logged and returned.
        verdict = safety.check_question(question)
        if not verdict.ok:
            print(f"[serve] BLOCKED ({verdict.category}): {question!r}")
            payload = {
                "answer": safety.REFUSAL,
                "blocked": True,
                "blocked_stage": "question",
                "blocked_category": verdict.category,
                "blocked_match": verdict.matched,
                "sources": [], "model": config.OLLAMA_MODEL,
                "animal": body.get("animal"), "question": question,
                "retrieve_ms": 0, "latency_ms": 0,
            }
            self._log_and_send(body, payload, t_recv)
            return

        try:
            result = ask(
                question,
                animal=body.get("animal"),
                top_k=body.get("top_k"),
                previous_question=body.get("previousQuestion"),
                previous_answer=body.get("previousAnswer"),
            )
        except requests.RequestException as e:
            self._send(502, {"error": f"ollama unreachable: {e}"})
            return
        except Exception as e:  # keep the session alive; the UI shows a retry
            self._send(500, {"error": f"{e.__class__.__name__}: {e}"})
            return

        self._log_and_send(body, result, t_recv)

    # Fields the client may send for record-keeping; echoed back and logged.
    _BOOKKEEPING = ("subjectId", "age", "experimentId", "sessionId",
                    "participantID", "trialNum", "questionIndex")

    def _log_and_send(self, body: dict, payload: dict, t_recv: float):
        """Attach subject/bookkeeping fields, persist a full record, then respond.

        Logging is durable (JSONL) + best-effort Mongo via datalog, and never blocks
        or fails the response — see src/datalog.py.
        """
        subject = (str(body.get("subjectId") or body.get("subject_id") or "").strip()
                   or None)
        # Merge bookkeeping into the response so the client sees what was recorded.
        payload["subjectId"] = subject
        payload["age"] = body.get("age")
        for k in ("experimentId", "sessionId", "participantID", "trialNum", "questionIndex"):
            if k in body:
                payload[k] = body[k]

        record = {
            "subjectId": subject,
            "age": body.get("age"),
            "question": payload.get("question"),
            "animal": payload.get("animal"),
            "answer": payload.get("answer"),
            "answer_raw": payload.get("answer_raw"),
            "blocked": payload.get("blocked", False),
            "blocked_stage": payload.get("blocked_stage"),
            "blocked_category": payload.get("blocked_category"),
            "sources": payload.get("sources"),
            "model": payload.get("model"),
            "retrieve_ms": payload.get("retrieve_ms"),
            "latency_ms": payload.get("latency_ms"),
            "received_at_epoch_ms": int(t_recv * 1000),
            "server_ms": int((time.time() - t_recv) * 1000),
            "client_ip": self.client_address[0] if self.client_address else None,
            "user_agent": self.headers.get("User-Agent"),
        }
        for k in ("experimentId", "sessionId", "participantID", "trialNum", "questionIndex"):
            if k in body:
                record[k] = body[k]
        datalog.log_interaction(record)  # datalog adds ts / ts_epoch_ms
        self._send(200, payload)


def transcribe_audio(raw: bytes) -> dict:
    """Transcribe browser-recorded audio locally with Whisper.

    Deliberately NOT the phone keyboard's dictation button: that ships the audio to
    Apple/Google. This study runs its model locally, and routing children's questions
    through a third-party speech service would undo that for no benefit — especially
    since faster-whisper is already a dependency here.

    faster-whisper accepts a file-like object and decodes via PyAV, so the browser's
    WebM/Opus blob goes straight in with no ffmpeg step.
    """
    from . import voice  # lazy — don't load Whisper unless transcription is used

    t0 = time.time()
    text = " ".join(
        s.text.strip()
        for s in voice._model().transcribe(io.BytesIO(raw), language="en",
                                           vad_filter=True)[0]
    ).strip()
    return {
        "text": text,
        "bytes": len(raw),
        "model": config.WHISPER_MODEL,
        "latency_ms": int((time.time() - t0) * 1000),
    }


def warm_up():
    """Load the embedding model + index, and pull the Ollama model into memory,
    once at process start so the first child's question doesn't pay for either.
    """
    try:
        t0 = time.time()
        ingest.search("sea creature", k=1)
        print(f"[serve] embedding warm-up complete in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"[serve] embedding warm-up failed: {e}")
    try:
        t0 = time.time()
        requests.post(
            f"{config.OLLAMA_HOST}/api/chat",
            json={
                "model": config.OLLAMA_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "keep_alive": config.OLLAMA_KEEP_ALIVE,
            },
            timeout=180,
        )
        print(f"[serve] ollama warm-up complete in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"[serve] ollama warm-up failed: {e}")


def run(port: int = 8077, host: str = "127.0.0.1") -> None:
    threading.Thread(target=warm_up, daemon=True).start()
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"[serve] birch LLM on http://{host}:{port}  (model={config.OLLAMA_MODEL})")
    print("[serve]   GET  /health")
    print("[serve]   POST /ask   {question, animal?}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] shutting down")
        srv.shutdown()
