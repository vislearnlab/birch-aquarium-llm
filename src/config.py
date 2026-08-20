from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_PATH = DATA_DIR / "index.npz"

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"  # chosen over llama3.1:8b: equal accuracy, ~2.5-3x faster (docs/EVAL.md)
# -1 = keep the model resident indefinitely. Sent on every /api/chat request rather
# than relied on as a server-launch flag: `ollama serve` env vars are easy to forget
# before a session, and a gap between questions (intro, animal browsing, choice
# screens) routinely exceeds Ollama's 5-minute default, evicting the model and
# forcing a full reload — the multi-second-to-tens-of-seconds stall kids won't wait
# through. See CLAUDE.md gotchas.
OLLAMA_KEEP_ALIVE = -1
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # local, ~130MB, 384-dim

SEED_URL = "https://aquarium.ucsd.edu/"
ALLOWED_HOST = "aquarium.ucsd.edu"
MAX_PAGES = 200
REQUEST_DELAY_SECONDS = 1.0
# URL path substrings to crawl first (prioritized in queue order)
PRIORITY_PATHS = ("/exhibits", "/animals", "/animals-exhibits", "/learn")
USER_AGENT = "birch-aquarium-llm/0.1 (+https://github.com/; contact: brialorelle@gmail.com)"

CHUNK_CHARS = 1500
CHUNK_OVERLAP = 200
# Was 6. Prompt prefill (system prompt + all TOP_K chunks) dominates per-question
# latency on the study laptop — measured ~9-11s of prefill alone at TOP_K=6, on top
# of generation, for context this hardware doesn't batch efficiently. Dropped to 4
# to cut prompt size; re-run tests/eval_models.py after any further change here, since
# the reference corpus and retrieval depth are what fixed the confabulation
# regressions documented in CLAUDE.md — fewer chunks is a real accuracy trade-off,
# not a free win.
TOP_K = 4

# Was 0.7. Manual A/B testing (5 well-grounded questions, temps 0.0-1.0) found
# well-grounded facts stay correct across the whole range, but two things get
# worse as temperature rises: an ungrounded question (no matching corpus
# content) goes from consistently admitting "we don't know" at 0.0 to
# confidently fabricating a different number ~75% of the time at 0.7, and
# even a normally-solid fact (seahorse egg-carrying) started to garble at
# 1.0. 0.3 keeps a little phrasing variety across sessions without reaching
# either failure mode.
TEMPERATURE = 0.3

# Voice (mic input + Whisper transcription)
WHISPER_MODEL = "base.en"  # ~150MB, English-only, fast on CPU
AUDIO_SAMPLE_RATE = 16000
