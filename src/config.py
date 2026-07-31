from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_PATH = DATA_DIR / "index.npz"

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"  # chosen over llama3.1:8b: equal accuracy, ~2.5-3x faster (docs/EVAL.md)
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
TOP_K = 6

# Voice (mic input + Whisper transcription)
WHISPER_MODEL = "base.en"  # ~150MB, English-only, fast on CPU
AUDIO_SAMPLE_RATE = 16000
