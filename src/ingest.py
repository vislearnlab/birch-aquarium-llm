"""Chunk scraped markdown and embed locally with sentence-transformers."""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from . import config

_MODEL: SentenceTransformer | None = None


def _model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        print(f"loading embedding model: {config.EMBED_MODEL} (first run downloads ~130MB)")
        _MODEL = SentenceTransformer(config.EMBED_MODEL)
    return _MODEL


def chunk_text(text: str, size: int = config.CHUNK_CHARS, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Paragraph-aware sliding window. Greedy pack paragraphs up to `size`."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 <= size:
            buf = f"{buf}\n\n{p}".strip()
        else:
            if buf:
                chunks.append(buf)
            if len(p) <= size:
                buf = p
            else:
                # paragraph itself is too long — hard split
                for i in range(0, len(p), size - overlap):
                    chunks.append(p[i : i + size])
                buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def _read_source(path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^source:\s*(\S+)", text, flags=re.MULTILINE)
    return (m.group(1) if m else path.name), text


def build_index() -> None:
    files = sorted(config.RAW_DIR.glob("*.md"))
    if not files:
        raise SystemExit(f"no scraped files in {config.RAW_DIR}. run `scrape` first.")

    chunks: list[str] = []
    sources: list[str] = []
    for path in files:
        source, text = _read_source(path)
        for c in chunk_text(text):
            chunks.append(c)
            sources.append(source)
    print(f"chunked {len(files)} files -> {len(chunks)} chunks")

    arr = _model().encode(
        chunks,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # so cosine == dot product
        convert_to_numpy=True,
    ).astype(np.float32)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.INDEX_PATH,
        embeddings=arr,
        chunks=np.array(chunks, dtype=object),
        sources=np.array(sources, dtype=object),
    )
    load_index.cache_clear()  # the on-disk index just changed
    print(f"wrote {config.INDEX_PATH}")


@lru_cache(maxsize=1)
def load_index() -> tuple[np.ndarray, list[str], list[str]]:
    """Load the embedding index. Cached — the server queries this on every request."""
    if not config.INDEX_PATH.exists():
        raise SystemExit(f"no index at {config.INDEX_PATH}. run `ingest` first.")
    z = np.load(config.INDEX_PATH, allow_pickle=True)
    return z["embeddings"], list(z["chunks"]), list(z["sources"])


def search(query: str, k: int = config.TOP_K) -> list[tuple[float, str, str]]:
    """Return top-k (score, chunk, source) tuples for the query."""
    embeddings, chunks, sources = load_index()
    qv = _model().encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)[0]
    scores = embeddings @ qv
    top = np.argsort(-scores)[:k]
    return [(float(scores[i]), chunks[i], sources[i]) for i in top]
