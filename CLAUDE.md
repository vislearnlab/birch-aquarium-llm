# CLAUDE.md — birch-aquarium-llm

Guidance for AI assistants (and new humans) working in this repo. Keep it current
when the facts below change.

## What this is

A **fully local** RAG assistant over Birch Aquarium + Wikipedia sea-life content. It
is the model backend for the **birch-ask** study (children ages 6–10 ask questions
about sea creatures; yoked controls hear the same answers). No API keys, no
per-query cost: embeddings run via sentence-transformers, generation via Ollama,
transcription via faster-whisper — all on the local machine. Local-only is a study
requirement (children's questions and audio must not leave the machine), not just a
cost choice.

The browser experiment lives in a **separate repo**, `birch-ask`
(github.com/vislearnlab/birch-ask), and reaches this backend over HTTP. This repo is
only the pipeline + HTTP endpoint.

## Pipeline

```
aquarium.ucsd.edu ─▶ scraper.py ─┐
                                 ├─▶ data/raw/*.md ─▶ ingest.py ─▶ data/index.npz
Wikipedia (18 species + topics) ─┘   (chunk + embed)   (numpy dot-product search)
                                                            │
                                       serve.py / chat.py ──┴─▶ Ollama (RAG) ─▶ answer
                                       (retrieve + safety gate + answer shaping)
```

## Commands

Everything goes through `scripts/run.py`. Use the venv: `.venv/bin/python`.

```bash
python scripts/run.py scrape          # crawl aquarium.ucsd.edu -> data/raw/*.md
python scripts/run.py reference       # add Wikipedia species/topic articles
python scripts/run.py ingest          # chunk + embed -> data/index.npz
python scripts/run.py chat            # interactive REPL (--ask "..." for one-shot, --voice for mic)
python scripts/run.py serve --host 0.0.0.0 --port 8077   # HTTP API + browser demo at /

python tests/test_safety.py           # safety gate: 22 must-block vs 30 must-pass
python tests/eval_models.py --reps=5 llama3.1:8b llama3.2:3b   # factual model comparison
```

After `reference`, re-run `ingest`, and **restart any running `serve`** — the index
is cached per process (`ingest.load_index` is `lru_cache`d).

## Key files

- `src/config.py` — model IDs, paths, tunables (`OLLAMA_MODEL`, `TOP_K`, chunk sizes)
- `src/ingest.py` — chunking + local embeddings + numpy index + `search()`
- `src/serve.py` — stdlib HTTP: `/` + `/demo` (browser chat page), `/health`, `/ask`,
  `/transcribe`. `ask()` is the real path (retrieval → prompt → safety → answer
  shaping); the eval calls it directly.
- `src/demo.html` — self-contained browser demo page (Subject ID + Age + chat),
  served same-origin by `serve.py`
- `src/datalog.py` — durable logging: append-only `data/sessions.jsonl` (source of
  truth) + best-effort MongoDB `birch_ask.llm_demo` mirror; never blocks/fails a session
- `data/index.npz` — prebuilt embedding index committed so the demo runs on a fresh
  clone; rebuild with `ingest` after changing the corpus
- `src/safety.py` — off-limits-topic gates on both question and answer, defaulting to
  "Hang on, let me check with the experimenter." Tuned NOT to block legitimate biology.
- `src/reference.py` — Wikipedia fetch for the 18 stimulus species + general topics
- `src/chat.py` — CLI REPL (streams; `serve.ask` does not)
- `tests/eval_models.py` + `docs/EVAL.md` — model comparison; read the doc before
  touching model choice

## Decisions already made (don't relitigate without reason)

- **Model: `llama3.2:3b`.** It and `llama3.1:8b` are indistinguishable on accuracy
  (16/18 clean each; the 8B's one flaky case is an ambiguous question, not a gap), and
  the 3B is ~2.5–3× faster. See `docs/EVAL.md`. `config.OLLAMA_MODEL` is set to the 3B.
  A `--reps=10` run to tighten the flaky-rate estimates is still worth doing but was
  not blocking the switch.
- **RAG, not fine-tuning** — the model reads scraped content at query time.
- **Reference corpus is load-bearing.** A Birch-only index confabulated on species the
  site barely covers (told a child octopuses are "safe to touch", inverted seahorse
  reproduction). Wikipedia articles fixed all three; the eval guards against regression.
- **Answers are shaped for the study** in `serve.py`: 2–4 sentences, trailing questions
  stripped, no mention of any specific aquarium (yoked-condition children must hear the
  same *kind* of content). Preserve these when editing `ask()`.

## Gotchas

- **Regex graders can't see negation.** In `tests/eval_models.py`, a `must_not` on
  "safe to touch" matches "NOT safe to touch". Always read the printed transcripts;
  never trust a headline number you haven't spot-checked. (`docs/EVAL.md` has the story.)
- **Cold model load — fixed.** Ollama's default 5-min idle unload meant any gap
  between questions (intro, animal browsing, choice screens) could evict the model
  and force a full reload on the next `/ask`. Both `ask()` and `warm_up()` now send
  `keep_alive: config.OLLAMA_KEEP_ALIVE` (`-1`, i.e. resident indefinitely) on every
  `/api/chat` call, and `warm_up()` also does a real Ollama round-trip at startup, not
  just the embedding index. This is per-request, so it holds even if `ollama serve`
  is launched without the env var.
- **Durable logging is on** (`src/datalog.py`): every `/ask` → `data/sessions.jsonl`
  (source of truth) + best-effort MongoDB `birch_ask.llm_demo`. Retrieval scores,
  safety verdicts, raw answers, subject ID/age, and timestamps are all captured.
  Mongo needs `mongo_auth.json` (gitignored secret) or `MONGO_URL`; without it,
  JSONL-only. JSONL survives a Mongo outage — backfill from it.
- **OpenMP collision on macOS.** `scripts/run.py` sets `KMP_DUPLICATE_LIB_OK` before any
  numpy/ctranslate2 import; don't move it below the imports.
- Embeddings are normalized, so cosine == dot product; `search()` is a plain matmul.

## Conventions

- Match the surrounding style: type hints, `from __future__ import annotations`, terse
  docstrings that explain *why*.
- Commit messages end with the `Co-Authored-By` trailer for the model used.
- This is the `experiment-api` branch; `main` is the original scaffold.
