# Birch Aquarium LLM

A fully local chat assistant with retrieval over scraped content from
[Birch Aquarium at Scripps](https://aquarium.ucsd.edu) (the public education
center of Scripps Institution of Oceanography at UC San Diego). No API keys
required — embeddings run via sentence-transformers and the chat LLM runs
via [Ollama](https://ollama.com).

> Note: This is RAG (retrieval-augmented generation), not fine-tuning. The LLM
> reads scraped Birch content at query time via a local embedding index. For
> factual Q&A about a specific institution, this generally beats fine-tuning.

## Architecture

```
aquarium.ucsd.edu  ──▶  scraper.py  ──▶  data/raw/*.md
                                              │
                                              ▼
                                         ingest.py  (chunk + local embeddings)
                                              │
                                              ▼
                                         data/index.npz
                                              │
                                              ▼
                            chat.py  ──▶  Ollama / llama3.2:3b (RAG)
```

## Setup

```bash
# 1. Install Ollama and pull the model
brew install ollama
brew services start ollama
ollama pull llama3.2:3b     # the default (see docs/EVAL.md); `ollama pull llama3.1:8b` if you want to compare

# 2. Python deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No API keys needed. Embeddings run locally via [sentence-transformers](https://www.sbert.net/)
(`BAAI/bge-small-en-v1.5`, ~130MB, downloaded on first ingest). Chat runs locally
via Ollama (`llama3.2:3b`, ~2GB).

## Quick demo (browser)

A prebuilt index (`data/index.npz`, 1,422 chunks) ships in the repo, so you do **not**
need to scrape or ingest to try it. After the Setup above:

```bash
python scripts/run.py serve            # starts on http://localhost:8077
```

Open **http://localhost:8077** — a chat page with a **Subject ID + Age** bar, an
animal picker, a question box, and example questions. Subject ID is required before
asking; both are remembered across reloads. From a tablet/phone on the same Wi-Fi,
first start with `serve --host 0.0.0.0` and open `http://<your-computer>.local:8077`.

That's the whole demo: clone → `pip install` → `ollama pull llama3.2:3b` → `serve`.
The first question loads the embedding model (~130MB) and, if the model isn't warm,
the LLM; run with `OLLAMA_KEEP_ALIVE=-1 ollama serve` to keep it resident between
sessions. To rebuild the index from scratch instead of using the shipped one, see
Usage below.

## Usage

```bash
# 1. Scrape the site (writes data/raw/*.md). Default: 100 pages, polite 1s delay.
python scripts/run.py scrape

# 2. Fetch general sea-life reference content (Wikipedia) alongside the Birch pages
python scripts/run.py reference

# 3. Chunk + embed (writes data/index.npz)
python scripts/run.py ingest

# 4. Chat
python scripts/run.py chat

# 5. HTTP endpoint for the browser experiment (see below)
python scripts/run.py serve --port 8077
```

## Serving the experiment

`scripts/run.py serve` exposes the pipeline over HTTP for the `birch-ask`
study (`vislearnlab/birch-ask`):

    GET  /            -> browser demo chat page (also /demo)
    GET  /health      -> {ok, model, index_chunks, ollama}
    POST /ask         -> {answer, sources, blocked, latency_ms, ...}
                         body: {question, animal?, top_k?}
    POST /transcribe  -> {text, ...}  body: {audio: base64}

## Logging & data capture

Every `/ask` — normal or safety-blocked — is logged with subject ID, age, question,
animal, answer (shaped + raw), retrieved sources with scores, safety verdict, model,
latencies, and timestamps. Two sinks:

1. **`data/sessions.jsonl`** (always, source of truth) — one JSON object per line,
   append-only, written synchronously. Gitignored. Never fails a session, and
   survives a Mongo outage — you can backfill Mongo from it.
2. **MongoDB `birch_ask.llm_demo`** (best-effort mirror) — enabled by dropping a
   `mongo_auth.json` (`{"url": "mongodb://…"}`) in the repo root, or setting
   `MONGO_URL`. Override target with `MONGO_DB` / `MONGO_COLLECTION`. Without
   credentials it degrades to JSONL-only with a startup notice; the Mongo write runs
   on a background thread and never blocks the answer.

`mongo_auth.json` is a **secret and gitignored** — it is never committed. `pymongo` is
in `requirements.txt`; if absent, logging falls back to JSONL. See `src/datalog.py`.

Timestamps per record: `ts` (ISO-8601 UTC), `ts_epoch_ms`, `received_at_epoch_ms`,
plus server-measured `retrieve_ms`, `latency_ms`, and `server_ms`.

## Serving

Stdlib only, no web framework. The experiment's Node server proxies to it, so
the model host is configurable (`BIRCH_LLM_URL`) and never public.

Answers are tuned for the study: 2-4 sentences, no follow-up questions (a
trailing question is also stripped defensively), and no references to any
particular aquarium — children in the yoked condition must hear the same kind
of content as children who asked.

## Why the reference corpus matters

The Birch scrape covers the animals Birch exhibits and little else. Measured on
16 realistic child questions, a Birch-only index got ~5 wrong, and the errors
clustered on species the site barely mentions: retrieval fell back to whatever
page was nearest (mangroves, newsroom archives) and the 8B model filled the gap.
It told a child octopuses are "safe to touch", answered an African-penguin
question about little blue penguins, and inverted seahorse reproduction.

`python scripts/run.py reference` adds Wikipedia articles for the 18 study
species plus 20 general topics (657 -> 1422 chunks) and fixes all of them.

> Re-run `ingest` after `reference`, and **restart any running `serve` process** —
> the index is cached per process.

## Safety

`src/safety.py` gates questions before the model and answers before the
experimenter, per the study protocol's commitment to "a list of off-limits
topics and words" defaulting to *"Hang on, let me check with the experimenter."*

The design priority is **not** blocking legitimate biology: children are looking
at seahorses, sharks and octopuses, so "how do babies get born?", "is it
poisonous?" and "do sharks bite people?" must get real answers. Patterns are
narrow rather than keyword-broad — the violence rule requires a human object, so
"does the octopus kill the crab" passes.

    python tests/test_safety.py     # 22 must-block vs 30 must-pass cases

## Choosing a model

`tests/eval_models.py` factually compares candidate models over the 18 stimulus
species, so a faster/smaller model can be vetted before it ships to children. It
samples each case repeatedly (production runs at `config.TEMPERATURE`, currently
0.3) and grades by
pass *rate*. See **[docs/EVAL.md](docs/EVAL.md)** for methodology and the current
`llama3.1:8b` vs `llama3.2:3b` results.

    python tests/eval_models.py --reps=5 llama3.1:8b llama3.2:3b

## Layout

- `src/scraper.py` — polite crawler for birchaquarium.org, respects robots.txt
- `src/ingest.py` — chunks markdown, embeds locally with sentence-transformers, saves a numpy index
- `src/chat.py` — retrieval + Ollama streaming chat
- `src/reference.py` — fetches Wikipedia reference articles for the study species
- `src/serve.py` — HTTP endpoint (`/health`, `/ask`, `/transcribe`) for the experiment
- `src/safety.py` — off-limits-topic gates on question and answer
- `src/config.py` — model IDs, paths, tunables
- `tests/eval_models.py` — factual model comparison (see [docs/EVAL.md](docs/EVAL.md))
