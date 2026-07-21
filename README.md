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
                            chat.py  ──▶  Ollama / llama3.1:8b (RAG)
```

## Setup

```bash
# 1. Install Ollama and pull the model
brew install ollama
brew services start ollama
ollama pull llama3.1:8b

# 2. Python deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No API keys needed. Embeddings run locally via [sentence-transformers](https://www.sbert.net/)
(`BAAI/bge-small-en-v1.5`, ~130MB, downloaded on first ingest). Chat runs locally
via Ollama (`llama3.1:8b`, ~5GB).

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
study in `vislearnlab/drawing_experiments`:

    GET  /health   -> {ok, model, index_chunks, ollama}
    POST /ask      -> {answer, sources, blocked, latency_ms, ...}
                      body: {question, animal?, top_k?}

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

## Layout

- `src/scraper.py` — polite crawler for birchaquarium.org, respects robots.txt
- `src/ingest.py` — chunks markdown, embeds locally with sentence-transformers, saves a numpy index
- `src/chat.py` — retrieval + Ollama streaming chat
- `src/config.py` — model IDs, paths, tunables
