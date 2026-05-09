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

# 2. Chunk + embed (writes data/index.npz)
python scripts/run.py ingest

# 3. Chat
python scripts/run.py chat
```

## Layout

- `src/scraper.py` — polite crawler for birchaquarium.org, respects robots.txt
- `src/ingest.py` — chunks markdown, embeds locally with sentence-transformers, saves a numpy index
- `src/chat.py` — retrieval + Ollama streaming chat
- `src/config.py` — model IDs, paths, tunables
