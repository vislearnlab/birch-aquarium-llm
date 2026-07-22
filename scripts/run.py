#!/usr/bin/env python
"""CLI entry point: scrape | ingest | reference | chat | serve."""
import os

# Must be set before ANY import that pulls in numpy or ctranslate2. Anaconda's numpy
# and faster-whisper (ctranslate2) ship separate OpenMP runtimes that collide on
# macOS; the duplicate-init guard aborts the process rather than raising, so setting
# it inside voice.py is too late once ingest has already imported numpy.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import chat, ingest, reference, scraper, serve, config


def main() -> None:
    parser = argparse.ArgumentParser(prog="birch-aquarium-llm")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scrape = sub.add_parser("scrape", help="crawl birchaquarium.org")
    p_scrape.add_argument("--max-pages", type=int, default=config.MAX_PAGES)

    sub.add_parser("ingest", help="chunk + embed scraped content")

    sub.add_parser("reference", help="fetch general sea-life reference content (Wikipedia)")

    p_chat = sub.add_parser("chat", help="interactive RAG chat")
    p_chat.add_argument("--ask", help="one-shot question; skip the REPL")
    p_chat.add_argument("--voice", action="store_true", help="use mic input (Whisper)")

    p_serve = sub.add_parser("serve", help="HTTP endpoint for the browser experiment")
    p_serve.add_argument("--port", type=int, default=8077)
    p_serve.add_argument("--host", default="127.0.0.1")

    args = parser.parse_args()

    if args.cmd == "scrape":
        scraper.crawl(max_pages=args.max_pages)
    elif args.cmd == "ingest":
        ingest.build_index()
    elif args.cmd == "reference":
        reference.fetch_all()
    elif args.cmd == "chat":
        if args.ask:
            chat.answer(args.ask)
        else:
            chat.repl(voice_mode=args.voice)
    elif args.cmd == "serve":
        serve.run(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
