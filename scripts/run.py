#!/usr/bin/env python
"""CLI entry point: scrape | ingest | chat."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import chat, ingest, scraper, config


def main() -> None:
    parser = argparse.ArgumentParser(prog="birch-aquarium-llm")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scrape = sub.add_parser("scrape", help="crawl birchaquarium.org")
    p_scrape.add_argument("--max-pages", type=int, default=config.MAX_PAGES)

    sub.add_parser("ingest", help="chunk + embed scraped content")

    p_chat = sub.add_parser("chat", help="interactive RAG chat")
    p_chat.add_argument("--ask", help="one-shot question; skip the REPL")
    p_chat.add_argument("--voice", action="store_true", help="use mic input (Whisper)")

    args = parser.parse_args()

    if args.cmd == "scrape":
        scraper.crawl(max_pages=args.max_pages)
    elif args.cmd == "ingest":
        ingest.build_index()
    elif args.cmd == "chat":
        if args.ask:
            chat.answer(args.ask)
        else:
            chat.repl(voice_mode=args.voice)


if __name__ == "__main__":
    main()
