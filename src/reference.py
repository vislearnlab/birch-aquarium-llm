"""Fetch general sea-life reference content to sit alongside the Birch corpus.

Why: the Birch scrape covers the animals Birch exhibits and little else. For species
it barely mentions (epaulette shark, African penguin, bat star), retrieval returns
whatever Birch page is nearest — mangroves, newsroom archives — and the model fills
the gap by confabulating. Measured error modes included telling a child octopuses are
"safe to touch", answering an African-penguin question about little blue penguins, and
inverting seahorse reproduction (males carry the eggs, not females).

Source is Wikipedia: CC BY-SA, well curated, and organised exactly at the
species level the experiment's depth manipulation depends on.

Files are written in the same format the scraper uses, so `ingest` needs no changes.

    python scripts/run.py reference     # fetch
    python scripts/run.py ingest        # rebuild the index over Birch + reference
"""
from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request

from . import config

API = "https://en.wikipedia.org/w/api.php"
UA = "birch-aquarium-llm/0.2 (research; contact: brialorelle@gmail.com)"
MAX_CHARS = 24000          # plenty for species facts; keeps the index balanced

# The 18 stimulus exemplars, by their Wikipedia titles (redirects are followed).
SPECIES = [
    "Leopard shark", "Horn shark", "Epaulette shark",
    "Giant Pacific octopus", "Octopus rubescens", "Octopus bimaculoides",
    "Little penguin", "Gentoo penguin", "African penguin",
    "Loggerhead sea turtle", "Green sea turtle", "Hawksbill sea turtle",
    "Dwarf seahorse", "Big-bellied seahorse", "Common seadragon",
    "Sunflower sea star", "Ochre sea star", "Patiria miniata",
]

# Basic-level categories and the concepts children's questions keep landing on.
GENERAL = [
    "Shark", "Octopus", "Penguin", "Sea turtle", "Seahorse", "Starfish",
    "Cephalopod", "Chromatophore", "Camouflage", "Bioluminescence",
    "Kelp forest", "Tide pool", "Coral reef", "Ocean", "Marine biology",
    "Fish", "Gill", "Plankton", "Aquarium", "Scripps Institution of Oceanography",
]

TITLES = SPECIES + GENERAL


def _fetch(title: str) -> tuple[str, str] | None:
    """Return (resolved_title, plaintext) for a Wikipedia article."""
    params = {
        "action": "query", "format": "json", "prop": "extracts",
        "explaintext": "1", "redirects": "1", "titles": title,
    }
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        import json
        data = json.load(r)
    pages = (data.get("query") or {}).get("pages") or {}
    for pid, page in pages.items():
        if pid == "-1" or "extract" not in page:
            return None
        text = (page.get("extract") or "").strip()
        if len(text) < 400:
            return None
        return page.get("title", title), text[:MAX_CHARS]
    return None


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def fetch_all(titles: list[str] | None = None) -> None:
    titles = titles or TITLES
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    ok = failed = 0
    for t in titles:
        try:
            got = _fetch(t)
        except Exception as e:
            print(f"  ! {t}: {e}")
            failed += 1
            time.sleep(0.4)
            continue
        if not got:
            print(f"  ! {t}: no article")
            failed += 1
            time.sleep(0.4)
            continue
        resolved, text = got
        url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(resolved.replace(" ", "_"))
        out = config.RAW_DIR / f"wikipedia-{_slug(resolved)}.md"
        out.write_text(f"# {resolved}\n\nsource: {url}\n\n{text}\n", encoding="utf-8")
        print(f"  ✓ {resolved:44} {len(text):>6} chars")
        ok += 1
        time.sleep(0.4)          # be polite
    print(f"\n{ok} fetched, {failed} failed -> {config.RAW_DIR}")
    print("now run:  python scripts/run.py ingest")
