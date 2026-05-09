"""Polite breadth-first crawler for birchaquarium.org.

Honors robots.txt, stays on the allowed host, sleeps between requests, and
writes one markdown file per page to data/raw/.
"""
from __future__ import annotations

import hashlib
import re
import time
from collections import deque
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify

from . import config


def _slug(url: str) -> str:
    h = hashlib.sha1(url.encode()).hexdigest()[:10]
    path = re.sub(r"[^a-z0-9]+", "-", urlparse(url).path.lower()).strip("-") or "index"
    return f"{path}-{h}.md"


def _normalize(url: str, base: str) -> str | None:
    absolute = urljoin(base, url)
    absolute, _ = urldefrag(absolute)
    parsed = urlparse(absolute)
    if parsed.scheme not in ("http", "https"):
        return None
    if parsed.netloc != config.ALLOWED_HOST:
        return None
    if any(parsed.path.lower().endswith(ext) for ext in (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".mp4")):
        return None
    return absolute


def _extract(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    for tag in soup(["script", "style", "nav", "footer", "header", "form", "noscript"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    md = markdownify(str(main), heading_style="ATX").strip()
    md = re.sub(r"\n{3,}", "\n\n", md)
    return title, md


def crawl(seed: str = config.SEED_URL, max_pages: int = config.MAX_PAGES) -> int:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = config.USER_AGENT
    # Anaconda's Python on macOS often can't verify the system cert chain.
    # We're scraping a public, read-only website — disable verification for dev.
    session.verify = False
    requests.packages.urllib3.disable_warnings()

    rp = RobotFileParser()
    robots_url = urljoin(seed, "/robots.txt")
    try:
        r = session.get(robots_url, timeout=10)
        if r.status_code == 200:
            rp.parse(r.text.splitlines())
        else:
            print(f"warning: robots.txt returned {r.status_code}; assuming allow-all")
            rp.parse(["User-agent: *", "Allow: /"])
    except requests.RequestException as e:
        print(f"warning: could not read robots.txt: {e}; assuming allow-all")
        rp.parse(["User-agent: *", "Allow: /"])

    seen: set[str] = set()
    queue: deque[str] = deque([seed])
    written = 0

    while queue and written < max_pages:
        url = queue.popleft()
        if url in seen:
            continue
        seen.add(url)

        if not rp.can_fetch(config.USER_AGENT, url):
            print(f"skip (robots): {url}")
            continue

        try:
            resp = session.get(url, timeout=15)
        except requests.RequestException as e:
            print(f"err  {url}: {e}")
            continue

        if resp.status_code != 200 or "text/html" not in resp.headers.get("content-type", ""):
            continue

        title, body = _extract(resp.text)
        if len(body) < 200:
            continue

        out_path = config.RAW_DIR / _slug(url)
        out_path.write_text(f"# {title}\n\nsource: {url}\n\n{body}\n", encoding="utf-8")
        written += 1
        print(f"[{written}/{max_pages}] {url}")

        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            link = _normalize(a["href"], url)
            if not link or link in seen:
                continue
            # Prioritize exhibit/animal/learn pages by jumping the queue
            if any(p in urlparse(link).path for p in config.PRIORITY_PATHS):
                queue.appendleft(link)
            else:
                queue.append(link)

        time.sleep(config.REQUEST_DELAY_SECONDS)

    print(f"done. wrote {written} pages to {config.RAW_DIR}")
    return written
