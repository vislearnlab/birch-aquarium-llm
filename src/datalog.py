"""Durable logging of every LLM interaction: append-only JSONL + best-effort MongoDB.

Design priority: **a child's session must never fail because logging failed.**

- Every record is written to a local JSONL file synchronously. That file is the
  source of truth — it survives a Mongo outage, so nothing is ever lost and you can
  backfill Mongo from it later.
- The same record is ALSO shipped to MongoDB on a background thread, best-effort.
  pymongo missing, no credentials, or an unreachable server all degrade to
  JSONL-only with a one-time notice — never an exception to the caller.

Connection string: `MONGO_URL` env var, else `mongo_auth.json` ({"url": "..."}) in the
repo root (gitignored, same convention as the birch-ask study). Target defaults to
`birch_ask.llm_demo`, overridable via `MONGO_DB` / `MONGO_COLLECTION`.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone

from . import config

JSONL_PATH = config.DATA_DIR / "sessions.jsonl"
MONGO_DB = os.environ.get("MONGO_DB", "birch_ask")
MONGO_COLLECTION = os.environ.get("MONGO_COLLECTION", "llm_demo")

_write_lock = threading.Lock()
_mongo_collection = None
_mongo_state = "unset"  # unset -> ok | disabled (terminal once resolved)


def _mongo_url() -> str | None:
    if os.environ.get("MONGO_URL"):
        return os.environ["MONGO_URL"]
    for p in (config.ROOT / "mongo_auth.json", config.DATA_DIR / "mongo_auth.json"):
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8")).get("url")
            except Exception as e:
                print(f"[datalog] could not read {p.name}: {e}")
                return None
    return None


def _collection():
    """Return the Mongo collection, or None. Resolves once, then caches the outcome."""
    global _mongo_collection, _mongo_state
    if _mongo_state == "ok":
        return _mongo_collection
    if _mongo_state == "disabled":
        return None

    url = _mongo_url()
    if not url:
        _mongo_state = "disabled"
        print("[datalog] no MONGO_URL / mongo_auth.json — logging to JSONL only")
        return None
    try:
        import pymongo
        client = pymongo.MongoClient(url, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")  # fail fast if unreachable, rather than per-insert
        _mongo_collection = client[MONGO_DB][MONGO_COLLECTION]
        _mongo_state = "ok"
        print(f"[datalog] MongoDB logging -> {MONGO_DB}.{MONGO_COLLECTION}")
        return _mongo_collection
    except ImportError:
        _mongo_state = "disabled"
        print("[datalog] pymongo not installed (`pip install pymongo`) — JSONL only")
    except Exception as e:
        # JSONL still has every record; backfill Mongo from it once it's reachable.
        _mongo_state = "disabled"
        print(f"[datalog] MongoDB unavailable ({e.__class__.__name__}: {e}) — JSONL only")
    return None


def _stamp(record: dict) -> dict:
    """Add UTC timestamps if the caller didn't. Keeps any it already set."""
    now = datetime.now(timezone.utc)
    record.setdefault("ts", now.isoformat())            # ISO-8601 UTC, e.g. 2026-07-31T18:04:11.123+00:00
    record.setdefault("ts_epoch_ms", int(now.timestamp() * 1000))
    return record


def _mongo_insert(record: dict) -> None:
    try:
        coll = _collection()
        if coll is not None:
            coll.insert_one(dict(record))  # copy: insert_one injects _id
    except Exception as e:
        print(f"[datalog] Mongo insert failed (record is safe in JSONL): {e}")


def log_interaction(record: dict) -> None:
    """Persist one interaction. Never raises — logging must not break a session."""
    record = _stamp(dict(record))

    # 1) Durable local JSONL — synchronous, the source of truth.
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        with _write_lock, open(JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        print(f"[datalog] JSONL write failed: {e}")

    # 2) Best-effort Mongo mirror — background thread, never blocks the response.
    threading.Thread(target=_mongo_insert, args=(record,), daemon=True).start()
