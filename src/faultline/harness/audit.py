"""Hash-chained JSONL audit log.

Each line is a JSON object holding the event (type, data, ts), `prev_sha256` (the previous line's
sha256, or 64 zeros for the first line) and `sha256`, the hash of the line's record with the
`sha256` field removed. Editing a line, deleting one from the middle, or reordering lines breaks
the chain, and `verify_chain` reports it.

Limit: the chain cannot detect a truncated tail or a fully rewritten file. Those need an
external anchor (for example, the last sha256 recorded somewhere else).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from faultline.types import Event

GENESIS = "0" * 64


def _digest(record: dict) -> str:
    body = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.events: list[Event] = []
        self._prev = GENESIS
        if path is not None and path.exists():
            # Continue the chain from an existing file rather than starting a new one.
            last = _last_line(path)
            if last is not None:
                self._prev = json.loads(last)["sha256"]

    def record(self, event: Event) -> None:
        rec = event.to_dict()
        rec["prev_sha256"] = self._prev
        rec["sha256"] = _digest(rec)
        if self.path is not None:
            line = json.dumps(rec, sort_keys=True, separators=(",", ":"))
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")
        self._prev = rec["sha256"]
        self.events.append(event)

    def of_type(self, type: str) -> list[Event]:
        return [e for e in self.events if e.type == type]


def verify_chain(path: Path) -> bool:
    """True if every line parses, links to the previous line, and matches its own hash."""
    prev = GENESIS
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            try:
                rec = json.loads(line)
            except ValueError:
                return False
            if not isinstance(rec, dict):
                return False
            claimed = rec.pop("sha256", None)
            if not isinstance(claimed, str):
                return False
            if rec.get("prev_sha256") != prev:
                return False
            if _digest(rec) != claimed:
                return False
            prev = claimed
    return True


def _last_line(path: Path) -> str | None:
    last: str | None = None
    with open(path, encoding="utf-8") as f:
        for raw in f:
            if raw.strip():
                last = raw.strip()
    return last
