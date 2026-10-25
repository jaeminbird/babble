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
import threading
from pathlib import Path

from babble.types import Event

GENESIS = "0" * 64


def _canonical_json(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def _sha256_of_record(record: dict) -> str:
    return hashlib.sha256(_canonical_json(record).encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.events: list[Event] = []
        self._previous_sha256 = GENESIS
        # Children of a parallel batch record concurrently. The lock keeps the chain and file in order.
        self._lock = threading.Lock()
        if path is not None and path.exists():
            # Continue the chain from an existing file rather than starting a new one.
            last_line = _last_line_of_existing_log(path)
            if last_line is not None:
                self._previous_sha256 = json.loads(last_line)["sha256"]

    def record(self, event: Event) -> None:
        with self._lock:
            chained_record = event.to_dict()
            chained_record["prev_sha256"] = self._previous_sha256
            chained_record["sha256"] = _sha256_of_record(chained_record)
            if self.path is not None:
                with open(self.path, "a", encoding="utf-8", newline="\n") as log_file:
                    log_file.write(_canonical_json(chained_record) + "\n")
            self._previous_sha256 = chained_record["sha256"]
            self.events.append(event)

    def of_type(self, type: str) -> list[Event]:
        return [event for event in self.events if event.type == type]


class TaggedAuditLog:
    """Forwards each event to a parent AuditLog with `tags` merged into the event's data.

    A subagent records through one of these, so its events land in the parent log tagged with
    the subagent's name: TaggedAuditLog(parent, subagent="r3").
    """

    def __init__(self, parent: AuditLog, **tags: str) -> None:
        self.parent = parent
        self.tags = tags

    def record(self, event: Event) -> None:
        colliding_keys = self.tags.keys() & event.data.keys()
        if colliding_keys:
            raise ValueError(
                f"tag keys {sorted(colliding_keys)} collide with keys in event {event.type!r}"
            )
        self.parent.record(Event(event.type, {**event.data, **self.tags}, event.ts))


def verify_chain(path: Path) -> bool:
    """True if every line parses, links to the previous line, and matches its own hash."""
    expected_prev_sha256 = GENESIS
    with open(path, encoding="utf-8") as log_file:
        for raw_line in log_file:
            line_text = raw_line.rstrip("\n")
            try:
                parsed_record = json.loads(line_text)
            except ValueError:
                return False
            if not isinstance(parsed_record, dict):
                return False
            claimed_sha256 = parsed_record.pop("sha256", None)
            if not isinstance(claimed_sha256, str):
                return False
            if parsed_record.get("prev_sha256") != expected_prev_sha256:
                return False
            if _sha256_of_record(parsed_record) != claimed_sha256:
                return False
            expected_prev_sha256 = claimed_sha256
    return True


def _last_line_of_existing_log(path: Path) -> str | None:
    """The final line of an existing log, or None for an empty file. Blank lines are an error,
    the same as in verify_chain."""
    last_line: str | None = None
    with open(path, encoding="utf-8") as log_file:
        for line_number, raw_line in enumerate(log_file, start=1):
            if not raw_line.strip():
                raise ValueError(f"{path}:{line_number}: blank line in audit log")
            last_line = raw_line
    return last_line
