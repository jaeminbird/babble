"""Tool output truncation. Applied before a tool result enters history, so history stays bounded."""

from __future__ import annotations

HEAD_SHARE = 0.6


def truncate(text: str, limit: int) -> str:
    """Return text unchanged if it fits in limit characters. Otherwise keep the head (60% of the
    limit) and the tail (40%), with a marker in between that says how much was dropped and how to
    narrow the command."""
    if limit < 0:
        raise ValueError("limit must be >= 0")
    if len(text) <= limit:
        return text
    head_len = int(limit * HEAD_SHARE)
    tail_len = int(limit * (1 - HEAD_SHARE))
    dropped = len(text) - head_len - tail_len
    marker = (
        f"\n[... {dropped} characters truncated. Narrow the command, "
        "e.g. a specific neighbor, prefix or interface ...]\n"
    )
    tail = text[len(text) - tail_len :] if tail_len else ""
    return text[:head_len] + marker + tail
