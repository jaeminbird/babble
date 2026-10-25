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
    head_length = int(limit * HEAD_SHARE)
    tail_length = int(limit * (1 - HEAD_SHARE))
    characters_dropped = len(text) - head_length - tail_length
    marker = (
        f"\n[... {characters_dropped} characters truncated. Narrow the command, "
        "e.g. a specific neighbor, prefix or interface ...]\n"
    )
    # text[len(text) - 0:] is empty, so a zero-length tail needs no special case.
    tail = text[len(text) - tail_length :]
    return text[:head_length] + marker + tail
