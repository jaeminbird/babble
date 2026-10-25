"""Command validation: read-only show commands and blocked config lines."""

from __future__ import annotations

import re

MAX_SHOW_LENGTH = 200
_FORBIDDEN_CHARS = ("|", ";", "&", "$", "`", "<", ">", "\n", "\r")

# Config lines that destroy state or change access. Matched against the stripped, lowercased line.
BLOCKED_CONFIG_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p)
    for p in (
        r"^no router (bgp|ospf)",
        r"^no interface",
        r"^hostname",
        r"^(write|copy|erase|reload)",
        r"^no ip forwarding",
        r"^password",
        r"^enable password",
        r"^line vty",
        r"^username",
    )
]


def validate_show(command: str) -> str | None:
    """Return an error message if `command` is not an acceptable read-only show command, else None."""
    cmd = command.strip()
    if not cmd.lower().startswith("show "):
        return "command must start with 'show '"
    if len(cmd) > MAX_SHOW_LENGTH:
        return f"command is longer than {MAX_SHOW_LENGTH} characters"
    for ch in _FORBIDDEN_CHARS:
        if ch in cmd:
            return f"command contains forbidden character {ch!r}"
    if ".." in cmd:
        return "command contains forbidden sequence '..'"
    return None


def blocked_config(lines: list[str]) -> list[str]:
    """Return one reason per config line that matches a blocked pattern. Empty list means all OK."""
    reasons: list[str] = []
    for line in lines:
        normalized = line.strip().lower()
        for pattern in BLOCKED_CONFIG_PATTERNS:
            if pattern.search(normalized):
                reasons.append(f"blocked config line {line.strip()!r} (matches {pattern.pattern})")
                break
    return reasons
