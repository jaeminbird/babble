"""Command validation: read-only show commands and blocked config lines."""

from __future__ import annotations

import re

MAX_SHOW_COMMAND_CHARACTERS = 200
_FORBIDDEN_SHELL_CHARACTERS = ("|", ";", "&", "$", "`", "<", ">", "\n", "\r")

# Config lines that destroy state or change access. Matched against the stripped, lowercased line.
_BLOCKED_CONFIG_PATTERNS: list[re.Pattern[str]] = [
    re.compile(pattern_text)
    for pattern_text in (
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


def validate_read_only_show_command(command: str) -> str | None:
    """Return an error message if `command` is not an acceptable read-only show command, else None."""
    stripped_command = command.strip()
    if not stripped_command.lower().startswith("show "):
        return "command must start with 'show '"
    if len(stripped_command) > MAX_SHOW_COMMAND_CHARACTERS:
        return f"command is longer than {MAX_SHOW_COMMAND_CHARACTERS} characters"
    for forbidden_character in _FORBIDDEN_SHELL_CHARACTERS:
        if forbidden_character in stripped_command:
            return f"command contains forbidden character {forbidden_character!r}"
    if ".." in stripped_command:
        return "command contains forbidden sequence '..'"
    return None


def find_blocked_config_reasons(lines: list[str]) -> list[str]:
    """Return one reason per config line that matches a blocked pattern. Empty list means all OK."""
    reasons: list[str] = []
    for line in lines:
        stripped_line = line.strip()
        normalized_line = stripped_line.lower()
        for blocked_pattern in _BLOCKED_CONFIG_PATTERNS:
            if blocked_pattern.search(normalized_line):
                reasons.append(
                    f"blocked config line {stripped_line!r} (matches {blocked_pattern.pattern})"
                )
                break
    return reasons
