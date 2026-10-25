"""Memory that survives compaction: a notes store the agent writes to, and a summarizing compactor
that starts a fresh conversation from a progress summary plus those notes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from faultline.types import ToolResult, ToolSpec

NO_NOTES = "(no notes yet)"


class NotesStore:
    """Short working notes. With a path, every change is written to disk as JSON."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._notes: list[str] = []
        if path is not None and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            notes = data.get("notes") if isinstance(data, dict) else None
            if not isinstance(notes, list) or not all(isinstance(n, str) for n in notes):
                raise ValueError(f"{path} is not a notes file")
            self._notes = list(notes)

    def read(self) -> str:
        if not self._notes:
            return NO_NOTES
        return "\n".join(f"{i}. {note}" for i, note in enumerate(self._notes, start=1))

    def render(self) -> str:
        if not self._notes:
            return NO_NOTES
        return "\n".join(self._notes)

    def append(self, text: str) -> None:
        self._notes.append(text)
        self._save()

    def replace(self, text: str) -> None:
        self._notes = [text]
        self._save()

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.write_text(json.dumps({"notes": self._notes}, indent=2), encoding="utf-8")


NOTES_TOOL = "notes"

NOTES_DESCRIPTION = (
    "Keep short working notes that survive context compaction: hypotheses ruled in or out, "
    "key evidence, and devices checked. action 'read' returns the notes; 'append' adds one note "
    "(text required); 'replace' swaps all notes for one note (text required)."
)


def make_notes_tool(store: NotesStore) -> ToolSpec:
    def handler(tool_input: dict) -> ToolResult:
        action = tool_input.get("action")
        text = tool_input.get("text")
        if action == "read":
            return ToolResult(store.read())
        if action not in ("append", "replace"):
            return ToolResult(f"Unknown action: {action!r}", is_error=True)
        if not isinstance(text, str) or not text.strip():
            return ToolResult(f"action {action!r} needs non-empty text", is_error=True)
        if action == "append":
            store.append(text)
            return ToolResult("Note added.")
        store.replace(text)
        return ToolResult("Notes replaced.")

    return ToolSpec(
        name=NOTES_TOOL,
        description=NOTES_DESCRIPTION,
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["read", "append", "replace"]},
                "text": {"type": "string"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        handler=handler,
        mutating=False,
        parallel_safe=False,
    )


# ---------------------------------------------------------------- transcript

RESULT_CHARS = 2000


def render_transcript(messages: list[dict[str, Any]]) -> str:
    """Plain-text rendering of a conversation for the summarizer. One section per message.
    Text is verbatim, calls and results are one line each, thinking is skipped."""
    sections = []
    for message in messages:
        header = f"{str(message.get('role', '')).upper()}:"
        content = message.get("content", "")
        if isinstance(content, str):
            lines = [content]
        else:
            lines = [line for block in content if (line := _render_block(block)) is not None]
        sections.append("\n".join([header, *lines]))
    return "\n\n".join(sections)


def _render_block(block: dict[str, Any]) -> str | None:
    kind = block.get("type")
    if kind == "text":
        return str(block.get("text", ""))
    if kind == "tool_use":
        return f"CALL {block.get('name')} {json.dumps(block.get('input') or {}, sort_keys=True)}"
    if kind == "tool_result":
        label = "RESULT (error)" if block.get("is_error") else "RESULT"
        return f"{label} {_result_text(block.get('content'))[:RESULT_CHARS]}"
    return None  # thinking, redacted_thinking and anything else the summary does not need


def _result_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
