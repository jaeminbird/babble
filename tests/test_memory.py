import json
from pathlib import Path

from faultline.harness.memory import NO_NOTES, NotesStore, make_notes_tool
from faultline.types import ToolResult


def _run(store: NotesStore, **tool_input) -> ToolResult:
    return make_notes_tool(store).handler(tool_input)


# --------------------------------------------------------------------- store


def test_empty_store_reads_placeholder():
    store = NotesStore()
    assert store.read() == NO_NOTES == "(no notes yet)"
    assert store.render() == NO_NOTES


def test_append_and_replace_in_memory():
    store = NotesStore()
    store.append("bgp on r1 is down")
    store.append("eth3 on rdc is shutdown")
    assert store.read() == "1. bgp on r1 is down\n2. eth3 on rdc is shutdown"
    assert store.render() == "bgp on r1 is down\neth3 on rdc is shutdown"

    store.replace("only the summary survives")
    assert store.read() == "1. only the summary survives"
    assert store.render() == "only the summary survives"


def test_persists_every_change_and_reloads(tmp_path: Path):
    path = tmp_path / "notes.json"
    store = NotesStore(path)
    assert not path.exists()

    store.append("ruled out ospf")
    assert json.loads(path.read_text(encoding="utf-8")) == {"notes": ["ruled out ospf"]}

    store.append("checked ra")
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "notes": ["ruled out ospf", "checked ra"]
    }

    store.replace("new state")
    assert json.loads(path.read_text(encoding="utf-8")) == {"notes": ["new state"]}

    reloaded = NotesStore(path)
    assert reloaded.read() == "1. new state"


def test_loads_existing_file_before_appending(tmp_path: Path):
    path = tmp_path / "notes.json"
    path.write_text(json.dumps({"notes": ["a", "b"]}), encoding="utf-8")
    store = NotesStore(path)
    store.append("c")
    assert store.render() == "a\nb\nc"


# --------------------------------------------------------------------- tool


def test_tool_spec_is_read_only_and_not_parallel():
    spec = make_notes_tool(NotesStore())
    assert spec.name == "notes"
    assert spec.mutating is False
    assert spec.parallel_safe is False
    assert spec.input_schema["required"] == ["action"]
    assert spec.input_schema["additionalProperties"] is False
    assert spec.input_schema["properties"]["action"]["enum"] == ["read", "append", "replace"]
    assert spec.input_schema["properties"]["text"]["type"] == "string"


def test_tool_read_returns_notes():
    store = NotesStore()
    store.append("first")
    result = _run(store, action="read")
    assert result == ToolResult("1. first")
    assert _run(NotesStore(), action="read").content == NO_NOTES


def test_tool_append_and_replace_change_store():
    store = NotesStore()
    assert _run(store, action="append", text="one").is_error is False
    assert _run(store, action="append", text="two").is_error is False
    assert store.render() == "one\ntwo"

    assert _run(store, action="replace", text="all").is_error is False
    assert store.render() == "all"


def test_tool_append_and_replace_require_text():
    store = NotesStore()
    for action in ("append", "replace"):
        missing = _run(store, action=action)
        assert missing.is_error is True
        blank = _run(store, action=action, text="   ")
        assert blank.is_error is True
    assert store.read() == NO_NOTES


def test_tool_rejects_unknown_action():
    result = _run(NotesStore(), action="delete")
    assert result.is_error is True
