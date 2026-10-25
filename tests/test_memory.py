import json
from pathlib import Path

import pytest

from babble.harness.accounting import Accounting
from babble.harness.audit import AuditLog
from babble.harness.loop import SUBMIT_TOOL, Agent
from babble.harness.memory import (
    NO_NOTES,
    NotesStore,
    SummaryCompactor,
    make_notes_tool,
    render_transcript,
)
from babble.harness.policy import AutoApprover, SafetyPolicy
from babble.llm import FakeLLMClient, assert_append_only, response, text, tool_use
from babble.types import HarnessConfig, ToolResult, ToolSpec, Usage


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


# ---------------------------------------------------------------- transcript


def test_render_transcript_formats_each_block_type():
    messages = [
        {"role": "user", "content": "Diagnose ha cannot reach hb."},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "secret reasoning", "signature": "x"},
                {"type": "text", "text": "Checking ra first."},
                {"type": "tool_use", "id": "t1", "name": "show", "input": {"device": "ra"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "eth2 up"},
                {
                    "type": "tool_result",
                    "tool_use_id": "t2",
                    "content": "no such device",
                    "is_error": True,
                },
            ],
        },
    ]
    rendered = render_transcript(messages)
    assert rendered == (
        "USER:\nDiagnose ha cannot reach hb.\n\n"
        "ASSISTANT:\nChecking ra first.\n"
        'CALL show {"device": "ra"}\n\n'
        "USER:\nRESULT eth2 up\nRESULT (error) no such device"
    )
    assert "secret reasoning" not in rendered


def test_render_transcript_truncates_each_result_to_2000_chars():
    long = "x" * 5000
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": long},
                {"type": "tool_result", "tool_use_id": "b", "content": long},
            ],
        }
    ]
    lines = render_transcript(messages).split("\n")
    assert lines[1] == "RESULT " + "x" * 2000
    assert lines[2] == "RESULT " + "x" * 2000


def test_render_transcript_skips_thinking_only_content_and_keeps_section():
    messages = [
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"}]},
    ]
    assert render_transcript(messages) == "ASSISTANT:"


def test_render_transcript_does_not_mutate_messages():
    import copy

    messages = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t", "name": "show", "input": {"a": 1}}],
        },
    ]
    before = copy.deepcopy(messages)
    render_transcript(messages)
    assert messages == before


# ---------------------------------------------------------------- compactor


def _transcript_messages() -> list[dict]:
    return [
        {"role": "user", "content": "ha cannot reach hb. Diagnose it."},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "show", "input": {"device": "ra"}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "eth2 up"}],
        },
    ]


def test_should_compact_at_threshold():
    compactor = SummaryCompactor(FakeLLMClient([]), "claude-haiku-5-5", threshold=1000)
    assert compactor.should_compact(Usage(input_tokens=999)) is False
    assert compactor.should_compact(Usage(input_tokens=1000)) is True
    # Cache reads and writes count toward the input the model had to read.
    split = Usage(input_tokens=400, cache_creation_input_tokens=300, cache_read_input_tokens=300)
    assert compactor.should_compact(split) is True


def test_compact_makes_one_toolless_summary_call():
    llm = FakeLLMClient([response(text("Summary: r1 is fine."))])
    compactor = SummaryCompactor(llm, "claude-haiku-5-5", threshold=1000)
    messages = _transcript_messages()
    compactor.compact("Diagnose it.", messages)

    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["tools"] == []
    assert call["effort"] == "low"
    assert call["cache"] is False
    assert call["messages"] == [{"role": "user", "content": render_transcript(messages)}]
    assert len(call["system"]) == 1
    assert "progress summary" in call["system"][0]["text"].lower()


def test_compact_returns_single_user_message_with_task_summary_and_notes(tmp_path: Path):
    store = NotesStore(tmp_path / "notes.json")
    store.append("rdc eth3 is shutdown")
    llm = FakeLLMClient([response(text("  Ruled out ospf.  "))])
    compactor = SummaryCompactor(llm, "claude-haiku-5-5", threshold=1000, notes=store)

    fresh = compactor.compact("Diagnose it.", _transcript_messages())

    assert fresh == [
        {
            "role": "user",
            "content": (
                "Diagnose it.\n\n"
                "<progress_summary>\nRuled out ospf.\n</progress_summary>\n\n"
                "<notes>\nrdc eth3 is shutdown\n</notes>\n\n"
                "Context was compacted. Continue the investigation from the summary."
            ),
        }
    ]


def test_compact_without_notes_omits_notes_section():
    llm = FakeLLMClient([response(text("Summary."))])
    compactor = SummaryCompactor(llm, "claude-haiku-5-5", threshold=1000)
    (fresh,) = compactor.compact("Diagnose it.", _transcript_messages())
    assert "<notes>" not in fresh["content"]
    assert "<progress_summary>\nSummary.\n</progress_summary>" in fresh["content"]


def test_compact_adds_summary_usage_to_accounting():
    usage = Usage(input_tokens=5000, output_tokens=300)
    llm = FakeLLMClient([response(text("s"), usage=usage, model="claude-haiku-5-5")])
    accounting = Accounting()
    compactor = SummaryCompactor(llm, "claude-haiku-5-5", threshold=1000, accounting=accounting)
    compactor.compact("task", _transcript_messages())
    assert accounting.usage == usage


def test_compact_never_mutates_messages():
    import copy

    messages = _transcript_messages()
    before = copy.deepcopy(messages)
    llm = FakeLLMClient([response(text("s"))])
    SummaryCompactor(llm, "claude-haiku-5-5", threshold=1000).compact("task", messages)
    assert messages == before


# ------------------------------------------------------------- loop integration

TASK = "ha cannot reach hb 192.168.2.10. Find the root cause."
DIAGNOSIS = {
    "root_cause_device": "rb",
    "category": "interface_down",
    "summary": "eth2 on rb is administratively down.",
    "evidence": ["show interface eth2 on rb shows shutdown"],
    "fix_applied": False,
}


def _show_spec() -> ToolSpec:
    return ToolSpec(
        name="show",
        description="Read a device.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda _input: ToolResult("eth2 is administratively down"),
    )


def _submit_spec() -> ToolSpec:
    return ToolSpec(
        name=SUBMIT_TOOL,
        description="Record the diagnosis.",
        input_schema={
            "type": "object",
            "properties": {
                "root_cause_device": {"type": "string"},
                "category": {"type": "string"},
                "summary": {"type": "string"},
                "evidence": {"type": "array", "items": {"type": "string"}},
                "fix_applied": {"type": "boolean"},
            },
            "required": list(DIAGNOSIS),
            "additionalProperties": False,
        },
        handler=lambda _input: ToolResult("Diagnosis recorded."),
        strict=True,
    )


def test_agent_compacts_into_fresh_conversation_and_finishes(tmp_path: Path):
    store = NotesStore(tmp_path / "notes.json")
    store.append("rdc eth3 is shutdown")
    llm = FakeLLMClient(
        [
            # Turn 1: small usage, one tool call.
            response(tool_use("show", {}), usage=Usage(input_tokens=1000)),
            # Turn 2: high usage, so the next turn compacts first.
            response(tool_use("show", {}), usage=Usage(input_tokens=200_000)),
            # The compactor's summary call.
            response(text("Checked rb: eth2 is down."), model="claude-haiku-5-5"),
            # Turn 3, in the fresh conversation: the diagnosis.
            response(tool_use(SUBMIT_TOOL, dict(DIAGNOSIS))),
        ]
    )
    accounting = Accounting()
    audit = AuditLog()
    compactor = SummaryCompactor(
        llm, "claude-haiku-5-5", threshold=100_000, notes=store, accounting=accounting
    )
    agent = Agent(
        llm=llm,
        tools=[_show_spec(), _submit_spec(), make_notes_tool(store)],
        config=HarnessConfig(),
        policy=SafetyPolicy(),
        approver=AutoApprover(),
        audit=audit,
        system_prompt="You diagnose faults.",
        accounting=accounting,
        compactor=compactor,
    )

    result = agent.run(TASK)

    assert result.stop_reason == "diagnosis"
    assert result.diagnosis is not None
    assert result.diagnosis.root_cause_device == "rb"
    assert len(audit.of_type("compaction")) == 1

    agent_calls = [
        call for call in llm.calls if call["system"][0]["text"] == "You diagnose faults."
    ]
    summary_call = llm.calls[2]
    before, after = agent_calls[:2], agent_calls[2]

    # The summary call is a single toolless call over the transcript.
    assert summary_call["tools"] == []
    assert summary_call["messages"][0]["content"].startswith("USER:\n" + TASK)
    assert "CALL show" in summary_call["messages"][0]["content"]

    # The agent's first conversation grew append-only up to the compaction.
    assert_append_only(before)
    assert len(before[1]["messages"]) > len(before[0]["messages"])

    # The call after compaction starts a new conversation with the summary and notes.
    assert after["messages"][0]["content"] != before[0]["messages"][0]["content"]
    assert_append_only([after])
    assert len(after["messages"]) == 1
    fresh = after["messages"][0]["content"]
    assert fresh.startswith(TASK)
    assert "<progress_summary>\nChecked rb: eth2 is down.\n</progress_summary>" in fresh
    assert "<notes>\nrdc eth3 is shutdown\n</notes>" in fresh


def test_compact_raises_when_the_summary_is_empty():
    llm = FakeLLMClient([response(text("   "))])
    compactor = SummaryCompactor(llm, "claude-haiku-5-5", threshold=1000)
    with pytest.raises(ValueError, match="empty summary"):
        compactor.compact("Diagnose it.", _transcript_messages())
