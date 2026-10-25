from typing import Any

from faultline.harness.audit import AuditLog
from faultline.harness.loop import NUDGE, SUBMIT_TOOL, Agent
from faultline.harness.policy import DECLINED_NOTE, AutoApprover, SafetyPolicy, ScriptedApprover
from faultline.llm import FakeLLMClient, assert_append_only, response, text, tool_use
from faultline.types import HarnessConfig, LLMResponse, ToolResult, ToolSpec, Usage

SYSTEM = "You diagnose faults."

DIAGNOSIS = {
    "root_cause_device": "r3",
    "category": "interface_down",
    "summary": "eth3 is administratively down.",
    "evidence": ["show interface eth3 shows shutdown"],
    "fix_applied": True,
}


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


def _show_spec(handler=None, **kwargs: Any) -> ToolSpec:
    return ToolSpec(
        name=kwargs.pop("name", "show"),
        description="Read a device.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=handler or (lambda _input: ToolResult("up")),
        **kwargs,
    )


def _submit(id: str | None = None) -> dict[str, Any]:
    return tool_use(SUBMIT_TOOL, dict(DIAGNOSIS), id=id)


def _make(
    responses: list[LLMResponse],
    tools: list[ToolSpec] | None = None,
    config: HarnessConfig | None = None,
    **kwargs: Any,
) -> tuple[Agent, FakeLLMClient, AuditLog]:
    llm = FakeLLMClient(responses)
    audit = AuditLog()
    agent = Agent(
        llm=llm,
        tools=tools if tools is not None else [_show_spec(), _submit_spec()],
        config=config or HarnessConfig(),
        policy=kwargs.pop("policy", SafetyPolicy()),
        approver=kwargs.pop("approver", AutoApprover()),
        audit=audit,
        system_prompt=SYSTEM,
        **kwargs,
    )
    return agent, llm, audit


def _types(audit: AuditLog) -> list[str]:
    return [e.type for e in audit.events]


# ------------------------------------------------------------------ happy path


def test_reads_then_submits_and_stops_with_diagnosis():
    first = response(tool_use("show", {"device": "r3"}, id="toolu_a"))
    agent, llm, audit = _make([first, response(_submit(id="toolu_b"))])
    result = agent.run("eth3 is down")

    assert result.stop_reason == "diagnosis"
    assert result.turns == 2
    assert result.diagnosis is not None
    assert result.diagnosis.root_cause_device == "r3"
    assert result.diagnosis.fix_applied is True
    assert result.tool_calls == 2

    msgs = result.messages
    assert msgs[0] == {"role": "user", "content": "eth3 is down"}
    # The assistant turn is stored as the block list the API returned.
    assert msgs[1] == {"role": "assistant", "content": first.content}
    # One user message carries the tool result, with no is_error key for a success.
    assert msgs[2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_a", "content": "up"}],
    }
    assert msgs[4]["content"][0]["tool_use_id"] == "toolu_b"

    assert _types(audit)[:2] == ["llm_call", "tool_call"]
    assert "diagnosis" in _types(audit)
    assert _types(audit)[-1] == "stop"
    assert audit.of_type("stop")[0].data == {"reason": "diagnosis", "turns": 2}
    assert_append_only(llm.calls)


def test_request_shape_on_every_call():
    agent, llm, _ = _make([response(tool_use("show", {}, id="t1")), response(_submit())])
    agent.run("task")

    first, second = llm.calls
    assert first["system"] == [{"type": "text", "text": SYSTEM}]
    assert first["tools"] == [spec.to_api() for spec in agent.tools]
    assert [t["name"] for t in first["tools"]] == ["show", SUBMIT_TOOL]
    assert first["cache"] is True
    assert first["effort"] == "medium"
    assert first["max_tokens"] == 16000
    assert first["model"] == "claude-opus-5-5"
    # The second request carries the first turn: task, assistant turn, tool results.
    assert len(second["messages"]) == 3
    assert second["messages"][0] == first["messages"][0]
    assert second["messages"][1]["role"] == "assistant"
    assert second["messages"][2]["role"] == "user"
    assert_append_only(llm.calls)


def test_usage_and_cost_are_accumulated():
    usage = Usage(input_tokens=1000, output_tokens=100)
    agent, llm, audit = _make(
        [
            response(tool_use("show", {}), usage=usage),
            response(_submit(), usage=usage),
        ]
    )
    result = agent.run("task")
    assert result.usage == Usage(2000, 200)
    assert result.cost_usd > 0
    assert audit.of_type("llm_call")[0].data["usage"]["input_tokens"] == 1000
    assert_append_only(llm.calls)


# ------------------------------------------------------------- tool execution


def test_unknown_tool_gets_an_error_result_and_the_run_goes_on():
    agent, llm, audit = _make(
        [response(tool_use("reboot_router", {}, id="t1")), response(_submit())]
    )
    result = agent.run("task")
    block = result.messages[2]["content"][0]
    assert block == {
        "type": "tool_result",
        "tool_use_id": "t1",
        "content": "Unknown tool: reboot_router",
        "is_error": True,
    }
    assert result.stop_reason == "diagnosis"
    assert audit.of_type("tool_result")[0].data["is_error"] is True
    assert_append_only(llm.calls)


def test_handler_exception_becomes_error_result_and_does_not_crash():
    def boom(_input):
        raise ValueError("boom")

    agent, llm, _ = _make(
        [response(tool_use("show", {}, id="t1")), response(_submit())],
        tools=[_show_spec(handler=boom), _submit_spec()],
    )
    result = agent.run("task")
    block = result.messages[2]["content"][0]
    assert block["content"] == "ValueError: boom"
    assert block["is_error"] is True
    assert result.stop_reason == "diagnosis"
    assert_append_only(llm.calls)


def test_tool_output_is_truncated_before_it_enters_history():
    big = "y" * 5000
    tools = [_show_spec(handler=lambda _i: ToolResult(big)), _submit_spec()]
    agent, llm, audit = _make(
        [response(tool_use("show", {}, id="t1")), response(_submit())], tools=tools
    )
    result = agent.run("task")
    content = result.messages[2]["content"][0]["content"]
    assert "characters truncated" in content
    assert len(content) < len(big)
    assert audit.of_type("tool_result")[0].data["chars"] == len(content)
    assert_append_only(llm.calls)


def test_truncation_off_keeps_full_output():
    big = "y" * 5000
    tools = [_show_spec(handler=lambda _i: ToolResult(big)), _submit_spec()]
    config = HarnessConfig(truncate_tool_output=None)
    agent, llm, _ = _make(
        [response(tool_use("show", {}, id="t1")), response(_submit())], tools=tools, config=config
    )
    result = agent.run("task")
    assert result.messages[2]["content"][0]["content"] == big
    assert_append_only(llm.calls)


# ---------------------------------------------------------------- stop paths


def test_nudge_then_submit_succeeds():
    agent, llm, _ = _make([response(text("Thinking about it.")), response(_submit())])
    result = agent.run("task")
    assert result.stop_reason == "diagnosis"
    assert result.messages[2] == {"role": "user", "content": NUDGE}
    assert_append_only(llm.calls)


def test_nudge_then_no_diagnosis_stops_the_run():
    agent, llm, _ = _make([response(text("Still thinking.")), response(text("Done, I think."))])
    result = agent.run("task")
    assert result.stop_reason == "no_diagnosis"
    assert result.diagnosis is None
    assert result.turns == 2
    assert len(llm.calls) == 2
    assert_append_only(llm.calls)


def test_max_turns_stops_the_run():
    config = HarnessConfig(max_turns=2)
    agent, llm, audit = _make(
        [
            response(tool_use("show", {}, id="t1")),
            response(tool_use("show", {}, id="t2")),
        ],
        config=config,
    )
    result = agent.run("task")
    assert result.stop_reason == "max_turns"
    assert result.turns == 2
    assert len(llm.calls) == 2
    assert audit.of_type("stop")[0].data == {"reason": "max_turns", "turns": 2}
    assert_append_only(llm.calls)


class _RaisingLLM:
    def complete(self, **_kwargs: Any) -> LLMResponse:
        raise RuntimeError("network down")


def test_llm_error_stops_the_run_with_an_error_event():
    audit = AuditLog()
    agent = Agent(
        llm=_RaisingLLM(),
        tools=[_show_spec(), _submit_spec()],
        config=HarnessConfig(),
        policy=SafetyPolicy(),
        approver=AutoApprover(),
        audit=audit,
        system_prompt=SYSTEM,
    )
    result = agent.run("task")
    assert result.stop_reason == "error"
    assert result.turns == 0
    assert result.diagnosis is None
    errors = audit.of_type("error")
    assert errors[0].data["message"] == "RuntimeError: network down"
    assert audit.of_type("stop")[0].data["reason"] == "error"


# ------------------------------------------------------- policy and approval


def _change_spec(handler=None) -> ToolSpec:
    return _show_spec(
        handler=handler or (lambda _i: ToolResult("applied")),
        name="propose_change",
        mutating=True,
        parallel_safe=False,
    )


def _change_then_submit(change_input: dict | None = None) -> list[LLMResponse]:
    return [
        response(tool_use("propose_change", change_input or {"lines": ["x"]}, id="t1")),
        response(_submit()),
    ]


def test_mutating_call_runs_after_approval():
    approver = ScriptedApprover([True])
    agent, llm, audit = _make(
        _change_then_submit(),
        tools=[_change_spec(), _submit_spec()],
        approver=approver,
    )
    result = agent.run("task")
    assert result.stop_reason == "diagnosis"
    assert result.approvals == 1
    assert result.denials == 0
    assert result.messages[2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "t1",
        "content": "applied",
    }
    assert len(approver.calls) == 1
    assert audit.of_type("approval")[0].data == {
        "id": "t1",
        "name": "propose_change",
        "approved": True,
        "note": "approved by script",
    }
    assert_append_only(llm.calls)


def test_declined_call_gets_the_note_and_never_runs():
    ran = []
    tools = [_change_spec(handler=lambda _i: ran.append(1) or ToolResult("applied")), _submit_spec()]
    agent, llm, audit = _make(
        _change_then_submit(), tools=tools, approver=ScriptedApprover([False])
    )
    result = agent.run("task")
    block = result.messages[2]["content"][0]
    assert block["is_error"] is True
    assert block["content"] == DECLINED_NOTE
    assert ran == []
    assert result.denials == 1
    assert result.approvals == 0
    assert audit.of_type("approval")[0].data["approved"] is False
    assert_append_only(llm.calls)


def test_policy_denial_never_reaches_the_approver_or_the_handler():
    ran = []
    tools = [
        _show_spec(handler=lambda _i: ran.append(1) or ToolResult("up")),
        _submit_spec(),
    ]
    approver = ScriptedApprover([])  # any approval request would fail the test
    agent, llm, audit = _make(
        [response(tool_use("show", {}, id="t1")), response(_submit())],
        tools=tools,
        policy=SafetyPolicy(deny_tools=["show"]),
        approver=approver,
    )
    result = agent.run("task")
    block = result.messages[2]["content"][0]
    assert block["is_error"] is True
    assert block["content"].startswith("Denied by policy: ")
    assert ran == []
    assert approver.calls == []
    assert result.denials == 1
    assert audit.of_type("denied")[0].data["name"] == "show"
    assert_append_only(llm.calls)


def test_safety_disabled_runs_mutating_tools_without_approval():
    approver = ScriptedApprover([])
    agent, llm, audit = _make(
        _change_then_submit(),
        tools=[_change_spec(), _submit_spec()],
        policy=SafetyPolicy(enabled=False),
        approver=approver,
    )
    result = agent.run("task")
    assert result.messages[2]["content"][0]["content"] == "applied"
    assert approver.calls == []
    assert audit.of_type("approval") == []
    assert_append_only(llm.calls)


def test_allow_list_skips_the_approver():
    approver = ScriptedApprover([])
    agent, llm, _ = _make(
        _change_then_submit(),
        tools=[_change_spec(), _submit_spec()],
        policy=SafetyPolicy(allow_tools=["propose_change"]),
        approver=approver,
    )
    result = agent.run("task")
    assert result.messages[2]["content"][0]["content"] == "applied"
    assert approver.calls == []
    assert_append_only(llm.calls)


def test_approver_that_raises_declines_the_call():
    class Broken:
        def approve(self, call, reason):
            raise RuntimeError("terminal closed")

    agent, llm, _ = _make(
        _change_then_submit(), tools=[_change_spec(), _submit_spec()], approver=Broken()
    )
    result = agent.run("task")
    block = result.messages[2]["content"][0]
    assert block["is_error"] is True
    assert "terminal closed" in block["content"]
    assert result.denials == 1
    assert_append_only(llm.calls)


def test_pre_hook_can_short_circuit_a_call():
    ran = []
    tools = [_show_spec(handler=lambda _i: ran.append(1) or ToolResult("up")), _submit_spec()]

    def cached(call):
        return ToolResult("cached: up") if call.name == "show" else None

    agent, llm, _ = _make(
        [response(tool_use("show", {}, id="t1")), response(_submit())],
        tools=tools,
        pre_hooks=[cached],
    )
    result = agent.run("task")
    assert result.messages[2]["content"][0]["content"] == "cached: up"
    assert ran == []
    assert_append_only(llm.calls)


def test_post_hook_can_replace_a_result_and_a_failing_hook_is_an_error():
    def redact(call, result):
        return ToolResult("[redacted]")

    agent, llm, _ = _make(
        [response(tool_use("show", {}, id="t1")), response(_submit())],
        post_hooks=[redact],
    )
    result = agent.run("task")
    assert result.messages[2]["content"][0]["content"] == "[redacted]"
    assert_append_only(llm.calls)

    def broken(call, result):
        raise KeyError("oops")

    agent, llm, _ = _make(
        [response(tool_use("show", {}, id="t1")), response(_submit())],
        post_hooks=[broken],
    )
    result = agent.run("task")
    block = result.messages[2]["content"][0]
    assert block["is_error"] is True
    assert block["content"] == "KeyError: 'oops'"
    assert_append_only(llm.calls)
