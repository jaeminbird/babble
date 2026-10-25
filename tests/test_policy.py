import json

import pytest

from babble.harness.policy import (
    DECLINED_NOTE,
    AutoApprover,
    AutoDenier,
    CliApprover,
    SafetyPolicy,
    ScriptedApprover,
    make_approver,
)
from babble.types import ToolCall, ToolResult, ToolSpec


def _spec(name: str, mutating: bool) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda _input: ToolResult("ok"),
        mutating=mutating,
    )


def _call(name: str, input: dict | None = None) -> ToolCall:
    return ToolCall(id="t1", name=name, input=input or {})


READ = _spec("show_device", mutating=False)
WRITE = _spec("configure_device", mutating=True)


# ----------------------------------------------------------------- decide()


def test_read_only_tool_is_allowed_by_default():
    decision, reason = SafetyPolicy().decide(_call("show_device"), READ)
    assert decision == "allow"
    assert "read-only" in reason


def test_mutating_tool_asks_by_default():
    decision, reason = SafetyPolicy().decide(_call("configure_device"), WRITE)
    assert decision == "ask"
    assert "requires approval" in reason


def test_deny_tools_denies_read_only_tool():
    policy = SafetyPolicy(deny_tools={"show_device"})
    decision, reason = policy.decide(_call("show_device"), READ)
    assert decision == "deny"
    assert "denied" in reason


def test_deny_tools_denies_mutating_tool():
    policy = SafetyPolicy(deny_tools={"configure_device"})
    decision, _ = policy.decide(_call("configure_device"), WRITE)
    assert decision == "deny"


def test_deny_wins_over_allow_tools():
    policy = SafetyPolicy(deny_tools={"configure_device"}, allow_tools={"configure_device"})
    decision, _ = policy.decide(_call("configure_device"), WRITE)
    assert decision == "deny"


def test_deny_wins_when_safety_disabled():
    # The unsafe baseline still honours an explicit deny list.
    policy = SafetyPolicy(enabled=False, deny_tools={"configure_device"})
    decision, _ = policy.decide(_call("configure_device"), WRITE)
    assert decision == "deny"


def test_safety_disabled_allows_mutating_tool():
    policy = SafetyPolicy(enabled=False)
    decision, reason = policy.decide(_call("configure_device"), WRITE)
    assert decision == "allow"
    assert "safety disabled" in reason


def test_safety_disabled_allows_read_only_tool():
    decision, _ = SafetyPolicy(enabled=False).decide(_call("show_device"), READ)
    assert decision == "allow"


def test_allow_tools_skips_approval_for_mutating_tool():
    policy = SafetyPolicy(allow_tools={"configure_device"})
    decision, reason = policy.decide(_call("configure_device"), WRITE)
    assert decision == "allow"
    assert "pre-approved" in reason


def test_allow_tools_does_not_affect_other_tools():
    policy = SafetyPolicy(allow_tools={"show_device"})
    decision, _ = policy.decide(_call("configure_device"), WRITE)
    assert decision == "ask"


def test_decision_is_keyed_on_call_name():
    # A call whose name is not in deny_tools is not denied, even if the spec is.
    policy = SafetyPolicy(deny_tools={"other_tool"})
    decision, _ = policy.decide(_call("configure_device"), WRITE)
    assert decision == "ask"


def test_accepts_any_iterable_for_tool_sets():
    policy = SafetyPolicy(deny_tools=["configure_device"], allow_tools=iter(["show_device"]))
    assert policy.decide(_call("configure_device"), WRITE)[0] == "deny"
    assert policy.decide(_call("show_device"), READ)[0] == "allow"


# ---------------------------------------------------------------- approvers


def _fake_input(answers):
    """Return an input_fn that yields the given answers and records each prompt."""
    prompts: list[str] = []
    queue = list(answers)

    def input_fn(prompt: str) -> str:
        prompts.append(prompt)
        if not queue:
            raise AssertionError("input_fn called more times than expected")
        answer = queue.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    input_fn.prompts = prompts  # type: ignore[attr-defined]
    return input_fn


def test_auto_approver_approves():
    assert AutoApprover().approve(_call("configure_device"), "r") == (True, "auto-approved")


def test_auto_denier_declines_with_note():
    approved, note = AutoDenier().approve(_call("configure_device"), "r")
    assert approved is False
    assert note == DECLINED_NOTE
    assert "declined" in note


@pytest.mark.parametrize("answer", ["y", "Y", "yes", " YES \n"])
def test_cli_approver_accepts_yes(answer):
    approver = CliApprover(input_fn=_fake_input([answer]), output_fn=lambda _s: None)
    assert approver.approve(_call("configure_device"), "needs approval") == (
        True,
        "approved by operator",
    )


@pytest.mark.parametrize("answer", ["", "n", "no", "maybe", "yess", "   "])
def test_cli_approver_declines_anything_else(answer):
    approver = CliApprover(input_fn=_fake_input([answer]), output_fn=lambda _s: None)
    approved, note = approver.approve(_call("configure_device"), "needs approval")
    assert approved is False
    assert note == DECLINED_NOTE


def test_cli_approver_declines_on_eof():
    approver = CliApprover(input_fn=_fake_input([EOFError()]), output_fn=lambda _s: None)
    assert approver.approve(_call("configure_device"), "r") == (False, DECLINED_NOTE)


def test_cli_approver_shows_tool_name_and_json_input():
    output: list[str] = []
    call = _call("configure_device", {"device": "r2", "lines": ["x"]})
    input_fn = _fake_input(["n"])
    CliApprover(input_fn=input_fn, output_fn=output.append).approve(call, "mutating")
    shown = "\n".join(output)
    assert "configure_device" in shown
    assert "mutating" in shown
    assert json.loads(output[2]) == {"device": "r2", "lines": ["x"]}
    assert input_fn.prompts == ["Approve? [y/N] "]


def test_scripted_approver_returns_decisions_in_order_and_records_calls():
    approver = ScriptedApprover([True, False])
    assert approver.approve(_call("configure_device"), "r1") == (True, "approved by script")
    assert approver.approve(_call("configure_device"), "r2") == (False, DECLINED_NOTE)
    assert [reason for _, reason in approver.calls] == ["r1", "r2"]


def test_scripted_approver_fails_when_exhausted():
    approver = ScriptedApprover([True])
    approver.approve(_call("configure_device"), "r")
    with pytest.raises(RuntimeError):
        approver.approve(_call("configure_device"), "r")


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("human", CliApprover),
        ("auto_approve", AutoApprover),
        ("auto_deny", AutoDenier),
    ],
)
def test_make_approver_maps_modes(mode, expected):
    assert isinstance(make_approver(mode), expected)


def test_make_approver_rejects_unknown_mode():
    with pytest.raises(ValueError):
        make_approver("yolo")  # type: ignore[arg-type]
