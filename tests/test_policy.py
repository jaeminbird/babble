from faultline.harness.policy import SafetyPolicy
from faultline.types import ToolCall, ToolResult, ToolSpec


def _spec(name: str, mutating: bool) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda _input: ToolResult("ok"),
        mutating=mutating,
    )


def _call(name: str) -> ToolCall:
    return ToolCall(id="t1", name=name, input={})


READ = _spec("show_device", mutating=False)
WRITE = _spec("configure_device", mutating=True)


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
