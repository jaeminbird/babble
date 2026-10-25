import pytest

from faultline.harness.audit import AuditLog
from faultline.harness.loop import SUBMIT_TOOL, Agent
from faultline.harness.policy import DECLINED_NOTE, SafetyPolicy, ScriptedApprover
from faultline.lab.scripted import ScriptedLab
from faultline.lab.topology import load_topology
from faultline.llm import FakeLLMClient, assert_append_only, response, tool_use
from faultline.netops.change import ChangeManager
from faultline.netops.tools import build_tools
from faultline.types import FAULT_CATEGORIES, Check, HarnessConfig, LabError, ToolResult, ToolSpec

TOPO = load_topology("metro")
FIX = ["interface eth3", "no shutdown"]
IF_DOWN = '{"ifName": "eth3", "administrativeStatus": "down"}'
IF_UP = '{"ifName": "eth3", "administrativeStatus": "up"}'
DC_PING = Check("ping", "ha", "172.16.0.10")
ORDER = ["list_devices", "show", "ping", "propose_change", "rollback", SUBMIT_TOOL]


def make_lab(lab_cls=ScriptedLab, **overrides):
    kwargs = {
        "devices": list(TOPO.devices),
        "hosts": set(TOPO.hosts()),
        "responses": {("r3", "show interface eth3 json"): IF_DOWN, ("r3", "show version"): "FRR"},
        "fixed_responses": {("r3", "show interface eth3 json"): IF_UP},
        "fix": {"r3": FIX},
        "pings": {("ha", "172.16.0.10"): False},
        "fixed_pings": {("ha", "172.16.0.10"): True},
    }
    kwargs.update(overrides)
    return lab_cls(**kwargs)


def make(lab_cls=ScriptedLab, checks=(DC_PING,), extra=(), config=None, **overrides):
    lab = make_lab(lab_cls, **overrides)
    audit = AuditLog()
    config = config or HarnessConfig()
    changes = ChangeManager(lab, list(checks), config, audit)
    tools = build_tools(lab, TOPO, changes, config, extra=extra)
    return {t.name: t for t in tools}, lab, audit, changes


def call(tools, name, **inp) -> ToolResult:
    return tools[name].handler(inp)


# ------------------------------------------------------------------ surface


def test_tools_are_in_fixed_order_with_submit_last():
    tools, *_ = make()
    assert list(tools) == ORDER


def test_extra_tools_go_before_submit():
    extra = [
        ToolSpec(name="inspect_device", description="d", input_schema={}, handler=lambda _: None),
        ToolSpec(name="notes", description="d", input_schema={}, handler=lambda _: None),
    ]
    tools, *_ = make(extra=extra)
    assert list(tools) == [
        "list_devices",
        "show",
        "ping",
        "propose_change",
        "rollback",
        "inspect_device",
        "notes",
        SUBMIT_TOOL,
    ]


@pytest.mark.parametrize("name", ORDER)
def test_every_schema_is_a_closed_object_with_all_properties_required(name):
    tools, *_ = make()
    schema = tools[name].input_schema
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


@pytest.mark.parametrize("name", ORDER)
def test_every_tool_has_a_description(name):
    tools, *_ = make()
    assert len(tools[name].description) > 40


def test_submit_schema_is_strict_and_uses_the_fault_categories():
    tools, *_ = make()
    submit = tools[SUBMIT_TOOL]
    props = submit.input_schema["properties"]

    assert submit.strict is True
    assert submit.input_schema["additionalProperties"] is False
    assert submit.input_schema["required"] == [
        "root_cause_device",
        "category",
        "summary",
        "evidence",
        "fix_applied",
    ]
    assert props["category"] == {"type": "string", "enum": list(FAULT_CATEGORIES)}
    assert props["evidence"]["type"] == "array"
    assert props["evidence"]["items"] == {"type": "string"}
    assert props["fix_applied"]["type"] == "boolean"


def test_submit_handler_records_the_diagnosis():
    tools, *_ = make()
    assert tools[SUBMIT_TOOL].handler({}).content == "Diagnosis recorded."


def test_mutating_and_parallel_flags():
    tools, *_ = make()
    assert tools["propose_change"].mutating is True
    assert tools["propose_change"].parallel_safe is False
    assert tools["rollback"].mutating is False
    assert tools["rollback"].parallel_safe is False
    for name in ("list_devices", "show", "ping"):
        assert tools[name].mutating is False
        assert tools[name].parallel_safe is True


# -------------------------------------------------------------- list_devices


def test_list_devices_describes_the_topology():
    tools, *_ = make()
    result = call(tools, "list_devices")
    assert not result.is_error
    assert result.content == TOPO.describe()
    assert "r3 (" in result.content


# ---------------------------------------------------------------------- show


def test_show_returns_the_lab_output():
    tools, *_ = make()
    result = call(tools, "show", device="r3", command="show interface eth3 json")
    assert not result.is_error
    assert result.content == IF_DOWN


def test_show_rejects_a_bad_command_without_running_it():
    tools, *_ = make()
    result = call(tools, "show", device="r3", command="configure terminal")
    assert result.is_error
    assert "must start with 'show '" in result.content
    assert "Example:" in result.content


def test_show_rejects_a_host():
    tools, *_ = make()
    result = call(tools, "show", device="ha", command="show version")
    assert result.is_error
    assert "ha is a host" in result.content


def test_show_lists_valid_devices_for_an_unknown_one():
    tools, *_ = make()
    result = call(tools, "show", device="r99", command="show version")
    assert result.is_error
    assert result.content.startswith("Unknown device 'r99'. Valid devices: ")
    assert "r1" in result.content and "hdc" in result.content


def test_show_turns_a_lab_error_into_an_error_result():
    class Failing(ScriptedLab):
        def show(self, device, command):
            raise LabError("vtysh is not responding")

    tools, *_ = make(lab_cls=Failing)
    result = call(tools, "show", device="r3", command="show version")
    assert result == ToolResult("vtysh is not responding", is_error=True)


# ---------------------------------------------------------------------- ping


def test_ping_reports_received_over_sent_and_the_raw_output():
    tools, *_ = make(pings={("ha", "172.16.0.10"): True})
    result = call(tools, "ping", device="ha", target="172.16.0.10")
    assert not result.is_error
    assert result.content == "3/3 received\n3 packets transmitted, 3 received"


def test_ping_from_a_host_with_no_reply():
    tools, *_ = make()
    result = call(tools, "ping", device="ha", target="172.16.0.10")
    assert result.content.startswith("0/3 received\n")


def test_ping_from_a_router_works():
    tools, *_ = make(pings={("r1", "10.0.13.2"): True})
    assert call(tools, "ping", device="r1", target="10.0.13.2").content.startswith("3/3")


@pytest.mark.parametrize("target", ["not-an-ip", "172.16.0.10; reboot", ""])
def test_ping_rejects_a_non_address(target):
    tools, *_ = make()
    result = call(tools, "ping", device="ha", target=target)
    assert result.is_error
    assert "is not an IP address" in result.content


def test_ping_lists_valid_devices_for_an_unknown_one():
    tools, *_ = make()
    result = call(tools, "ping", device="nope", target="172.16.0.10")
    assert result.is_error
    assert "Valid devices:" in result.content


# ------------------------------------------------------------ propose_change


def test_propose_change_applies_through_the_change_manager():
    tools, lab, audit, _ = make()
    result = call(tools, "propose_change", device="r3", lines=FIX, rationale="eth3 is down")

    assert not result.is_error
    assert lab.applied_lines("r3") == FIX
    assert lab.state == "fixed"
    assert "Service checks: 1/1 passing." in result.content
    assert audit.of_type("change_applied")


def test_propose_change_rejects_lines_that_are_not_strings():
    tools, lab, *_ = make()
    result = call(tools, "propose_change", device="r3", lines="interface eth3", rationale="x")
    assert result.is_error
    assert "lines must be a list of strings" in result.content
    assert lab.config_calls == []


def test_propose_change_rejects_empty_lines():
    tools, lab, *_ = make()
    result = call(tools, "propose_change", device="r3", lines=[], rationale="x")
    assert result.is_error
    assert lab.config_calls == []


def test_propose_change_rejects_a_host_and_an_unknown_device():
    tools, lab, *_ = make()
    assert call(tools, "propose_change", device="ha", lines=FIX, rationale="x").is_error
    result = call(tools, "propose_change", device="nope", lines=FIX, rationale="x")
    assert result.is_error
    assert "Valid devices:" in result.content
    assert lab.config_calls == []


def test_propose_change_blocked_line_is_an_error_and_nothing_applies():
    tools, lab, *_ = make()
    result = call(tools, "propose_change", device="r3", lines=["hostname evil"], rationale="x")
    assert result.is_error
    assert "blocked config line" in result.content
    assert lab.config_calls == []


# ----------------------------------------------------------------- rollback


def test_rollback_tool_undoes_the_last_kept_change():
    tools, lab, *_ = make()
    call(tools, "propose_change", device="r3", lines=FIX, rationale="eth3 is down")
    result = call(tools, "rollback", device="r3")

    assert not result.is_error
    assert lab.applied_lines("r3") == []
    assert lab.state == "broken"


def test_rollback_tool_without_a_change_is_an_error():
    tools, *_ = make()
    result = call(tools, "rollback", device="r3")
    assert result.is_error
    assert "No kept change on r3" in result.content


def test_rollback_tool_refuses_a_host():
    tools, *_ = make()
    result = call(tools, "rollback", device="ha")
    assert result.is_error
    assert "has no changes" in result.content


def test_unknown_device_for_rollback_lists_valid_devices():
    tools, *_ = make()
    result = call(tools, "rollback", device="zz")
    assert result.is_error
    assert "Valid devices:" in result.content


# ------------------------------------------------------------- end to end

SYSTEM = "You diagnose faults in the metro lab."
TASK = "ha cannot reach the DC server 172.16.0.10. Diagnose and fix it."
SUBMIT_FIXED = {
    "root_cause_device": "r3",
    "category": "interface_down",
    "summary": "eth3 on r3 is administratively down; re-enabled it.",
    "evidence": ["show interface eth3 json reports administrativeStatus down"],
    "fix_applied": True,
}
SUBMIT_UNFIXED = {
    **SUBMIT_FIXED,
    "summary": "eth3 is down and the change was declined.",
    "fix_applied": False,
}
PROPOSE = {"device": "r3", "lines": FIX, "rationale": "eth3 is administratively down"}


def _run(lab, approver, final_submit):
    audit = AuditLog()
    config = HarnessConfig()
    changes = ChangeManager(lab, [DC_PING], config, audit)
    tools = build_tools(lab, TOPO, changes, config)
    llm = FakeLLMClient(
        [
            response(tool_use("show", {"device": "r3", "command": "show interface eth3 json"})),
            response(tool_use("propose_change", PROPOSE)),
            response(tool_use(SUBMIT_TOOL, final_submit)),
        ]
    )
    agent = Agent(
        llm=llm,
        tools=tools,
        config=config,
        policy=SafetyPolicy(),
        approver=approver,
        audit=audit,
        system_prompt=SYSTEM,
    )
    return agent.run(TASK), llm, audit


def test_agent_diagnoses_and_fixes_the_lab_through_the_tools():
    lab = make_lab()
    approver = ScriptedApprover([True])
    result, llm, audit = _run(lab, approver, SUBMIT_FIXED)

    assert result.stop_reason == "diagnosis"
    assert result.diagnosis is not None
    assert result.diagnosis.root_cause_device == "r3"
    assert result.diagnosis.fix_applied is True
    assert lab.state == "fixed"
    assert lab.applied_lines("r3") == FIX

    assert len(approver.calls) == 1
    gated_call, _reason = approver.calls[0]
    assert gated_call.name == "propose_change"
    assert gated_call.input == PROPOSE
    assert result.approvals == 1
    assert result.denials == 0
    assert [e.data["device"] for e in audit.of_type("change_applied")] == ["r3"]
    assert_append_only(llm.calls)


def test_agent_stays_broken_when_the_operator_declines_the_change():
    lab = make_lab()
    approver = ScriptedApprover([False])
    result, llm, audit = _run(lab, approver, SUBMIT_UNFIXED)

    assert result.stop_reason == "diagnosis"
    assert result.diagnosis is not None
    assert result.diagnosis.fix_applied is False
    assert lab.state == "broken"
    assert lab.config_calls == []
    assert result.approvals == 0
    assert result.denials == 1
    assert audit.of_type("change_applied") == []

    # The model saw the decline as an error result on its propose_change call.
    declined = llm.calls[2]["messages"][-1]["content"][0]
    assert declined["content"] == DECLINED_NOTE
    assert declined["is_error"] is True
    assert_append_only(llm.calls)
