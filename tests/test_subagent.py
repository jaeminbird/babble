from typing import Any

from babbler.harness.accounting import Accounting
from babbler.harness.audit import AuditLog
from babbler.harness.loop import SUBMIT_TOOL, Agent
from babbler.harness.policy import AutoApprover, SafetyPolicy
from babbler.harness.subagent import make_inspect_device_tool
from babbler.lab.scripted import ScriptedLab
from babbler.lab.topology import load_topology
from babbler.llm import FakeLLMClient, assert_append_only, response, text, tool_use
from babbler.netops.change import ChangeManager
from babbler.netops.tools import build_tools
from babbler.types import HarnessConfig, LLMResponse, PingResult, ToolResult, Usage

TOPO = load_topology("metro")
HAIKU = "claude-haiku-5-5"
QUESTION = "Why is the BGP session to 10.100.0.2 down?"
BGP_SUMMARY = '{"ipv4Unicast": {"peers": {"10.100.0.2": {"state": "Established"}}}}'
OSPF_NEIGHBORS = '{"neighbors": []}'


class RecordingLab(ScriptedLab):
    """ScriptedLab that also records every show and ping it answers, in call order."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.show_calls: list[tuple[str, str]] = []
        self.ping_calls: list[tuple[str, str]] = []

    def show(self, device: str, command: str) -> str:
        self.show_calls.append((device, command))
        return super().show(device, command)

    def ping(self, device: str, target: str, count: int = 3) -> PingResult:
        self.ping_calls.append((device, target))
        return super().ping(device, target, count)


def _lab() -> RecordingLab:
    return RecordingLab(
        devices=list(TOPO.devices),
        hosts=set(TOPO.hosts()),
        responses={
            ("r3", "show ip bgp summary json"): BGP_SUMMARY,
            ("r1", "show ip ospf neighbor json"): OSPF_NEIGHBORS,
        },
        pings={("r3", "10.100.0.2"): True},
    )


def _child(*blocks: dict[str, Any]) -> LLMResponse:
    """A child's response: the child runs on the subagent model."""
    return response(*blocks, model=HAIKU, usage=Usage(100, 10))


def _findings(summary: str, facts: list[str]) -> dict[str, Any]:
    return tool_use("submit_findings", {"summary": summary, "facts": facts})


def _tool(child_llm, lab=None, config=None, audit=None, accounting=None):
    lab = lab if lab is not None else _lab()
    audit = audit if audit is not None else AuditLog()
    accounting = accounting if accounting is not None else Accounting()
    config = config if config is not None else HarnessConfig()
    tool = make_inspect_device_tool(child_llm, lab, TOPO, config, accounting, audit, AutoApprover())
    return tool, lab, audit, accounting


def _invoke(tool, device: str, question: str = QUESTION) -> ToolResult:
    return tool.handler({"device": device, "question": question})


# ---------------------------------------------------------------- child scope


def test_child_runs_only_on_its_router_with_show_ping_and_submit_findings():
    child_llm = FakeLLMClient(
        [
            _child(tool_use("show", {"command": "show ip bgp summary json"}, id="c1")),
            _child(tool_use("ping", {"target": "10.100.0.2"}, id="c2")),
            _child(_findings("Session is up.", ["BGP summary shows Established"])),
        ]
    )
    tool, lab, _, _ = _tool(child_llm)

    _invoke(tool, "r3")

    assert lab.show_calls == [("r3", "show ip bgp summary json")]
    assert lab.ping_calls == [("r3", "10.100.0.2")]
    tools = child_llm.calls[0]["tools"]
    assert [t["name"] for t in tools] == ["show", "ping", "submit_findings"]
    assert tools[0]["input_schema"]["properties"].keys() == {"command"}
    assert_append_only(child_llm.calls)


def test_child_show_that_is_not_read_only_never_reaches_the_lab():
    child_llm = FakeLLMClient(
        [
            _child(tool_use("show", {"command": "show version; reload"}, id="c1")),
            _child(_findings("Could not inspect.", [])),
        ]
    )
    tool, lab, _, _ = _tool(child_llm)

    result = _invoke(tool, "r3")

    assert lab.show_calls == []
    assert result.is_error is False
    error = child_llm.calls[1]["messages"][2]["content"][0]
    assert error["is_error"] is True
    assert "forbidden character" in error["content"]
    assert_append_only(child_llm.calls)


# ------------------------------------------------------------------ findings


def test_findings_come_back_as_a_compact_summary_with_facts():
    child_llm = FakeLLMClient(
        [
            _child(tool_use("show", {"command": "show ip bgp summary json"}, id="c1")),
            _child(
                _findings(
                    "The session to 10.100.0.2 is Established, so it is not down.",
                    ["show ip bgp summary json: peer 10.100.0.2 state Established"],
                )
            ),
        ]
    )
    tool, _, _, _ = _tool(child_llm)

    result = _invoke(tool, "r3")

    assert result.is_error is False
    assert result.content == (
        "Findings from r3:\n"
        "The session to 10.100.0.2 is Established, so it is not down.\n"
        "- show ip bgp summary json: peer 10.100.0.2 state Established"
    )
    assert_append_only(child_llm.calls)


def test_child_gets_the_router_and_question_as_its_task_on_the_subagent_model():
    child_llm = FakeLLMClient([_child(_findings("Done.", []))])
    config = HarnessConfig(model="claude-opus-5-5", effort="high", max_turns=40)
    tool, _, _, _ = _tool(child_llm, config=config)

    _invoke(tool, "r1", question="Is OSPF up?")

    first = child_llm.calls[0]
    assert first["messages"] == [{"role": "user", "content": "Router: r1\nQuestion: Is OSPF up?"}]
    assert first["model"] == HAIKU
    assert first["effort"] == "low"
    assert first["max_tokens"] == config.max_tokens
    assert first["system"][0]["text"].startswith("You are a network sub-investigator.")
    assert_append_only(child_llm.calls)


def test_child_usage_lands_in_the_shared_accounting_under_the_subagent_model():
    child_llm = FakeLLMClient(
        [
            _child(tool_use("show", {"command": "show ip bgp summary json"}, id="c1")),
            _child(_findings("Done.", [])),
        ]
    )
    accounting = Accounting()
    tool, _, _, _ = _tool(child_llm, accounting=accounting)

    _invoke(tool, "r3")

    assert accounting.calls == 2
    assert accounting.by_model == {HAIKU: Usage(200, 20)}
    assert accounting.cost_usd > 0
    assert_append_only(child_llm.calls)


# --------------------------------------------------------------------- audit


def test_child_events_are_tagged_with_the_device_and_a_subagent_event_is_recorded():
    child_llm = FakeLLMClient(
        [
            _child(tool_use("show", {"command": "show ip bgp summary json"}, id="c1")),
            _child(_findings("Done.", [])),
        ]
    )
    tool, _, audit, _ = _tool(child_llm)

    _invoke(tool, "r3")

    child_events = [e for e in audit.events if e.type in ("llm_call", "tool_call", "tool_result")]
    assert child_events
    assert all(e.data["subagent"] == "r3" for e in child_events)
    (summary,) = audit.of_type("subagent")
    assert summary.data == {
        "device": "r3",
        "question": QUESTION,
        "stop_reason": "finished",
        "turns": 2,
    }
    assert_append_only(child_llm.calls)


# -------------------------------------------------------------------- errors


def test_unknown_device_is_an_error_that_lists_the_routers_and_runs_no_child():
    child_llm = FakeLLMClient([])
    tool, _, audit, _ = _tool(child_llm)

    result = _invoke(tool, "r9")

    assert result.is_error is True
    assert "'r9' is not a router" in result.content
    assert "Routers: r1, r2, r3, ra, rb, rdc" in result.content
    assert child_llm.calls == []
    assert audit.of_type("subagent") == []


def test_host_is_refused_like_an_unknown_device():
    child_llm = FakeLLMClient([])
    tool, _, _, _ = _tool(child_llm)

    result = _invoke(tool, "ha")

    assert result.is_error is True
    assert "'ha' is not a router" in result.content
    assert child_llm.calls == []


def test_child_that_never_finishes_returns_an_error_naming_its_stop_reason():
    child_llm = FakeLLMClient([_child(text("Looking.")), _child(text("Still looking."))])
    tool, _, audit, _ = _tool(child_llm)

    result = _invoke(tool, "r3")

    assert result.is_error is True
    assert result.content == (
        "Subagent for r3 did not finish: stop_reason no_diagnosis after 2 turns."
    )
    (summary,) = audit.of_type("subagent")
    assert summary.data["stop_reason"] == "no_diagnosis"
    assert summary.data["turns"] == 2
    assert_append_only(child_llm.calls)


# ---------------------------------------------------------------- integration


class _RoutedLLM:
    """Sends each child's calls to the FakeLLMClient for its router, read from the task line."""

    def __init__(self, by_router: dict[str, FakeLLMClient]) -> None:
        self.by_router = by_router

    def complete(self, **kwargs: Any) -> LLMResponse:
        router = kwargs["messages"][0]["content"].splitlines()[0].removeprefix("Router: ")
        return self.by_router[router].complete(**kwargs)


def test_parent_inspects_two_routers_in_one_turn_then_submits_a_diagnosis():
    lab = _lab()
    audit = AuditLog()
    accounting = Accounting()
    r3_llm = FakeLLMClient(
        [
            _child(tool_use("show", {"command": "show ip bgp summary json"}, id="c1")),
            _child(_findings("BGP to 10.100.0.2 is Established.", ["summary shows Established"])),
        ]
    )
    r1_llm = FakeLLMClient(
        [
            _child(tool_use("show", {"command": "show ip ospf neighbor json"}, id="c2")),
            _child(_findings("No OSPF neighbors on r1.", ["ospf neighbor json is empty"])),
        ]
    )
    inspect = make_inspect_device_tool(
        _RoutedLLM({"r3": r3_llm, "r1": r1_llm}),
        lab,
        TOPO,
        HarnessConfig(),
        accounting,
        audit,
        AutoApprover(),
    )
    config = HarnessConfig()
    changes = ChangeManager(lab, [], config, audit)
    parent_llm = FakeLLMClient(
        [
            response(
                tool_use("inspect_device", {"device": "r3", "question": QUESTION}, id="p1"),
                tool_use("inspect_device", {"device": "r1", "question": "Is OSPF up?"}, id="p2"),
            ),
            response(
                tool_use(
                    SUBMIT_TOOL,
                    {
                        "root_cause_device": "r3",
                        "category": "bgp_neighbor_misconfig",
                        "summary": "Checked both routers; nothing is wrong.",
                        "evidence": ["r3 BGP summary", "r1 OSPF neighbors"],
                        "fix_applied": False,
                    },
                    id="p3",
                )
            ),
        ]
    )
    parent = Agent(
        llm=parent_llm,
        tools=build_tools(lab, TOPO, changes, extra=[inspect]),
        config=config,
        policy=SafetyPolicy(),
        approver=AutoApprover(),
        audit=audit,
        system_prompt="You diagnose faults.",
        accounting=accounting,
    )

    result = parent.run("Alert: BGP session to 10.100.0.2 is down.")

    assert result.stop_reason == "diagnosis"
    assert result.diagnosis is not None
    assert result.diagnosis.root_cause_device == "r3"
    # Both findings reach the parent in one user message, in call order.
    blocks = parent_llm.calls[1]["messages"][2]["content"]
    assert [b["tool_use_id"] for b in blocks] == ["p1", "p2"]
    assert blocks[0]["content"].startswith("Findings from r3:\nBGP to 10.100.0.2 is Established.")
    assert blocks[1]["content"].startswith("Findings from r1:\nNo OSPF neighbors on r1.")
    assert {e.data["device"] for e in audit.of_type("subagent")} == {"r1", "r3"}
    # Parent calls and both children's calls share one accounting.
    assert accounting.calls == len(parent_llm.calls) + len(r3_llm.calls) + len(r1_llm.calls)
    for llm in (parent_llm, r3_llm, r1_llm):
        assert_append_only(llm.calls)
