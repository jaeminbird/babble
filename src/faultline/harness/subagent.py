"""inspect_device: hands one router to a child agent with a fresh, small context.

The child sees only that router's show and ping tools, plus submit_findings. It returns a compact
findings summary, so the raw command output never fills the parent's context.
"""

from __future__ import annotations

import dataclasses
import ipaddress
from typing import Any

from faultline.harness.accounting import Accounting
from faultline.harness.audit import AuditLog, TaggedAuditLog
from faultline.harness.loop import Agent, RunResult
from faultline.harness.policy import SafetyPolicy
from faultline.harness.prompts import SUBAGENT_PROMPT
from faultline.lab.topology import Topology
from faultline.netops.commands import validate_show
from faultline.types import (
    Approver,
    Event,
    HarnessConfig,
    LabBackend,
    LabError,
    LLMClient,
    ToolResult,
    ToolSpec,
)

FINDINGS_TOOL = "submit_findings"


def make_inspect_device_tool(
    llm: LLMClient,
    lab: LabBackend,
    topology: Topology,
    harness_config: HarnessConfig,
    accounting: Accounting,
    audit: AuditLog,
    approver: Approver,
) -> ToolSpec:
    def handle(inp: dict[str, Any]) -> ToolResult:
        device = inp.get("device")
        question = inp.get("question")
        if device not in topology.routers():
            return ToolResult(
                f"Not run: {device!r} is not a router. Routers: {', '.join(topology.routers())}",
                is_error=True,
            )
        if not isinstance(question, str) or not question.strip():
            return ToolResult("Not run: question must be a non-empty string.", is_error=True)

        run = _investigate(device, question, llm, lab, harness_config, accounting, audit, approver)
        audit.record(
            Event(
                "subagent",
                {
                    "device": device,
                    "question": question,
                    "stop_reason": run.stop_reason,
                    "turns": run.turns,
                },
            )
        )
        if run.finish_tool_input is None:
            return ToolResult(
                f"Subagent for {device} did not finish: stop_reason {run.stop_reason} "
                f"after {run.turns} turns.",
                is_error=True,
            )
        return ToolResult(_report(device, run.finish_tool_input))

    return ToolSpec(
        name="inspect_device",
        description=(
            "Hand one router to a sub-investigator with a fresh context. Ask one specific "
            "question about that router, e.g. 'Why is the BGP session to 10.100.0.2 down?'. "
            "It inspects only that router, read-only, and returns a findings summary with the "
            "evidence. Call several in parallel for different routers."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "device": {"type": "string", "description": "Router name, e.g. r3."},
                "question": {
                    "type": "string",
                    "description": "One specific question about that router.",
                },
            },
            "required": ["device", "question"],
            "additionalProperties": False,
        },
        handler=handle,
        parallel_safe=True,
    )


def _investigate(
    device: str,
    question: str,
    llm: LLMClient,
    lab: LabBackend,
    harness_config: HarnessConfig,
    accounting: Accounting,
    audit: AuditLog,
    approver: Approver,
) -> RunResult:
    child_config = dataclasses.replace(
        harness_config,
        model=harness_config.subagent_model,
        effort="low",
        max_turns=12,
        subagents=False,
        compaction_threshold=None,
        memory_notes=False,
    )
    child = Agent(
        llm=llm,
        tools=_child_tools(lab, device),
        config=child_config,
        policy=SafetyPolicy(enabled=harness_config.safety),
        approver=approver,
        audit=TaggedAuditLog(audit, subagent=device),
        system_prompt=SUBAGENT_PROMPT,
        accounting=accounting,
        finish_tool_name=FINDINGS_TOOL,
    )
    return child.run(f"Router: {device}\nQuestion: {question}")


def _child_tools(lab: LabBackend, device: str) -> list[ToolSpec]:
    return [
        ToolSpec(
            name="show",
            description=(
                f"Run one read-only vtysh show command on {device}. Append ' json' where "
                "supported, and query one neighbor/prefix/interface at a time."
            ),
            input_schema=_schema(
                {"command": _string("A show command, e.g. 'show ip route json'.")}
            ),
            handler=_show(lab, device),
        ),
        ToolSpec(
            name="ping",
            description=f"Ping an IP address from {device} and get 'N/N received' with the output.",
            input_schema=_schema({"target": _string("An IP address, e.g. 10.100.0.2.")}),
            handler=_ping(lab, device),
        ),
        ToolSpec(
            name=FINDINGS_TOOL,
            description=(
                "Record your findings and end the investigation. Call it exactly once, with a "
                "summary that answers the question and a list of facts, each with its evidence."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "summary": _string("A few sentences that answer the question."),
                    "facts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Facts found, each with the command and key values.",
                    },
                },
                "required": ["summary", "facts"],
                "additionalProperties": False,
            },
            handler=lambda _inp: ToolResult("Findings recorded."),
            strict=True,
        ),
    ]


def _show(lab: LabBackend, device: str):
    def handle(inp: dict[str, Any]) -> ToolResult:
        command = inp.get("command")
        if not isinstance(command, str):
            return ToolResult("Not run: command must be a string.", is_error=True)
        problem = validate_show(command)
        if problem:
            return ToolResult(
                f"Not run: {problem}. Example: show ip route 172.16.0.0/24 json", is_error=True
            )
        try:
            return ToolResult(lab.show(device, command))
        except LabError as exc:
            return ToolResult(str(exc), is_error=True)

    return handle


def _ping(lab: LabBackend, device: str):
    def handle(inp: dict[str, Any]) -> ToolResult:
        target = inp.get("target")
        if not isinstance(target, str):
            return ToolResult("Not run: target must be an IP address string.", is_error=True)
        try:
            ipaddress.ip_address(target)
        except ValueError:
            return ToolResult(f"Not run: {target!r} is not an IP address.", is_error=True)
        try:
            result = lab.ping(device, target)
        except LabError as exc:
            return ToolResult(str(exc), is_error=True)
        return ToolResult(f"{result.received}/{result.sent} received\n{result.output}")

    return handle


def _report(device: str, findings: dict[str, Any]) -> str:
    lines = [f"Findings from {device}:", findings["summary"]]
    lines.extend(f"- {fact}" for fact in findings["facts"])
    return "\n".join(lines)


def _schema(properties: dict[str, Any]) -> dict[str, Any]:
    """Every property is required; additional properties are rejected."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _string(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}
