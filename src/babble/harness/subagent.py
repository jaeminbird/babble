"""inspect_device: hands one router to a child agent with a fresh, small context.

The child sees only that router's show and ping tools, plus submit_findings. It returns a compact
findings summary, so the raw command output never fills the parent's context.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from babble.harness.accounting import Accounting
from babble.harness.audit import AuditLog, TaggedAuditLog
from babble.harness.loop import Agent, RunResult
from babble.harness.policy import SafetyPolicy
from babble.harness.prompts import SUBAGENT_PROMPT
from babble.lab.topology import Topology
from babble.netops.tools import run_ping, run_show_command
from babble.types import (
    Approver,
    Event,
    HarnessConfig,
    LabBackend,
    LLMClient,
    ToolResult,
    ToolSpec,
    object_schema,
    string_property,
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
    def handle(tool_input: dict[str, Any]) -> ToolResult:
        device = tool_input.get("device")
        question = tool_input.get("question")
        if device not in topology.routers():
            return ToolResult(
                f"Not run: {device!r} is not a router. Routers: {', '.join(topology.routers())}",
                is_error=True,
            )
        if not isinstance(question, str) or not question.strip():
            return ToolResult("Not run: question must be a non-empty string.", is_error=True)

        investigation = _investigate(
            device, question, llm, lab, harness_config, accounting, audit, approver
        )
        audit.record(
            Event(
                "subagent",
                {
                    "device": device,
                    "question": question,
                    "stop_reason": investigation.stop_reason,
                    "turns": investigation.turns,
                },
            )
        )
        if investigation.finish_tool_input is None:
            return ToolResult(
                f"Subagent for {device} did not finish: stop_reason {investigation.stop_reason} "
                f"after {investigation.turns} turns.",
                is_error=True,
            )
        return ToolResult(_report(device, investigation.finish_tool_input))

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
    child_agent = Agent(
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
    return child_agent.run(f"Router: {device}\nQuestion: {question}")


def _child_tools(lab: LabBackend, device: str) -> list[ToolSpec]:
    return [
        ToolSpec(
            name="show",
            description=(
                f"Run one read-only vtysh show command on {device}. Append ' json' where "
                "supported, and query one neighbor/prefix/interface at a time."
            ),
            input_schema=object_schema(
                {"command": string_property("A show command, e.g. 'show ip route json'.")}
            ),
            handler=lambda tool_input: run_show_command(lab, device, tool_input.get("command")),
        ),
        ToolSpec(
            name="ping",
            description=f"Ping an IP address from {device} and get 'N/N received' with the output.",
            input_schema=object_schema(
                {"target": string_property("An IP address, e.g. 10.100.0.2.")}
            ),
            handler=lambda tool_input: run_ping(lab, device, tool_input.get("target")),
        ),
        ToolSpec(
            name=FINDINGS_TOOL,
            description=(
                "Record your findings and end the investigation. Call it exactly once, with a "
                "summary that answers the question and a list of facts, each with its evidence."
            ),
            input_schema=object_schema(
                {
                    "summary": string_property("A few sentences that answer the question."),
                    "facts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Facts found, each with the command and key values.",
                    },
                }
            ),
            handler=lambda _tool_input: ToolResult("Findings recorded."),
            strict=True,
        ),
    ]


def _report(device: str, findings: dict[str, Any]) -> str:
    lines = [f"Findings from {device}:", findings["summary"]]
    lines.extend(f"- {fact}" for fact in findings["facts"])
    return "\n".join(lines)
