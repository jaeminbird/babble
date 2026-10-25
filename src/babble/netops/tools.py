"""The agent's tool surface for a netops lab: build_tools() returns the ToolSpecs the loop runs.

Every handler reports a lab failure as an error result the model can read, never as a raised
exception. Device names are checked against the lab first, so a typo gets the list of valid names.
"""

from __future__ import annotations

import functools
import ipaddress
from collections.abc import Callable, Sequence
from typing import Any, TypeGuard

from babble.harness.loop import SUBMIT_TOOL
from babble.lab.topology import Topology
from babble.netops.change import ChangeManager
from babble.netops.commands import validate_read_only_show_command
from babble.types import (
    FAULT_CATEGORIES,
    LabBackend,
    LabError,
    ToolResult,
    ToolSpec,
    object_schema,
    string_property,
)

Handler = Callable[[dict[str, Any]], ToolResult]


def build_tools(
    lab: LabBackend,
    topology: Topology,
    change_manager: ChangeManager,
    extra_tools: Sequence[ToolSpec] = (),
) -> list[ToolSpec]:
    """The tools for one run, in a fixed order: list_devices, show, ping, propose_change,
    rollback, then `extra_tools` (subagent and notes tools), then submit_diagnosis last.
    """
    specs = [
        ToolSpec(
            name="list_devices",
            description=(
                "List the routers and hosts in the lab with their roles, addresses and links, "
                "and the service health checks. Call it first to learn the device names."
            ),
            input_schema=object_schema({}),
            handler=_report_lab_errors_as_results(
                lambda _tool_input: ToolResult(topology.describe())
            ),
        ),
        ToolSpec(
            name="show",
            description=(
                "Run one read-only vtysh show command on one router. Append ' json' where "
                "supported (show bgp summary json, show ip route json, show ip ospf neighbor json, "
                "show interface eth1 json) and query one neighbor/prefix/interface at a time to "
                "keep output small."
            ),
            input_schema=object_schema(
                {
                    "device": string_property("Router name, e.g. r1."),
                    "command": string_property(
                        "A show command, e.g. 'show bgp neighbors 10.1.0.2 json'."
                    ),
                }
            ),
            handler=_report_lab_errors_as_results(_make_show_handler(lab, topology)),
        ),
        ToolSpec(
            name="ping",
            description=(
                "Ping an IP address from one device (a router or a host) and get the result as "
                "'N/N received' with the raw ping output. Use it to test end-to-end reachability, "
                "e.g. device ha, target 172.16.0.10 (the DC server)."
            ),
            input_schema=object_schema(
                {
                    "device": string_property("Router or host name, e.g. ha."),
                    "target": string_property("An IPv4 or IPv6 address, e.g. 172.16.0.10."),
                }
            ),
            handler=_report_lab_errors_as_results(_make_ping_handler(lab)),
        ),
        ToolSpec(
            name="propose_change",
            description=(
                "Apply config lines to one router in vtysh configuration mode, then re-run the "
                "service checks. Lines are config-mode lines only, without 'configure terminal' "
                'or \'end\', e.g. ["interface eth3", "no shutdown"]. Needs operator approval. '
                "If a check that passed before now fails, the change is rolled back "
                "automatically. Make the smallest change that fixes the diagnosed fault."
            ),
            input_schema=object_schema(
                {
                    "device": string_property("Router to change, e.g. r3."),
                    "lines": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "Config-mode lines, in order.",
                    },
                    "rationale": string_property(
                        "Why this change fixes the fault, in one sentence."
                    ),
                }
            ),
            handler=_report_lab_errors_as_results(
                _make_propose_change_handler(lab, topology, change_manager)
            ),
            mutating=True,
            parallel_safe=False,
        ),
        ToolSpec(
            name="rollback",
            description=(
                "Undo the most recent kept change on one router, restoring the config it "
                "replaced. Use it only when a change you made is wrong and must be reverted."
            ),
            input_schema=object_schema(
                {"device": string_property("Router to roll back, e.g. r3.")}
            ),
            handler=_report_lab_errors_as_results(
                _make_rollback_handler(lab, topology, change_manager)
            ),
            mutating=True,
            parallel_safe=False,
        ),
    ]
    specs.extend(extra_tools)
    specs.append(_submit_diagnosis_spec())
    return specs


# ------------------------------------------------------------------ handlers


def _make_show_handler(lab: LabBackend, topology: Topology) -> Handler:
    def handle_show_command(tool_input: dict[str, Any]) -> ToolResult:
        device = tool_input.get("device")
        if not _is_lab_device(device, lab):
            return ToolResult(_unknown_device_message(lab, device), is_error=True)
        if device in topology.hosts():
            return ToolResult(
                f"Not run: {device} is a host. show runs on routers only.", is_error=True
            )
        return run_show_command(lab, device, tool_input.get("command"))

    return handle_show_command


def run_show_command(lab: LabBackend, device: str, command: Any) -> ToolResult:
    """Run one validated read-only show command on a device. Shared by the agent's show tool and
    the subagent's, so both validate and report the same way."""
    if not isinstance(command, str):
        return ToolResult("Not run: command must be a string.", is_error=True)
    problem_message = validate_read_only_show_command(command)
    if problem_message:
        return ToolResult(
            f"Not run: {problem_message}. Example: show ip route 172.16.0.0/24 json",
            is_error=True,
        )
    try:
        return ToolResult(lab.show(device, command))
    except LabError as lab_error:
        return ToolResult(str(lab_error), is_error=True)


def _make_ping_handler(lab: LabBackend) -> Handler:
    def handle_ping_command(tool_input: dict[str, Any]) -> ToolResult:
        device = tool_input.get("device")
        if not _is_lab_device(device, lab):
            return ToolResult(_unknown_device_message(lab, device), is_error=True)
        return run_ping(lab, device, tool_input.get("target"))

    return handle_ping_command


def run_ping(lab: LabBackend, device: str, target: Any) -> ToolResult:
    """Ping an IP address from a device and report 'N/N received' with the raw output. Shared by
    the agent's ping tool and the subagent's."""
    if not isinstance(target, str):
        return ToolResult("Not run: target must be an IP address string.", is_error=True)
    try:
        ipaddress.ip_address(target)
    except ValueError:
        return ToolResult(
            f"Not run: {target!r} is not an IP address, e.g. 172.16.0.10.", is_error=True
        )
    try:
        ping_result = lab.ping(device, target)
    except LabError as lab_error:
        return ToolResult(str(lab_error), is_error=True)
    return ToolResult(f"{ping_result.received}/{ping_result.sent} received\n{ping_result.output}")


def _make_propose_change_handler(
    lab: LabBackend, topology: Topology, change_manager: ChangeManager
) -> Handler:
    def handle_propose_change(tool_input: dict[str, Any]) -> ToolResult:
        device = tool_input.get("device")
        lines = tool_input.get("lines")
        rationale = tool_input.get("rationale")
        if not _is_lab_device(device, lab):
            return ToolResult(_unknown_device_message(lab, device), is_error=True)
        if device in topology.hosts():
            return ToolResult(
                f"Not applied: {device} is a host. Changes go to routers only.", is_error=True
            )
        if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
            return ToolResult("Not applied: lines must be a list of strings.", is_error=True)
        if not isinstance(rationale, str):
            return ToolResult("Not applied: rationale must be a string.", is_error=True)
        return change_manager.apply(device, lines, rationale)

    return handle_propose_change


def _make_rollback_handler(
    lab: LabBackend, topology: Topology, change_manager: ChangeManager
) -> Handler:
    def handle_rollback(tool_input: dict[str, Any]) -> ToolResult:
        device = tool_input.get("device")
        if not _is_lab_device(device, lab):
            return ToolResult(_unknown_device_message(lab, device), is_error=True)
        if device in topology.hosts():
            return ToolResult(f"Not run: {device} is a host, so it has no changes.", is_error=True)
        return change_manager.rollback(device)

    return handle_rollback


def _report_lab_errors_as_results(handler: Handler) -> Handler:
    """Report a LabError as an error result. Other exceptions reach the loop, which reports them."""

    @functools.wraps(handler)
    def handle_and_report_lab_errors(tool_input: dict[str, Any]) -> ToolResult:
        try:
            return handler(tool_input)
        except LabError as lab_error:
            return ToolResult(str(lab_error), is_error=True)

    return handle_and_report_lab_errors


def _is_lab_device(device: object, lab: LabBackend) -> TypeGuard[str]:
    """True when `device` names a device in the lab. Narrows the caller's device to str."""
    return isinstance(device, str) and device in lab.devices()


def _unknown_device_message(lab: LabBackend, device: object) -> str:
    return f"Unknown device {device!r}. Valid devices: {', '.join(lab.devices())}"


def _submit_diagnosis_spec() -> ToolSpec:
    return ToolSpec(
        name=SUBMIT_TOOL,
        description=(
            "Record your final diagnosis and end the run. Call it once, when you have identified "
            "the fault or are confident there is none. Set fix_applied true only if an approved "
            "propose_change is in place and the service checks now pass."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "root_cause_device": string_property("The router where the fault is, e.g. r3."),
                "category": {"type": "string", "enum": list(FAULT_CATEGORIES)},
                "summary": string_property(
                    "One or two sentences: what is wrong and what you did about it."
                ),
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "The command outputs or observations that support the diagnosis."
                    ),
                },
                "fix_applied": {"type": "boolean"},
            },
            "required": [
                "root_cause_device",
                "category",
                "summary",
                "evidence",
                "fix_applied",
            ],
            "additionalProperties": False,
        },
        handler=lambda _tool_input: ToolResult("Diagnosis recorded."),
        strict=True,
    )
