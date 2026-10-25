"""Shared contracts. Every module builds against these; change them only on main."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

# --------------------------------------------------------------------------- LLM


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
        )

    @property
    def total_input(self) -> int:
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    # Raw content blocks as plain dicts. The loop appends these to history verbatim:
    # thinking blocks must go back unchanged, and history is append-only.
    content: list[dict[str, Any]]
    stop_reason: str
    usage: Usage
    model: str

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [
            ToolCall(b["id"], b["name"], b.get("input") or {})
            for b in self.content
            if b.get("type") == "tool_use"
        ]

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.content if b.get("type") == "text")


class LLMClient(Protocol):
    def complete(
        self,
        *,
        model: str,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
        effort: str | None = None,
        cache: bool = True,
    ) -> LLMResponse: ...


# ------------------------------------------------------------------------- tools


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], ToolResult]
    # Mutating tools change device state. The safety policy gates them; read-only tools run freely.
    mutating: bool = False
    # Safe to run concurrently with other parallel_safe calls in the same turn.
    parallel_safe: bool = True
    # Ask the API to guarantee schema-valid input (schema needs additionalProperties: false).
    strict: bool = False

    def to_api(self) -> dict[str, Any]:
        api = {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
        if self.strict:
            api["strict"] = True
        return api


def object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    """A strict object schema. `required` defaults to every property; additional properties are
    rejected."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties) if required is None else required,
        "additionalProperties": False,
    }


def string_property(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


# ------------------------------------------------------------------------ safety

Decision = Literal["allow", "ask"]


class Approver(Protocol):
    """Decides on a gated tool call. Returns (approved, note shown to the model)."""

    def approve(self, call: ToolCall, reason: str) -> tuple[bool, str]: ...


# --------------------------------------------------------------------------- lab


class LabError(Exception):
    """A lab operation failed (device unknown, command rejected, container down)."""


# Echo requests sent by each ping.
PING_PACKET_COUNT = 3


@dataclass
class PingResult:
    sent: int
    received: int
    output: str

    @property
    def ok(self) -> bool:
        return self.received > 0


class LabBackend(Protocol):
    """A running network the agent can inspect and change. Device names are router hostnames."""

    def devices(self) -> list[str]: ...

    def show(self, device: str, command: str) -> str:
        """Run one read-only vtysh command (e.g. 'show ip bgp summary json'). Returns stdout."""
        ...

    def configure(self, device: str, lines: list[str]) -> str:
        """Apply config lines in vtysh configuration mode. Returns any CLI output."""
        ...

    def snapshot(self, device: str) -> str:
        """Return the device's running config, for rollback."""
        ...

    def restore(self, device: str, config: str) -> None:
        """Replace the running config with a snapshot taken earlier."""
        ...

    def ping(self, device: str, target: str) -> PingResult: ...

    def host_exec(self, device: str, argv: list[str]) -> str:
        """Run a raw command in the device container. Bench-only (fault injection), never an
        agent tool."""
        ...


CheckKind = Literal["ping", "bgp_established", "route_present", "ospf_full"]


@dataclass(frozen=True)
class Check:
    """A service-level health check. The harness runs these to verify a change; the bench runs
    them to score a fix.

    ping:            device pings target (an IP)
    bgp_established: device has a BGP session in state Established with neighbor target (an IP)
    route_present:   device's routing table has a route for target (a prefix, e.g. 172.16.0.0/24)
    ospf_full:       device has an OSPF adjacency in state Full with neighbor router-id target
    """

    kind: CheckKind
    device: str
    target: str

    def describe(self) -> str:
        return f"{self.kind} {self.device} -> {self.target}"


# ------------------------------------------------------------------------- audit


@dataclass
class Event:
    # llm_call | tool_call | tool_result | approval | rollback | verify |
    # compaction | subagent | diagnosis | error | stop
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditSink(Protocol):
    """Anything that records audit events. AuditLog and TaggedAuditLog both satisfy it. Consumers
    that only record events annotate with this, not with the file-backed AuditLog."""

    def record(self, event: Event) -> None: ...


# ------------------------------------------------------------------ diagnosis

FAULT_CATEGORIES = (
    "interface_down",
    "ip_misconfig",
    "bgp_neighbor_misconfig",
    "bgp_policy",
    "ospf_misconfig",
    "static_route",
    "acl_or_filter",
    "mtu_mismatch",
    "link_degradation",
    "no_fault",
    "other",
)


@dataclass
class Diagnosis:
    root_cause_device: str
    category: str  # one of FAULT_CATEGORIES
    summary: str
    evidence: list[str] = field(default_factory=list)
    fix_applied: bool = False

    @classmethod
    def from_tool_input(cls, tool_input: dict[str, Any]) -> Diagnosis:
        """Build a Diagnosis from submit_diagnosis input. Raises ValueError naming every problem."""
        problems: list[str] = []
        for name in ("root_cause_device", "category", "summary"):
            if name not in tool_input:
                problems.append(f"missing {name}")
            elif not isinstance(tool_input[name], str):
                problems.append(f"{name} has the wrong type")
        if "evidence" not in tool_input:
            problems.append("missing evidence")
        elif not isinstance(tool_input["evidence"], list):
            problems.append("evidence has the wrong type")
        if "fix_applied" not in tool_input:
            problems.append("missing fix_applied")
        elif not isinstance(tool_input["fix_applied"], bool):
            problems.append("fix_applied has the wrong type")
        if isinstance(tool_input.get("evidence"), list) and not all(
            isinstance(item, str) for item in tool_input["evidence"]
        ):
            problems.append("evidence must be a list of strings")
        if problems:
            raise ValueError("; ".join(problems))
        return cls(
            root_cause_device=tool_input["root_cause_device"],
            category=tool_input["category"],
            summary=tool_input["summary"],
            evidence=list(tool_input["evidence"]),
            fix_applied=tool_input["fix_applied"],
        )


# --------------------------------------------------------------------- harness

ApprovalMode = Literal["human", "auto_approve", "auto_deny"]


@dataclass
class HarnessConfig:
    """Every feature flag here is something the bench can ablate."""

    model: str = "claude-opus-5-5"
    subagent_model: str = "claude-haiku-5-5"
    effort: str = "medium"
    max_turns: int = 40
    max_tokens: int = 16000

    prompt_caching: bool = True
    # Truncate each tool result to this many characters when it enters history. None = off.
    truncate_tool_output: int | None = 4000
    # Offer an inspect_device tool that runs a read-only subagent with fresh context.
    subagents: bool = True
    # Start a fresh conversation from a summary once input tokens pass this. None = off.
    compaction_threshold: int | None = 120_000
    # Give the agent a notes tool whose contents survive compaction.
    memory_notes: bool = True

    # Safety layer. With safety off, mutating tools run without approval (the unsafe baseline).
    safety: bool = True
    approval: ApprovalMode = "auto_approve"
    verify_after_change: bool = True
    auto_rollback: bool = True
