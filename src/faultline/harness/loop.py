"""The agent's tool-use loop, on the raw Messages API.

History is append-only: an assistant turn is appended exactly as the API returned it, and every
tool call's result goes back in one user message. Nothing already in `messages` is edited.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from faultline.harness.accounting import Accounting
from faultline.harness.audit import AuditLog
from faultline.harness.policy import SafetyPolicy
from faultline.harness.truncate import truncate
from faultline.types import (
    Approver,
    Diagnosis,
    Event,
    HarnessConfig,
    LLMClient,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)

SUBMIT_TOOL = "submit_diagnosis"
SUBMIT_FIELDS = ("root_cause_device", "category", "summary", "evidence", "fix_applied")

CUT_OFF = "not run: response was cut off"
SKIPPED = "not run: diagnosis already submitted"

NUDGE = (
    f"You have not finished. Call {SUBMIT_TOOL} now with your diagnosis, using the evidence "
    "you have. If you are unsure, say so in the summary."
)


class Compactor(Protocol):
    """Starts a fresh conversation from a summary once the context has grown too large."""

    def should_compact(self, last_usage: Usage) -> bool: ...

    def compact(self, task: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


# A pre-hook may return a ToolResult to short-circuit the call, or None to continue.
PreHook = Callable[[ToolCall], ToolResult | None]
# A post-hook may return a replacement ToolResult, or None to keep the result.
PostHook = Callable[[ToolCall, ToolResult], ToolResult | None]


@dataclass
class RunResult:
    diagnosis: Diagnosis | None
    # diagnosis | max_turns | refusal | no_diagnosis | error
    stop_reason: str
    turns: int
    usage: Usage
    cost_usd: float
    messages: list[dict[str, Any]]
    # Tool calls the model issued across all turns (including ones not run).
    tool_calls: int
    # Approver approvals of gated calls.
    approvals: int
    # Approver declines plus calls denied by policy.
    denials: int


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        tools: Sequence[ToolSpec],
        config: HarnessConfig,
        policy: SafetyPolicy,
        approver: Approver,
        audit: AuditLog,
        system_prompt: str,
        accounting: Accounting | None = None,
        compactor: Compactor | None = None,
        pre_hooks: Sequence[PreHook] = (),
        post_hooks: Sequence[PostHook] = (),
    ) -> None:
        self.llm = llm
        self.tools = list(tools)
        self.config = config
        self.policy = policy
        self.approver = approver
        self.audit = audit
        self.system_prompt = system_prompt
        self.accounting = accounting
        self.compactor = compactor
        self.pre_hooks = tuple(pre_hooks)
        self.post_hooks = tuple(post_hooks)

    def run(self, task: str) -> RunResult:
        return _Run(self, task).execute()


class _Run:
    """State for one run. Agent stays reusable: every run gets its own."""

    def __init__(self, agent: Agent, task: str) -> None:
        self.agent = agent
        self.config = agent.config
        self.audit = agent.audit
        self.task = task
        self.messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        self.system = [{"type": "text", "text": agent.system_prompt}]
        self.specs = {spec.name: spec for spec in agent.tools}
        # Built once per run, in the given order, so the cached prefix stays stable.
        self.api_tools = [spec.to_api() for spec in agent.tools]
        self.accounting = agent.accounting if agent.accounting is not None else Accounting()
        self.turns = 0
        self.tool_calls = 0
        self.approvals = 0
        self.denials = 0
        self.nudged = False
        self.diagnosis: Diagnosis | None = None
        self.last_usage = Usage()

    # ------------------------------------------------------------------ loop

    def execute(self) -> RunResult:
        cfg = self.config
        for turn in range(1, cfg.max_turns + 1):
            try:
                self._maybe_compact(turn)
                response = self.agent.llm.complete(
                    model=cfg.model,
                    system=self.system,
                    messages=self.messages,
                    tools=self.api_tools,
                    max_tokens=cfg.max_tokens,
                    effort=cfg.effort,
                    cache=cfg.prompt_caching,
                )
            # An API failure ends the run with stop_reason error. It must not crash the caller.
            except Exception as exc:  # noqa: BLE001
                self.audit.record(
                    Event(
                        "error",
                        {
                            "where": "llm.complete",
                            "turn": turn,
                            "message": f"{type(exc).__name__}: {exc}",
                        },
                    )
                )
                return self._stop("error")

            self.turns = turn
            self.accounting.add(response.usage, response.model)
            self.last_usage = response.usage
            self.audit.record(
                Event(
                    "llm_call",
                    {
                        "turn": turn,
                        "model": response.model,
                        "stop_reason": response.stop_reason,
                        "usage": asdict(response.usage),
                    },
                )
            )
            # The assistant turn goes back exactly as the API returned it.
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "refusal":
                return self._stop("refusal")

            calls = response.tool_calls
            if calls and response.stop_reason == "max_tokens":
                # The tool_use blocks may be truncated, so none of them runs. Each still gets a
                # result, so the history stays valid for the next turn.
                self._answer_cut_off(calls)
                continue
            if not calls:
                if self.nudged:
                    return self._stop("no_diagnosis")
                self.nudged = True
                self.messages.append({"role": "user", "content": NUDGE})
                continue

            self._run_calls(calls)
            if self.diagnosis is not None:
                return self._stop("diagnosis")

        return self._stop("max_turns")

    def _maybe_compact(self, turn: int) -> None:
        compactor = self.agent.compactor
        if compactor is None or not compactor.should_compact(self.last_usage):
            return
        before = len(self.messages)
        self.messages = compactor.compact(self.task, self.messages)
        self.audit.record(
            Event(
                "compaction",
                {"turn": turn, "messages_before": before, "messages_after": len(self.messages)},
            )
        )

    def _stop(self, reason: str) -> RunResult:
        self.audit.record(Event("stop", {"reason": reason, "turns": self.turns}))
        return RunResult(
            diagnosis=self.diagnosis,
            stop_reason=reason,
            turns=self.turns,
            usage=self.accounting.usage,
            cost_usd=self.accounting.cost_usd,
            messages=self.messages,
            tool_calls=self.tool_calls,
            approvals=self.approvals,
            denials=self.denials,
        )

    # ----------------------------------------------------------------- tools

    def _run_calls(self, calls: list[ToolCall]) -> None:
        self.tool_calls += len(calls)
        specs = [self.specs.get(call.name) for call in calls]
        known = [spec for spec in specs if spec is not None]
        if len(known) == len(calls) and _is_parallel_batch(known):
            results = self._run_parallel(calls, known)
        else:
            results = [self._run_one(call, spec) for call, spec in zip(calls, specs, strict=True)]
        # All results for one turn go back in one user message, in call order.
        blocks = [
            self._result_block(call, result) for call, result in zip(calls, results, strict=True)
        ]
        self.messages.append({"role": "user", "content": blocks})

    def _run_one(self, call: ToolCall, spec: ToolSpec | None) -> ToolResult:
        """Gate, run and finish one call. A serial batch handles each call in full before the next
        is gated, so a later call's approval never comes before an earlier call has run."""
        self._audit_call(call)
        gated = self._gate(call, spec)
        if gated is not None:
            return self._finish(call, gated)
        assert spec is not None  # _gate returns an error for unknown tools
        return self._complete(call, spec, self._execute(call, spec))

    def _run_parallel(self, calls: list[ToolCall], specs: list[ToolSpec]) -> list[ToolResult]:
        """Gate every call on this thread, run the allowed handlers concurrently, then finish them
        in call order. Only read-only, parallel-safe tools reach this path, so nothing here asks
        for approval."""
        for call in calls:
            self._audit_call(call)
        gated = [self._gate(call, spec) for call, spec in zip(calls, specs, strict=True)]
        runnable = [i for i, outcome in enumerate(gated) if outcome is None]
        ran: dict[int, ToolResult] = {}
        if runnable:
            with ThreadPoolExecutor(max_workers=len(runnable)) as pool:
                outputs = pool.map(lambda i: self._execute(calls[i], specs[i]), runnable)
                ran = dict(zip(runnable, outputs, strict=True))
        results: list[ToolResult] = []
        for i, (call, spec) in enumerate(zip(calls, specs, strict=True)):
            if i in ran:
                results.append(self._complete(call, spec, ran[i]))
            else:
                outcome = gated[i]
                assert outcome is not None
                results.append(self._finish(call, outcome))
        return results

    def _complete(self, call: ToolCall, spec: ToolSpec, result: ToolResult) -> ToolResult:
        result = self._finish(call, result)
        if spec.name == SUBMIT_TOOL and not result.is_error:
            self._record_diagnosis(call.input)
        return result

    def _gate(self, call: ToolCall, spec: ToolSpec | None) -> ToolResult | None:
        """Return a final result if the call must not run, or None to run it."""
        if self.diagnosis is not None:
            return ToolResult(SKIPPED, is_error=True)
        if spec is None:
            return ToolResult(f"Unknown tool: {call.name}", is_error=True)
        for hook in self.agent.pre_hooks:
            try:
                short_circuit = hook(call)
            except Exception as exc:  # noqa: BLE001 - a broken hook is reported, not raised
                return ToolResult(f"{type(exc).__name__}: {exc}", is_error=True)
            if short_circuit is not None:
                return short_circuit
        decision, reason = self.agent.policy.decide(call, spec)
        if decision == "deny":
            self.denials += 1
            self.audit.record(Event("denied", {"id": call.id, "name": call.name, "reason": reason}))
            return ToolResult(f"Denied by policy: {reason}", is_error=True)
        if decision == "ask":
            approved, note = self._ask(call, reason)
            self.audit.record(
                Event(
                    "approval",
                    {"id": call.id, "name": call.name, "approved": approved, "note": note},
                )
            )
            if approved:
                self.approvals += 1
                return None
            self.denials += 1
            return ToolResult(note, is_error=True)
        return None

    def _ask(self, call: ToolCall, reason: str) -> tuple[bool, str]:
        try:
            return self.agent.approver.approve(call, reason)
        except Exception as exc:  # noqa: BLE001 - an approver that fails declines the call
            return (
                False,
                f"Approval failed, so the change was not made: {type(exc).__name__}: {exc}",
            )

    def _answer_cut_off(self, calls: list[ToolCall]) -> None:
        self.tool_calls += len(calls)
        blocks = []
        for call in calls:
            self._audit_call(call)
            result = self._finish(call, ToolResult(CUT_OFF, is_error=True))
            blocks.append(self._result_block(call, result))
        self.messages.append({"role": "user", "content": blocks})

    def _record_diagnosis(self, data: dict[str, Any]) -> None:
        if self.diagnosis is not None:
            return
        self.diagnosis = Diagnosis(
            root_cause_device=data["root_cause_device"],
            category=data["category"],
            summary=data["summary"],
            evidence=list(data["evidence"]),
            fix_applied=bool(data["fix_applied"]),
        )
        self.audit.record(Event("diagnosis", asdict(self.diagnosis)))

    def _audit_call(self, call: ToolCall) -> None:
        self.audit.record(
            Event("tool_call", {"id": call.id, "name": call.name, "input": call.input})
        )

    def _execute(self, call: ToolCall, spec: ToolSpec) -> ToolResult:
        try:
            result = spec.handler(call.input)
        except Exception as exc:  # noqa: BLE001 - a handler failure is reported to the model
            return ToolResult(f"{type(exc).__name__}: {exc}", is_error=True)
        if spec.name == SUBMIT_TOOL and not result.is_error:
            problems = _submit_problems(call.input)
            if problems:
                return ToolResult(
                    "Diagnosis not recorded: "
                    + "; ".join(problems)
                    + f". Call {SUBMIT_TOOL} again with every field.",
                    is_error=True,
                )
        return result

    def _finish(self, call: ToolCall, result: ToolResult) -> ToolResult:
        for hook in self.agent.post_hooks:
            try:
                replacement = hook(call, result)
            except Exception as exc:  # noqa: BLE001 - a broken hook is reported, not raised
                result = ToolResult(f"{type(exc).__name__}: {exc}", is_error=True)
                continue
            if replacement is not None:
                result = replacement
        limit = self.config.truncate_tool_output
        if limit is not None:
            result = ToolResult(truncate(result.content, limit), result.is_error)
        self.audit.record(
            Event(
                "tool_result",
                {
                    "id": call.id,
                    "name": call.name,
                    "is_error": result.is_error,
                    "chars": len(result.content),
                },
            )
        )
        return result

    @staticmethod
    def _result_block(call: ToolCall, result: ToolResult) -> dict[str, Any]:
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": result.content,
        }
        if result.is_error:
            block["is_error"] = True
        return block


def _is_parallel_batch(specs: list[ToolSpec]) -> bool:
    """More than one call, all read-only and parallel-safe: the handlers may run concurrently."""
    return len(specs) > 1 and all(s.parallel_safe and not s.mutating for s in specs)


_SUBMIT_TYPES: dict[str, type | tuple[type, ...]] = {
    "root_cause_device": str,
    "category": str,
    "summary": str,
    "evidence": list,
    "fix_applied": bool,
}


def _submit_problems(data: dict[str, Any]) -> list[str]:
    """Why a submit_diagnosis input cannot become a Diagnosis. Empty when it can."""
    problems = []
    for field in SUBMIT_FIELDS:
        if field not in data:
            problems.append(f"missing {field}")
        elif not isinstance(data[field], _SUBMIT_TYPES[field]):
            problems.append(f"{field} has the wrong type")
    evidence = data.get("evidence")
    if isinstance(evidence, list) and not all(isinstance(item, str) for item in evidence):
        problems.append("evidence must be a list of strings")
    return problems
