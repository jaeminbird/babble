"""The agent's tool-use loop, on the raw Messages API.

History is append-only: an assistant turn is appended exactly as the API returned it, and every
tool call's result goes back in one user message. Nothing already in `messages` is edited.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from babble.harness.accounting import Accounting
from babble.harness.policy import SafetyPolicy
from babble.harness.truncate import truncate
from babble.types import (
    Approver,
    AuditSink,
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

CUT_OFF = "not run: response was cut off"
SKIPPED_AFTER_FINISH = "not run: the run already finished"

# Format with finish_tool_name=... before sending.
NUDGE = (
    "You have not finished. Call {finish_tool_name} now with your result, using the evidence "
    "you have. If you are unsure, say so."
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
    # Input of the finish call when the finish tool is not submit_diagnosis. None otherwise.
    finish_tool_input: dict[str, Any] | None = None
    # "{ExceptionType}: {message}" of the failure that ended the run. Set only when stop_reason
    # is error.
    error_message: str | None = None


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        tools: Sequence[ToolSpec],
        config: HarnessConfig,
        policy: SafetyPolicy,
        approver: Approver,
        audit: AuditSink,
        system_prompt: str,
        accounting: Accounting | None = None,
        compactor: Compactor | None = None,
        pre_hooks: Sequence[PreHook] = (),
        post_hooks: Sequence[PostHook] = (),
        # A successful call to this tool ends the run. For submit_diagnosis the input is parsed
        # into a Diagnosis; for any other tool the input is kept as RunResult.finish_tool_input.
        finish_tool_name: str = SUBMIT_TOOL,
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
        self.finish_tool_name = finish_tool_name

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
        self.system_blocks = [{"type": "text", "text": agent.system_prompt}]
        self.tool_specs_by_name = {spec.name: spec for spec in agent.tools}
        # Built once per run, in the given order, so the cached prefix stays stable.
        self.tool_definitions_for_api = [spec.to_api() for spec in agent.tools]
        self.accounting = agent.accounting if agent.accounting is not None else Accounting()
        self.turns = 0
        self.tool_calls = 0
        self.approvals = 0
        self.denials = 0
        self.nudge_sent = False
        self.run_has_finished = False
        self.diagnosis: Diagnosis | None = None
        self.finish_tool_input: dict[str, Any] | None = None
        self.last_usage = Usage()

    # ------------------------------------------------------------------ loop

    def execute(self) -> RunResult:
        harness_config = self.config
        for turn in range(1, harness_config.max_turns + 1):
            # A failure in either step ends the run with stop_reason error. It must not crash
            # the caller.
            try:
                self._maybe_compact(turn)
            except Exception as exc:  # noqa: BLE001
                return self._end_run_with_error("compaction", turn, exc)
            try:
                response = self.agent.llm.complete(
                    model=harness_config.model,
                    system=self.system_blocks,
                    messages=self.messages,
                    tools=self.tool_definitions_for_api,
                    max_tokens=harness_config.max_tokens,
                    effort=harness_config.effort,
                    cache=harness_config.prompt_caching,
                )
            except Exception as exc:  # noqa: BLE001
                return self._end_run_with_error("llm.complete", turn, exc)

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

            tool_calls_in_turn = response.tool_calls
            if tool_calls_in_turn and response.stop_reason == "max_tokens":
                # The tool_use blocks may be truncated, so none of them runs. Each still gets a
                # result, so the history stays valid for the next turn.
                self._answer_cut_off_calls(tool_calls_in_turn)
                continue
            if not tool_calls_in_turn:
                if self.nudge_sent:
                    return self._stop("no_diagnosis")
                self.nudge_sent = True
                nudge = NUDGE.format(finish_tool_name=self.agent.finish_tool_name)
                self.messages.append({"role": "user", "content": nudge})
                continue

            self._run_tool_calls_from_turn(tool_calls_in_turn)
            if self.run_has_finished:
                return self._stop("diagnosis" if self.diagnosis is not None else "finished")

        return self._stop("max_turns")

    def _maybe_compact(self, turn: int) -> None:
        compactor = self.agent.compactor
        if compactor is None or not compactor.should_compact(self.last_usage):
            return
        messages_before = len(self.messages)
        self.messages = compactor.compact(self.task, self.messages)
        self.audit.record(
            Event(
                "compaction",
                {
                    "turn": turn,
                    "messages_before": messages_before,
                    "messages_after": len(self.messages),
                },
            )
        )

    def _end_run_with_error(self, where: str, turn: int, exc: Exception) -> RunResult:
        error_message = f"{type(exc).__name__}: {exc}"
        self.audit.record(
            Event("error", {"where": where, "turn": turn, "message": error_message})
        )
        return self._stop("error", error_message=error_message)

    def _stop(self, reason: str, error_message: str | None = None) -> RunResult:
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
            finish_tool_input=self.finish_tool_input,
            error_message=error_message,
        )

    # ----------------------------------------------------------------- tools

    def _run_tool_calls_from_turn(self, tool_calls_from_turn: list[ToolCall]) -> None:
        self.tool_calls += len(tool_calls_from_turn)
        tool_specs = [self.tool_specs_by_name.get(call.name) for call in tool_calls_from_turn]
        known_tool_specs = [spec for spec in tool_specs if spec is not None]
        if len(known_tool_specs) == len(tool_calls_from_turn) and _is_parallel_safe_batch(
            known_tool_specs
        ):
            results = self._run_parallel_read_only_batch(tool_calls_from_turn, known_tool_specs)
        else:
            results = [
                self._gate_and_run_single_tool_call(call, spec)
                for call, spec in zip(tool_calls_from_turn, tool_specs, strict=True)
            ]
        self._append_tool_results(tool_calls_from_turn, results)

    def _append_tool_results(self, tool_calls: list[ToolCall], results: list[ToolResult]) -> None:
        """All results for one turn go back in one user message, in call order."""
        result_blocks = [
            self._tool_result_block(call, result)
            for call, result in zip(tool_calls, results, strict=True)
        ]
        self.messages.append({"role": "user", "content": result_blocks})

    def _gate_and_run_single_tool_call(
        self, call: ToolCall, tool_spec: ToolSpec | None
    ) -> ToolResult:
        """Gate, run and finish one call. A serial batch handles each call in full before the next
        is gated, so a later call's approval never comes before an earlier call has run."""
        self._audit_call(call)
        gated_result = self._gate_before_running(call, tool_spec)
        if gated_result is not None:
            return self._apply_post_hooks_truncate_and_audit(call, gated_result)
        assert tool_spec is not None  # _gate_before_running returns an error for unknown tools
        return self._finish_and_record_if_finish_call(
            call, tool_spec, self._invoke_tool_handler(call, tool_spec)
        )

    def _run_parallel_read_only_batch(
        self, tool_calls: list[ToolCall], tool_specs: list[ToolSpec]
    ) -> list[ToolResult]:
        """Gate every call on this thread, run the allowed handlers concurrently, then finish them
        in call order. Only read-only, parallel-safe tools reach this path, so nothing here asks
        for approval."""
        for call in tool_calls:
            self._audit_call(call)
        gate_outcomes = [
            self._gate_before_running(call, spec)
            for call, spec in zip(tool_calls, tool_specs, strict=True)
        ]
        indexes_to_run = [i for i, outcome in enumerate(gate_outcomes) if outcome is None]
        results_by_index: dict[int, ToolResult] = {}
        if indexes_to_run:
            with ThreadPoolExecutor(max_workers=len(indexes_to_run)) as pool:
                outputs = pool.map(
                    lambda i: self._invoke_tool_handler(tool_calls[i], tool_specs[i]),
                    indexes_to_run,
                )
                results_by_index = dict(zip(indexes_to_run, outputs, strict=True))
        results: list[ToolResult] = []
        for i, (call, spec) in enumerate(zip(tool_calls, tool_specs, strict=True)):
            if i in results_by_index:
                results.append(
                    self._finish_and_record_if_finish_call(call, spec, results_by_index[i])
                )
            else:
                gate_outcome = gate_outcomes[i]
                assert gate_outcome is not None
                results.append(self._apply_post_hooks_truncate_and_audit(call, gate_outcome))
        return results

    def _finish_and_record_if_finish_call(
        self, call: ToolCall, tool_spec: ToolSpec, result: ToolResult
    ) -> ToolResult:
        result = self._apply_post_hooks_truncate_and_audit(call, result)
        if tool_spec.name == self.agent.finish_tool_name and not result.is_error:
            self._record_finish_tool_call(call.input)
        return result

    def _gate_before_running(self, call: ToolCall, tool_spec: ToolSpec | None) -> ToolResult | None:
        """Return a final result if the call must not run, or None to run it."""
        if self.run_has_finished:
            return ToolResult(SKIPPED_AFTER_FINISH, is_error=True)
        if tool_spec is None:
            return ToolResult(f"Unknown tool: {call.name}", is_error=True)
        for hook in self.agent.pre_hooks:
            try:
                short_circuit = hook(call)
            except Exception as exc:  # noqa: BLE001 - a broken hook is reported, not raised
                return ToolResult(f"{type(exc).__name__}: {exc}", is_error=True)
            if short_circuit is not None:
                return short_circuit
        decision, reason = self.agent.policy.decide(call, tool_spec)
        if decision == "deny":
            self.denials += 1
            self.audit.record(Event("denied", {"id": call.id, "name": call.name, "reason": reason}))
            return ToolResult(f"Denied by policy: {reason}", is_error=True)
        if decision == "ask":
            approved, note = self._ask_approver_about_call(call, reason)
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

    def _ask_approver_about_call(self, call: ToolCall, reason: str) -> tuple[bool, str]:
        try:
            return self.agent.approver.approve(call, reason)
        except Exception as exc:  # noqa: BLE001 - an approver that fails declines the call
            return (
                False,
                f"Approval failed, so the change was not made: {type(exc).__name__}: {exc}",
            )

    def _answer_cut_off_calls(self, tool_calls: list[ToolCall]) -> None:
        self.tool_calls += len(tool_calls)
        results = []
        for call in tool_calls:
            self._audit_call(call)
            results.append(
                self._apply_post_hooks_truncate_and_audit(call, ToolResult(CUT_OFF, is_error=True))
            )
        self._append_tool_results(tool_calls, results)

    def _record_finish_tool_call(self, finish_input: dict[str, Any]) -> None:
        if self.run_has_finished:
            return
        self.run_has_finished = True
        if self.agent.finish_tool_name != SUBMIT_TOOL:
            self.finish_tool_input = finish_input
            return
        self.diagnosis = Diagnosis.from_tool_input(finish_input)
        self.audit.record(Event("diagnosis", asdict(self.diagnosis)))

    def _audit_call(self, call: ToolCall) -> None:
        self.audit.record(
            Event("tool_call", {"id": call.id, "name": call.name, "input": call.input})
        )

    def _invoke_tool_handler(self, call: ToolCall, tool_spec: ToolSpec) -> ToolResult:
        try:
            result = tool_spec.handler(call.input)
        except Exception as exc:  # noqa: BLE001 - a handler failure is reported to the model
            return ToolResult(f"{type(exc).__name__}: {exc}", is_error=True)
        if tool_spec.name == SUBMIT_TOOL and not result.is_error:
            try:
                Diagnosis.from_tool_input(call.input)
            except ValueError as problem:
                return ToolResult(
                    f"Diagnosis not recorded: {problem}. Call {SUBMIT_TOOL} again with every field.",
                    is_error=True,
                )
        return result

    def _apply_post_hooks_truncate_and_audit(
        self, call: ToolCall, result: ToolResult
    ) -> ToolResult:
        for hook in self.agent.post_hooks:
            try:
                replacement = hook(call, result)
            except Exception as exc:  # noqa: BLE001 - a broken hook is reported, not raised
                result = ToolResult(f"{type(exc).__name__}: {exc}", is_error=True)
                continue
            if replacement is not None:
                result = replacement
        truncate_limit = self.config.truncate_tool_output
        if truncate_limit is not None:
            result = ToolResult(truncate(result.content, truncate_limit), result.is_error)
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
    def _tool_result_block(call: ToolCall, result: ToolResult) -> dict[str, Any]:
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": result.content,
        }
        if result.is_error:
            block["is_error"] = True
        return block


def _is_parallel_safe_batch(tool_specs: list[ToolSpec]) -> bool:
    """More than one call, all read-only and parallel-safe: the handlers may run concurrently."""
    return len(tool_specs) > 1 and all(
        spec.parallel_safe and not spec.mutating for spec in tool_specs
    )
