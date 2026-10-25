"""Safety policy: decides whether a tool call runs directly or needs approval."""

from __future__ import annotations

import json
from collections.abc import Callable

from babble.types import ApprovalMode, Approver, Decision, ToolCall, ToolSpec


class SafetyPolicy:
    """Order of evaluation (first match wins):

    1. safety disabled         -> allow  (the unsafe baseline)
    2. spec.mutating           -> ask    (needs an approver)
    3. otherwise               -> allow  (read-only)

    Destructive config lines are refused by the ChangeManager, whatever the policy says.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def decide(self, call: ToolCall, spec: ToolSpec) -> tuple[Decision, str]:
        if not self.enabled:
            return "allow", "safety disabled"
        if spec.mutating:
            return "ask", f"mutating tool {call.name} requires approval"
        return "allow", f"read-only tool {call.name}"


# ------------------------------------------------------------------ approvers

DECLINED_NOTE = "The operator declined this change. Do not retry it."


class AutoApprover:
    """Approves every gated call. Used for the unattended bench."""

    def approve(self, call: ToolCall, reason: str) -> tuple[bool, str]:
        return True, "auto-approved"


class AutoDenier:
    """Declines every gated call."""

    def approve(self, call: ToolCall, reason: str) -> tuple[bool, str]:
        return False, DECLINED_NOTE


class CliApprover:
    """Asks a human on the terminal. input_fn and output_fn are injectable for tests."""

    def __init__(
        self,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], object] = print,
    ) -> None:
        self.input_fn = input_fn
        self.output_fn = output_fn

    def approve(self, call: ToolCall, reason: str) -> tuple[bool, str]:
        self.output_fn(f"Approval required: {reason}")
        self.output_fn(f"Tool: {call.name}")
        self.output_fn(json.dumps(call.input, indent=2, sort_keys=True))
        try:
            operator_answer = self.input_fn("Approve? [y/N] ")
        except EOFError:
            # No terminal input available: fail closed.
            return False, DECLINED_NOTE
        if operator_answer.strip().lower() in ("y", "yes"):
            return True, "approved by operator"
        return False, DECLINED_NOTE


class ScriptedApprover:
    """Returns pre-set decisions in order. Fails loudly if asked more often than scripted."""

    def __init__(self, decisions: list[bool]) -> None:
        self._scripted_approvals_remaining = list(decisions)
        self.calls: list[tuple[ToolCall, str]] = []

    def approve(self, call: ToolCall, reason: str) -> tuple[bool, str]:
        self.calls.append((call, reason))
        if not self._scripted_approvals_remaining:
            raise RuntimeError(f"ScriptedApprover has no decision left for {call.name}")
        scripted_approval = self._scripted_approvals_remaining.pop(0)
        if scripted_approval:
            return True, "approved by script"
        return False, DECLINED_NOTE


def make_approver(mode: ApprovalMode) -> Approver:
    if mode == "human":
        return CliApprover()
    if mode == "auto_approve":
        return AutoApprover()
    if mode == "auto_deny":
        return AutoDenier()
    raise ValueError(f"unknown approval mode: {mode!r}")
