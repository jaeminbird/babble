"""Safety policy: decides whether a tool call runs, needs approval, or is denied."""

from __future__ import annotations

from collections.abc import Iterable

from faultline.types import Decision, ToolCall, ToolSpec


class SafetyPolicy:
    """Order of evaluation (first match wins):

    1. tool in deny_tools      -> deny   (applies even when safety is disabled)
    2. safety disabled         -> allow  (the unsafe baseline)
    3. tool in allow_tools     -> allow
    4. spec.mutating           -> ask    (needs an approver)
    5. otherwise               -> allow  (read-only)
    """

    def __init__(
        self,
        enabled: bool = True,
        deny_tools: Iterable[str] = (),
        allow_tools: Iterable[str] = (),
    ) -> None:
        self.enabled = enabled
        self.deny_tools = frozenset(deny_tools)
        self.allow_tools = frozenset(allow_tools)

    def decide(self, call: ToolCall, spec: ToolSpec) -> tuple[Decision, str]:
        if call.name in self.deny_tools:
            return "deny", f"tool {call.name} is denied by policy"
        if not self.enabled:
            return "allow", "safety disabled"
        if call.name in self.allow_tools:
            return "allow", f"tool {call.name} is pre-approved by policy"
        if spec.mutating:
            return "ask", f"mutating tool {call.name} requires approval"
        return "allow", f"read-only tool {call.name}"
