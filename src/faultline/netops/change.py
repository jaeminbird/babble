"""ChangeManager: snapshot, apply, verify and roll back config changes on a LabBackend.

Every change follows the same path:

1. Blocked config lines are rejected before anything is touched.
2. The device is snapshotted, and the service checks are evaluated (if verification is on).
3. The lines are applied. A CLI error restores the snapshot.
4. The checks are evaluated again once routing has had time to converge (see
   `convergence_wait_seconds`). A check that passed before and fails now is a regression.
   With auto_rollback on, a regression restores the snapshot. Otherwise the change is kept and
   the regression is reported.

Kept changes push their snapshot onto a per-device undo stack, so `rollback` can undo them one
at a time, newest first.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from faultline.harness.audit import AuditLog
from faultline.netops.checks import evaluate_all
from faultline.netops.commands import blocked_config
from faultline.types import Check, Event, HarnessConfig, LabBackend, LabError, ToolResult

# Even when every check passes early, keep watching this long: a bad change can take a few
# seconds to tear down a session.
MINIMUM_OBSERVATION_SECONDS = 10.0

# Result texts are read by the model, so long lists are cut to this many entries.
LIST_LIMIT = 8


def summarize(items: Sequence[str]) -> str:
    """Join up to LIST_LIMIT items, then say how many were left out."""
    shown = ", ".join(items[:LIST_LIMIT])
    extra = len(items) - LIST_LIMIT
    return f"{shown}, and {extra} more" if extra > 0 else shown


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


class ChangeManager:
    def __init__(
        self,
        lab: LabBackend,
        checks: Sequence[Check],
        config: HarnessConfig,
        audit: AuditLog,
        convergence_wait_seconds: float = 0.0,
        poll_interval_seconds: float = 3.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """convergence_wait_seconds: how long to let BGP/OSPF settle after a change before
        judging it. 0 judges immediately (the in-memory test lab converges instantly); a live
        lab needs around 30."""
        self.lab = lab
        self.convergence_wait_seconds = convergence_wait_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.sleep = sleep
        self.checks = list(dict.fromkeys(checks))  # duplicates would be counted twice
        self.config = config
        self.audit = audit
        # One dict per applied change: {device, lines, rationale, rolled_back, regressions}.
        self.changes: list[dict[str, Any]] = []
        # Per device, (snapshot taken before a kept change, index into self.changes), oldest first.
        self._undo: dict[str, list[tuple[str, int]]] = {}

    # ----------------------------------------------------------------- apply

    def apply(self, device: str, lines: list[str], rationale: str) -> ToolResult:
        lines = list(lines)
        if not lines:
            return ToolResult("Change not applied: no config lines were given.", is_error=True)

        reasons = blocked_config(lines)
        if reasons:
            self._reject(device, reasons)
            return ToolResult(
                f"Change on {device} was not applied: {'; '.join(reasons)}. Nothing was changed.",
                is_error=True,
            )

        try:
            snapshot = self.lab.snapshot(device)
        except LabError as exc:
            return ToolResult(f"Change on {device} was not applied: {exc}", is_error=True)

        verify = self.config.verify_after_change
        before = evaluate_all(self.checks, self.lab) if verify else {}

        try:
            self.lab.configure(device, lines)
        except LabError as exc:
            # The device may have taken some of the lines before failing, so put it back.
            restore_error = self._restore(device, snapshot)
            self._reject(device, [f"CLI error: {exc}"])
            text = f"Change on {device} was not applied: {exc}."
            if restore_error is None:
                text += " The device config was restored to its previous state."
            else:
                text += f" Restoring the previous config also failed: {restore_error}."
            return ToolResult(text, is_error=True)

        after = self._evaluate_after_convergence() if verify else {}
        regressions = [c for c in self.checks if before.get(c) and not after.get(c)]
        regression_names = [c.describe() for c in regressions]

        if regressions and self.config.auto_rollback:
            restore_error = self._restore(device, snapshot)
            self.changes.append(
                _change_record(
                    device, lines, rationale, rolled_back=True, regressions=regression_names
                )
            )
            self.audit.record(
                Event(
                    "rollback",
                    {
                        "device": device,
                        "reason": "service checks regressed",
                        "regressions": regression_names,
                    },
                )
            )
            if restore_error is not None:
                return ToolResult(
                    f"Change on {device} broke {summarize(regression_names)}, and the automatic "
                    f"rollback failed: {restore_error}. Check the device before doing anything else.",
                    is_error=True,
                )
            return ToolResult(
                f"Change on {device} was rolled back automatically because it broke: "
                f"{summarize(regression_names)}. The device is back to its previous config.",
                is_error=True,
            )

        index = len(self.changes)
        self.changes.append(
            _change_record(
                device, lines, rationale, rolled_back=False, regressions=regression_names
            )
        )
        self._undo.setdefault(device, []).append((snapshot, index))

        passing = sum(1 for c in self.checks if after.get(c)) if verify else None
        self.audit.record(
            Event(
                "change_applied",
                {
                    "device": device,
                    "lines": lines,
                    "rationale": rationale,
                    "checks_passing": passing,
                    "checks_total": len(self.checks),
                },
            )
        )
        return ToolResult(self._applied_text(device, len(lines), verify, after, regressions))

    def _evaluate_after_convergence(self) -> dict[Check, bool]:
        """Poll the checks until they all pass (after a minimum observation window) or the
        convergence wait runs out, and return the last evaluation."""
        started = time.monotonic()
        minimum = min(MINIMUM_OBSERVATION_SECONDS, self.convergence_wait_seconds)
        while True:
            results = evaluate_all(self.checks, self.lab)
            elapsed = time.monotonic() - started
            if elapsed >= self.convergence_wait_seconds:
                return results
            if all(results.values()) and elapsed >= minimum:
                return results
            self.sleep(self.poll_interval_seconds)

    def _applied_text(
        self,
        device: str,
        n_lines: int,
        verify: bool,
        after: dict[Check, bool],
        regressions: list[Check],
    ) -> str:
        head = f"Applied {_plural(n_lines, 'line')} on {device}."
        if not verify:
            return f"{head} Service checks were not evaluated because verify_after_change is off."
        if not self.checks:
            return f"{head} No service checks are configured, so nothing was verified."
        passing = sum(1 for c in self.checks if after.get(c))
        text = f"{head} Service checks: {passing}/{len(self.checks)} passing."
        if regressions:
            names = [c.describe() for c in regressions]
            text += f" Regressed, kept because auto_rollback is off: {summarize(names)}."
        failing = [c.describe() for c in self.checks if not after.get(c)]
        if failing:
            text += f" Still failing: {summarize(failing)}."
        return text

    # -------------------------------------------------------------- rollback

    def rollback(self, device: str) -> ToolResult:
        """Undo the most recent kept change on `device`, restoring the config it replaced."""
        stack = self._undo.get(device)
        if not stack:
            return ToolResult(f"No kept change on {device} to roll back.", is_error=True)
        snapshot, index = stack[-1]
        try:
            self.lab.restore(device, snapshot)
        except LabError as exc:
            return ToolResult(f"Rollback on {device} failed: {exc}", is_error=True)
        stack.pop()
        self.changes[index]["rolled_back"] = True
        self.audit.record(
            Event(
                "rollback",
                {"device": device, "reason": "rolled back on request", "regressions": []},
            )
        )
        return ToolResult(
            f"Rolled back the last change on {device}. The device is back to its previous config."
        )

    # --------------------------------------------------------------- helpers

    def _restore(self, device: str, snapshot: str) -> str | None:
        """Restore a snapshot. Returns None on success, or the error text."""
        try:
            self.lab.restore(device, snapshot)
        except LabError as exc:
            return str(exc)
        return None

    def _reject(self, device: str, reasons: list[str]) -> None:
        self.audit.record(Event("change_rejected", {"device": device, "reasons": reasons}))


def _change_record(
    device: str,
    lines: list[str],
    rationale: str,
    *,
    rolled_back: bool,
    regressions: list[str],
) -> dict[str, Any]:
    return {
        "device": device,
        "lines": list(lines),
        "rationale": rationale,
        "rolled_back": rolled_back,
        "regressions": list(regressions),
    }
