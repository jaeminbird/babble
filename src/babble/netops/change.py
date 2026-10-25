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
at a time, newest first. A change whose automatic rollback failed is still on the device, so it
goes on the stack too: `rollback` can retry the restore.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from babble.netops.checks import evaluate_all, evaluate_all_after_convergence
from babble.netops.commands import find_blocked_config_reasons
from babble.types import AuditSink, Check, Event, HarnessConfig, LabBackend, LabError, ToolResult

# Result texts are read by the model, so long lists are cut to this many entries.
MAX_LISTED_CHECKS = 8


def _summarize_check_list(items: Sequence[str]) -> str:
    """Join up to MAX_LISTED_CHECKS items, then say how many were left out."""
    shown_items = ", ".join(items[:MAX_LISTED_CHECKS])
    items_left_out = len(items) - MAX_LISTED_CHECKS
    return f"{shown_items}, and {items_left_out} more" if items_left_out > 0 else shown_items


def _pluralize(count: int, singular_noun: str) -> str:
    return f"{count} {singular_noun}" if count == 1 else f"{count} {singular_noun}s"


class ChangeManager:
    def __init__(
        self,
        lab: LabBackend,
        checks: Sequence[Check],
        config: HarnessConfig,
        audit: AuditSink,
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
        # One dict per applied change: {device, lines, rationale, rolled_back, rollback_failed,
        # regressions}. A failed automatic rollback leaves the change on the device, so it is
        # recorded with rolled_back False and rollback_failed True.
        self.change_records: list[dict[str, Any]] = []
        # Per device, (snapshot taken before a kept change, index into self.change_records),
        # oldest first.
        self._undo_stacks_by_device: dict[str, list[tuple[str, int]]] = {}

    # ----------------------------------------------------------------- apply

    def apply(self, device: str, lines: list[str], rationale: str) -> ToolResult:
        lines = list(lines)
        if not lines:
            return ToolResult("Change not applied: no config lines were given.", is_error=True)

        reasons = find_blocked_config_reasons(lines)
        if reasons:
            self._reject(device, reasons)
            return ToolResult(
                f"Change on {device} was not applied: {'; '.join(reasons)}. Nothing was changed.",
                is_error=True,
            )

        try:
            snapshot = self.lab.snapshot(device)
        except LabError as lab_error:
            return ToolResult(f"Change on {device} was not applied: {lab_error}", is_error=True)

        verify_after_change = self.config.verify_after_change
        checks_passing_before_change = (
            evaluate_all(self.checks, self.lab) if verify_after_change else {}
        )

        try:
            self.lab.configure(device, lines)
        except LabError as lab_error:
            # The device may have taken some of the lines before failing, so put it back.
            restore_error = self._restore(device, snapshot)
            self._reject(device, [f"CLI error: {lab_error}"])
            result_text = f"Change on {device} was not applied: {lab_error}."
            if restore_error is None:
                result_text += " The device config was restored to its previous state."
            else:
                result_text += f" Restoring the previous config also failed: {restore_error}."
            return ToolResult(result_text, is_error=True)

        checks_passing_after_change = (
            evaluate_all_after_convergence(
                self.checks,
                self.lab,
                self.convergence_wait_seconds,
                self.poll_interval_seconds,
                self.sleep,
            )
            if verify_after_change
            else {}
        )
        checks_regressed_by_change = [
            check
            for check in self.checks
            if checks_passing_before_change.get(check)
            and not checks_passing_after_change.get(check)
        ]
        regressed_check_descriptions = [check.describe() for check in checks_regressed_by_change]

        if checks_regressed_by_change and self.config.auto_rollback:
            restore_error = self._restore(device, snapshot)
            index_of_recorded_change = len(self.change_records)
            self.change_records.append(
                _change_record(
                    device,
                    lines,
                    rationale,
                    rolled_back=restore_error is None,
                    rollback_failed=restore_error is not None,
                    regressed_check_descriptions=regressed_check_descriptions,
                )
            )
            if restore_error is not None:
                self._undo_stacks_by_device.setdefault(device, []).append(
                    (snapshot, index_of_recorded_change)
                )
                self._record_rollback_failed(device, regressed_check_descriptions, restore_error)
                return ToolResult(
                    f"Change on {device} broke "
                    f"{_summarize_check_list(regressed_check_descriptions)}, and the "
                    f"automatic rollback failed: {restore_error}. Check the device before doing "
                    "anything else.",
                    is_error=True,
                )
            # A restored config can take a BGP update cycle to carry traffic again, so wait for
            # the broken checks to recover instead of claiming they have.
            checks_passing_after_rollback = evaluate_all_after_convergence(
                checks_regressed_by_change,
                self.lab,
                self.convergence_wait_seconds,
                self.poll_interval_seconds,
                self.sleep,
            )
            still_failing_after_rollback = [
                check.describe()
                for check in checks_regressed_by_change
                if not checks_passing_after_rollback[check]
            ]
            self._record_rollback(
                device,
                "service checks regressed",
                regressed_check_descriptions,
                still_failing_after_rollback,
            )
            result_text = (
                f"Change on {device} was rolled back automatically because it broke: "
                f"{_summarize_check_list(regressed_check_descriptions)}. "
                "The device is back to its previous config"
            )
            if still_failing_after_rollback:
                result_text += (
                    ", but these checks have not recovered: "
                    f"{_summarize_check_list(still_failing_after_rollback)}."
                )
            else:
                result_text += ", and those checks pass again."
            return ToolResult(result_text, is_error=True)

        index_of_recorded_change = len(self.change_records)
        self.change_records.append(
            _change_record(
                device,
                lines,
                rationale,
                rolled_back=False,
                rollback_failed=False,
                regressed_check_descriptions=regressed_check_descriptions,
            )
        )
        self._undo_stacks_by_device.setdefault(device, []).append(
            (snapshot, index_of_recorded_change)
        )

        number_of_checks_passing = (
            sum(1 for check in self.checks if checks_passing_after_change.get(check))
            if verify_after_change
            else None
        )
        self.audit.record(
            Event(
                "change_applied",
                {
                    "device": device,
                    "lines": lines,
                    "rationale": rationale,
                    "checks_passing": number_of_checks_passing,
                    "checks_total": len(self.checks),
                },
            )
        )
        return ToolResult(
            self._applied_text(
                device,
                len(lines),
                verify_after_change,
                checks_passing_after_change,
                checks_regressed_by_change,
            )
        )

    def _applied_text(
        self,
        device: str,
        number_of_lines: int,
        verify_after_change: bool,
        checks_passing_after_change: dict[Check, bool],
        checks_regressed_by_change: list[Check],
    ) -> str:
        opening_sentence = f"Applied {_pluralize(number_of_lines, 'line')} on {device}."
        if not verify_after_change:
            return (
                f"{opening_sentence} Service checks were not evaluated because "
                "verify_after_change is off."
            )
        if not self.checks:
            return f"{opening_sentence} No service checks are configured, so nothing was verified."
        number_passing = sum(1 for check in self.checks if checks_passing_after_change.get(check))
        result_text = (
            f"{opening_sentence} Service checks: {number_passing}/{len(self.checks)} passing."
        )
        if checks_regressed_by_change:
            regressed_names = [check.describe() for check in checks_regressed_by_change]
            result_text += (
                f" Regressed, kept because auto_rollback is off: "
                f"{_summarize_check_list(regressed_names)}."
            )
        still_failing = [
            check.describe() for check in self.checks if not checks_passing_after_change.get(check)
        ]
        if still_failing:
            result_text += f" Still failing: {_summarize_check_list(still_failing)}."
        return result_text

    # -------------------------------------------------------------- rollback

    def rollback(self, device: str) -> ToolResult:
        """Undo the most recent kept change on `device`, restoring the config it replaced."""
        undo_stack = self._undo_stacks_by_device.get(device)
        if not undo_stack:
            return ToolResult(f"No kept change on {device} to roll back.", is_error=True)
        snapshot_before_change, index_of_change = undo_stack[-1]
        try:
            self.lab.restore(device, snapshot_before_change)
        except LabError as lab_error:
            return ToolResult(f"Rollback on {device} failed: {lab_error}", is_error=True)
        undo_stack.pop()
        self.change_records[index_of_change]["rolled_back"] = True
        self._record_rollback(device, "rolled back on request", [], [])
        return ToolResult(
            f"Rolled back the last change on {device}. The device is back to its previous config."
        )

    # --------------------------------------------------------------- helpers

    def _restore(self, device: str, snapshot: str) -> str | None:
        """Restore a snapshot. Returns None on success, or the error text."""
        try:
            self.lab.restore(device, snapshot)
        except LabError as lab_error:
            return str(lab_error)
        return None

    def _record_rollback(
        self,
        device: str,
        reason: str,
        regressions: list[str],
        still_failing_after_rollback: list[str],
    ) -> None:
        self.audit.record(
            Event(
                "rollback",
                {
                    "device": device,
                    "reason": reason,
                    "regressions": regressions,
                    "still_failing_after_rollback": still_failing_after_rollback,
                },
            )
        )

    def _record_rollback_failed(
        self, device: str, regressions: list[str], restore_error: str
    ) -> None:
        self.audit.record(
            Event(
                "rollback_failed",
                {
                    "device": device,
                    "regressions": regressions,
                    "restore_error": restore_error,
                },
            )
        )

    def _reject(self, device: str, reasons: list[str]) -> None:
        self.audit.record(Event("change_rejected", {"device": device, "reasons": reasons}))


def _change_record(
    device: str,
    lines: list[str],
    rationale: str,
    *,
    rolled_back: bool,
    rollback_failed: bool,
    regressed_check_descriptions: list[str],
) -> dict[str, Any]:
    return {
        "device": device,
        "lines": list(lines),
        "rationale": rationale,
        "rolled_back": rolled_back,
        "rollback_failed": rollback_failed,
        "regressions": list(regressed_check_descriptions),
    }
