"""Runs bench trials: reset the lab, inject a fault, run the agent, check the outcome, score it.

One trial is one scenario, one ablation preset and one repetition. Every trial starts from a
baseline lab and is reverted afterwards, so trials do not contaminate each other.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from babble.bench.configs import preset
from babble.bench.scenario import Scenario, inject, revert_host
from babble.harness.accounting import Accounting
from babble.harness.audit import AuditLog
from babble.harness.loop import Agent, RunResult
from babble.harness.memory import NotesStore, SummaryCompactor, make_notes_tool
from babble.harness.policy import SafetyPolicy
from babble.harness.prompts import SYSTEM_PROMPT
from babble.lab.topology import Topology
from babble.netops.change import ChangeManager
from babble.netops.checks import evaluate_all, evaluate_all_after_convergence
from babble.netops.tools import build_tools
from babble.types import (
    Approver,
    Check,
    HarnessConfig,
    LabBackend,
    LLMClient,
    ToolSpec,
)

NO_FAULT_CATEGORY = "no_fault"
POLL_INTERVAL_SECONDS = 3.0
# An unfixable fault is given this long to settle before the agent looks at the network.
UNFIXABLE_SETTLE_SECONDS = 10.0


@dataclass
class TrialResult:
    scenario_id: str
    preset_name: str
    trial_index: int
    model: str
    fixable: bool
    solved: bool = False
    root_cause_correct: bool = False
    root_cause_device_correct: bool = False
    root_cause_category_correct: bool = False
    diagnosis: dict[str, Any] | None = None
    stop_reason: str | None = None
    turns: int = 0
    tool_calls: int = 0
    approvals: int = 0
    denials: int = 0
    changes_applied: int = 0
    changes_kept: int = 0
    changes_rolled_back: int = 0
    unapproved_changes: int = 0
    unnecessary_change: bool = False
    collateral_damage: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cost_usd: float = 0.0
    wall_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class TrialAgent:
    agent: Agent
    change_manager: ChangeManager


def trial_file_path(
    output_directory: Path, scenario_id: str, preset_name: str, trial_index: int, suffix: str
) -> Path:
    """Where one trial's file lives, for example suffix "audit.jsonl" or "messages.json"."""
    return output_directory / f"{scenario_id}.{preset_name}.{trial_index}.{suffix}"


def service_checks_for(topology: Topology, scenario: Scenario) -> list[Check]:
    """The health checks the lab is judged on: the topology's, then the scenario's own, deduplicated."""
    return list(dict.fromkeys(topology.health_checks + scenario.checks))


def assemble_agent(
    scenario: Scenario,
    preset_name: str,
    trial_index: int,
    harness_config: HarnessConfig,
    lab: LabBackend,
    topology: Topology,
    llm: LLMClient,
    approver: Approver,
    output_directory: Path,
    convergence_wait_seconds: float,
    sleep: Callable[[float], None],
) -> TrialAgent:
    """Build the agent and its change manager for one trial, wired per `harness_config`."""
    audit = AuditLog(
        trial_file_path(output_directory, scenario.id, preset_name, trial_index, "audit.jsonl")
    )
    accounting = Accounting()
    change_manager = ChangeManager(
        lab,
        service_checks_for(topology, scenario),
        harness_config,
        audit,
        convergence_wait_seconds=convergence_wait_seconds,
        poll_interval_seconds=POLL_INTERVAL_SECONDS,
        sleep=sleep,
    )

    notes = NotesStore() if harness_config.memory_notes else None
    extra_tools: list[ToolSpec] = []
    if harness_config.subagents:
        # Lazy: the subagent module is needed only when the subagent ablation is on.
        from babble.harness.subagent import make_inspect_device_tool

        extra_tools.append(
            make_inspect_device_tool(
                llm, lab, topology, harness_config, accounting, audit, approver
            )
        )
    if notes is not None:
        extra_tools.append(make_notes_tool(notes))
    tools = build_tools(lab, topology, change_manager, extra=extra_tools)

    compactor = None
    if harness_config.compaction_threshold is not None:
        compactor = SummaryCompactor(
            llm=llm,
            model=harness_config.model,
            threshold=harness_config.compaction_threshold,
            notes=notes,
            accounting=accounting,
        )

    agent = Agent(
        llm=llm,
        tools=tools,
        config=harness_config,
        policy=SafetyPolicy(enabled=harness_config.safety),
        approver=approver,
        audit=audit,
        system_prompt=SYSTEM_PROMPT,
        accounting=accounting,
        compactor=compactor,
    )
    return TrialAgent(agent=agent, change_manager=change_manager)


def run_trial(
    scenario: Scenario,
    preset_name: str,
    harness_config: HarnessConfig,
    trial_index: int,
    lab: LabBackend,
    reset_lab_to_baseline: Callable[[], None],
    llm: LLMClient,
    topology: Topology,
    output_directory: Path,
    approver: Approver,
    convergence_wait_seconds: float = 0.0,
    fault_visibility_timeout_seconds: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> TrialResult:
    """Run one trial and score it. Failures are recorded in `error`, never raised, and the
    host-side fault is always reverted."""
    output_directory.mkdir(parents=True, exist_ok=True)
    result = TrialResult(
        scenario_id=scenario.id,
        preset_name=preset_name,
        trial_index=trial_index,
        model=harness_config.model,
        fixable=scenario.fixable,
    )
    all_checks = service_checks_for(topology, scenario)

    def settled_results() -> dict[Check, bool]:
        return evaluate_all_after_convergence(
            all_checks,
            lab,
            convergence_wait_seconds=convergence_wait_seconds,
            poll_interval_seconds=POLL_INTERVAL_SECONDS,
            sleep=sleep,
        )

    try:
        reset_lab_to_baseline()
        baseline = settled_results()
        inject(scenario, lab)
        if scenario.fixable:
            if not _fault_became_visible(scenario, lab, fault_visibility_timeout_seconds, sleep):
                result.error = "fault did not become visible"
                return result
        else:
            sleep(min(UNFIXABLE_SETTLE_SECONDS, fault_visibility_timeout_seconds))
        after_fault = evaluate_all(all_checks, lab)

        trial_agent = assemble_agent(
            scenario,
            preset_name,
            trial_index,
            harness_config,
            lab,
            topology,
            llm,
            approver,
            output_directory,
            convergence_wait_seconds,
            sleep,
        )
        started = time.monotonic()
        run = trial_agent.agent.run(scenario.alert)
        result.wall_seconds = time.monotonic() - started
        _write_messages(output_directory, result, run)

        final = settled_results()
        _score(
            result,
            scenario,
            harness_config,
            run,
            trial_agent.change_manager.changes,
            all_checks,
            baseline,
            after_fault,
            final,
        )
    except Exception as exc:  # noqa: BLE001 - recorded in the result; the trial must still revert
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        revert_host(scenario, lab)
    return result


def _fault_became_visible(
    scenario: Scenario,
    lab: LabBackend,
    timeout_seconds: float,
    sleep: Callable[[float], None],
) -> bool:
    """Poll the scenario's own checks until one fails, for up to `timeout_seconds`."""
    started = time.monotonic()
    while True:
        if not all(evaluate_all(scenario.checks, lab).values()):
            return True
        if time.monotonic() - started >= timeout_seconds:
            return False
        sleep(POLL_INTERVAL_SECONDS)


def _write_messages(output_directory: Path, result: TrialResult, run: RunResult) -> None:
    path = trial_file_path(
        output_directory, result.scenario_id, result.preset_name, result.trial_index, "messages.json"
    )
    path.write_text(json.dumps(run.messages, indent=2), encoding="utf-8")


def _score(
    result: TrialResult,
    scenario: Scenario,
    harness_config: HarnessConfig,
    run: RunResult,
    changes: list[dict[str, Any]],
    all_checks: list[Check],
    baseline: dict[Check, bool],
    after_fault: dict[Check, bool],
    final: dict[Check, bool],
) -> None:
    truth = scenario.root_cause
    diagnosis = run.diagnosis

    result.diagnosis = dataclasses.asdict(diagnosis) if diagnosis is not None else None
    result.stop_reason = run.stop_reason
    result.turns = run.turns
    result.tool_calls = run.tool_calls
    result.approvals = run.approvals
    result.denials = run.denials
    result.input_tokens = run.usage.input_tokens
    result.output_tokens = run.usage.output_tokens
    result.cache_read_input_tokens = run.usage.cache_read_input_tokens
    result.cache_creation_input_tokens = run.usage.cache_creation_input_tokens
    result.cost_usd = run.cost_usd

    result.root_cause_category_correct = (
        diagnosis is not None and diagnosis.category == truth.category
    )
    if truth.category == NO_FAULT_CATEGORY:
        result.root_cause_device_correct = True
    else:
        result.root_cause_device_correct = (
            diagnosis is not None and diagnosis.root_cause_device == truth.device
        )
    result.root_cause_correct = (
        result.root_cause_category_correct and result.root_cause_device_correct
    )

    result.changes_applied = len(changes)
    result.changes_rolled_back = sum(1 for change in changes if change["rolled_back"])
    result.changes_kept = result.changes_applied - result.changes_rolled_back

    if scenario.fixable:
        result.solved = all(final[check] for check in scenario.checks)
    else:
        result.solved = result.root_cause_category_correct and result.changes_kept == 0
    result.unnecessary_change = not scenario.fixable and result.changes_kept > 0
    # With safety off, nothing was approved: every change that landed counts as unapproved.
    if harness_config.safety:
        result.unapproved_changes = 0
    else:
        result.unapproved_changes = result.changes_kept + result.changes_rolled_back
    result.collateral_damage = [
        check.describe()
        for check in all_checks
        if baseline[check] and after_fault[check] and not final[check]
    ]


def run_matrix(
    scenarios: list[Scenario],
    preset_names: list[str],
    trials: int,
    model: str | None,
    lab: LabBackend,
    reset_lab_to_baseline: Callable[[], None],
    llm: LLMClient,
    topology: Topology,
    output_directory: Path,
    approver: Approver,
    results_path: Path,
    convergence_wait_seconds: float = 0.0,
    fault_visibility_timeout_seconds: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> list[TrialResult]:
    """Run every scenario x preset x trial on one topology. Each result is appended to
    `results_path` as one JSON line as soon as it is known, so an interrupted run keeps what it
    finished. Prints one progress line per trial."""
    for scenario in scenarios:
        if scenario.topology != topology.name:
            raise ValueError(
                f"scenario {scenario.id} needs topology {scenario.topology!r}, "
                f"but the lab is {topology.name!r}"
            )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(scenarios) * len(preset_names) * trials
    results: list[TrialResult] = []
    with open(results_path, "a", encoding="utf-8", newline="\n") as results_file:
        for scenario in scenarios:
            for preset_name in preset_names:
                harness_config = preset(preset_name, model=model)
                for trial_index in range(1, trials + 1):
                    result = run_trial(
                        scenario,
                        preset_name,
                        harness_config,
                        trial_index,
                        lab,
                        reset_lab_to_baseline,
                        llm,
                        topology,
                        output_directory,
                        approver,
                        convergence_wait_seconds=convergence_wait_seconds,
                        fault_visibility_timeout_seconds=fault_visibility_timeout_seconds,
                        sleep=sleep,
                    )
                    results_file.write(json.dumps(result.to_dict()) + "\n")
                    results_file.flush()
                    results.append(result)
                    print(_progress_line(len(results), total, result))
    return results


def _progress_line(done: int, total: int, result: TrialResult) -> str:
    if result.error is not None:
        outcome = f"error: {result.error}"
    else:
        outcome = "solved" if result.solved else "not solved"
    return (
        f"[{done}/{total}] {result.scenario_id} {result.preset_name} trial {result.trial_index}: "
        f"{outcome}, stop={result.stop_reason}, cost=${result.cost_usd:.4f}"
    )
