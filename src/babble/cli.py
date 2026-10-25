"""Command-line entry point: `babble lab`, `scenarios`, `run`, `bench` and `report`.

Factories (`build_lab_manager`, `build_llm_client`) are the only places that touch Docker or the
Anthropic API, so tests replace them with fakes.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from babble.bench.configs import PRESETS, preset
from babble.bench.report import (
    load_results,
    per_scenario_table,
    render_markdown_table,
    summarize_by_preset,
)
from babble.bench.runner import TrialResult, run_matrix, run_trial, trial_file_path
from babble.bench.scenario import Scenario, load_all
from babble.harness.policy import make_approver
from babble.lab.clab import CommandRunner, LabManager
from babble.lab.topology import Topology, load_topology
from babble.llm import AnthropicClient
from babble.netops.checks import evaluate
from babble.types import ApprovalMode, LabError, LLMClient

DEFAULT_TOPOLOGY_NAME = "metro"
APPROVAL_MODES: tuple[str, ...] = ("human", "auto_approve", "auto_deny")
BENCH_APPROVAL_MODE: ApprovalMode = "auto_approve"

LAB_UP_TIMEOUT_SECONDS = 150
LAB_READY_TIMEOUT_SECONDS = 30
LAB_POLL_INTERVAL_SECONDS = 5
CONVERGENCE_WAIT_SECONDS = 30
FAULT_VISIBILITY_TIMEOUT_SECONDS = 90

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2


class UsageError(Exception):
    """A bad command-line value. Reported with exit code 2, before the lab is touched."""


# ------------------------------------------------------------------- factories


def build_lab_manager(topology: Topology) -> LabManager:
    return LabManager(topology, CommandRunner.from_env())


def build_llm_client() -> LLMClient:
    return AnthropicClient()


# ------------------------------------------------------------------- selection


def scenarios_named(scenario_ids: list[str]) -> list[Scenario]:
    by_id = {scenario.id: scenario for scenario in load_all()}
    unknown = [name for name in scenario_ids if name not in by_id]
    if unknown:
        raise UsageError(
            f"unknown scenario(s): {', '.join(unknown)}; valid scenarios: {', '.join(by_id)}"
        )
    return [by_id[name] for name in scenario_ids]


def validated_preset_names(preset_names: list[str]) -> list[str]:
    unknown = [name for name in preset_names if name not in PRESETS]
    if unknown:
        raise UsageError(
            f"unknown preset(s): {', '.join(unknown)}; valid presets: {', '.join(PRESETS)}"
        )
    return preset_names


def split_comma_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def default_output_directory() -> Path:
    return Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")


# ------------------------------------------------------------------ lab handling


def bring_lab_to_baseline(manager: LabManager) -> None:
    """Fail fast unless the running lab is healthy, then record it as the baseline to reset to."""
    manager.wait_healthy(timeout=LAB_READY_TIMEOUT_SECONDS, interval=LAB_POLL_INTERVAL_SECONDS)
    manager.take_baseline()


def command_lab_up(args: argparse.Namespace) -> int:
    topology = load_topology(args.topology)
    manager = build_lab_manager(topology)
    manager.deploy()
    manager.wait_healthy(timeout=LAB_UP_TIMEOUT_SECONDS, interval=LAB_POLL_INTERVAL_SECONDS)
    print(f"lab {topology.name} is healthy")
    return EXIT_OK


def command_lab_down(args: argparse.Namespace) -> int:
    topology = load_topology(args.topology)
    build_lab_manager(topology).destroy()
    print(f"lab {topology.name} is destroyed")
    return EXIT_OK


def command_lab_status(args: argparse.Namespace) -> int:
    topology = load_topology(args.topology)
    lab = build_lab_manager(topology).lab
    all_checks_pass = True
    for check in topology.health_checks:
        passing = evaluate(check, lab)
        all_checks_pass = all_checks_pass and passing
        print(f"{'PASS' if passing else 'FAIL'}  {check.describe()}")
    return EXIT_OK if all_checks_pass else EXIT_FAILURE


# -------------------------------------------------------------------- commands


def command_scenarios(_: argparse.Namespace) -> int:
    print(f"{'ID':<28} {'DIFFICULTY':<10} {'FIXABLE':<8} TITLE")
    for scenario in load_all():
        fixable = "yes" if scenario.fixable else "no"
        print(f"{scenario.id:<28} {scenario.difficulty:<10} {fixable:<8} {scenario.title}")
    return EXIT_OK


def command_run(args: argparse.Namespace) -> int:
    scenario = scenarios_named([args.scenario_id])[0]
    preset_name = validated_preset_names([args.preset])[0]
    harness_config = preset(preset_name, model=args.model)
    output_directory = Path(args.output_directory or default_output_directory())
    topology = load_topology(DEFAULT_TOPOLOGY_NAME)

    manager = build_lab_manager(topology)
    bring_lab_to_baseline(manager)
    result = run_trial(
        scenario,
        preset_name,
        harness_config,
        1,
        manager.lab,
        manager.reset,
        build_llm_client(),
        topology,
        output_directory,
        make_approver(args.approval),
        convergence_wait_seconds=CONVERGENCE_WAIT_SECONDS,
        fault_visibility_timeout_seconds=FAULT_VISIBILITY_TIMEOUT_SECONDS,
    )
    print(describe_trial(result, scenario, output_directory))
    return EXIT_OK if result.solved else EXIT_FAILURE


def command_bench(args: argparse.Namespace) -> int:
    scenarios = (
        load_all() if args.scenarios == "all" else scenarios_named(split_comma_list(args.scenarios))
    )
    preset_names = validated_preset_names(split_comma_list(args.presets))
    output_directory = Path(args.output_directory or default_output_directory())
    topology = load_topology(DEFAULT_TOPOLOGY_NAME)

    manager = build_lab_manager(topology)
    bring_lab_to_baseline(manager)
    results_path = output_directory / "results.jsonl"
    run_matrix(
        scenarios,
        preset_names,
        args.trials,
        args.model,
        manager.lab,
        manager.reset,
        build_llm_client(),
        topology,
        output_directory,
        make_approver(BENCH_APPROVAL_MODE),
        results_path,
        convergence_wait_seconds=CONVERGENCE_WAIT_SECONDS,
        fault_visibility_timeout_seconds=FAULT_VISIBILITY_TIMEOUT_SECONDS,
    )
    report_text = render_report(load_results(results_path))
    (output_directory / "report.md").write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"results: {results_path}")
    print(f"report: {output_directory / 'report.md'}")
    return EXIT_OK


def command_report(args: argparse.Namespace) -> int:
    print(render_report(load_results(Path(args.results_jsonl))))
    return EXIT_OK


# ------------------------------------------------------------------- rendering


def describe_trial(result: TrialResult, scenario: Scenario, output_directory: Path) -> str:
    expected = f"{scenario.root_cause.device or '-'}/{scenario.root_cause.category}"
    if result.diagnosis is None:
        diagnosed = "no diagnosis"
    else:
        diagnosed = f"{result.diagnosis['root_cause_device'] or '-'}/{result.diagnosis['category']}"
    root_cause_verdict = "correct" if result.root_cause_correct else "wrong"
    lines = [
        f"Scenario: {scenario.id} ({scenario.title})",
        f"Solved: {'yes' if result.solved else 'no'}",
        f"Root cause: diagnosed {diagnosed}, expected {expected} ({root_cause_verdict})",
        (
            f"Changes: applied {result.changes_applied}, kept {result.changes_kept}, "
            f"rolled back {result.changes_rolled_back}"
        ),
        f"Turns: {result.turns}  Stop reason: {result.stop_reason}",
        f"Cost: ${result.cost_usd:.4f}  Wall time: {result.wall_seconds:.1f}s",
        f"Audit log: {_trial_file(result, output_directory, 'audit.jsonl')}",
        f"Messages: {_trial_file(result, output_directory, 'messages.json')}",
    ]
    if result.error is not None:
        lines.append(f"Error: {result.error}")
    return "\n".join(lines)


def _trial_file(result: TrialResult, output_directory: Path, suffix: str) -> Path:
    return trial_file_path(
        output_directory, result.scenario_id, result.preset_name, result.trial_index, suffix
    )


def render_report(results: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "## Summary by preset",
            "",
            render_markdown_table(summarize_by_preset(results)),
            "",
            "## Solved per scenario",
            "",
            per_scenario_table(results),
            "",
        ]
    )


# --------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="babble", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    lab = commands.add_parser("lab", help="bring the containerlab topology up, down or check it")
    lab_commands = lab.add_subparsers(dest="lab_command", required=True)
    for name, handler, help_text in [
        ("up", command_lab_up, "deploy the topology and wait until it is healthy"),
        ("down", command_lab_down, "destroy the topology"),
        ("status", command_lab_status, "print PASS or FAIL for each health check"),
    ]:
        lab_parser = lab_commands.add_parser(name, help=help_text)
        lab_parser.add_argument("--topology", default=DEFAULT_TOPOLOGY_NAME)
        lab_parser.set_defaults(handler=handler)

    scenarios = commands.add_parser("scenarios", help="list the fault scenarios")
    scenarios.set_defaults(handler=command_scenarios)

    run = commands.add_parser("run", help="run one scenario once against the running lab")
    run.add_argument("scenario_id")
    run.add_argument("--preset", default="full")
    run.add_argument("--model")
    run.add_argument("--approval", choices=APPROVAL_MODES, default="auto_approve")
    run.add_argument("--output-directory")
    run.set_defaults(handler=command_run)

    bench = commands.add_parser("bench", help="run scenarios x presets x trials and report")
    bench.add_argument("--scenarios", default="all", help="comma-separated ids, or all")
    bench.add_argument("--presets", default="full", help="comma-separated preset names")
    bench.add_argument("--trials", type=int, default=3)
    bench.add_argument("--model")
    bench.add_argument("--output-directory")
    bench.set_defaults(handler=command_bench)

    report = commands.add_parser("report", help="print the report for a results.jsonl file")
    report.add_argument("results_jsonl")
    report.set_defaults(handler=command_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        handler: Callable[[argparse.Namespace], int] = args.handler
        return handler(args)
    except UsageError as error:
        print(f"babble: {error}", file=sys.stderr)
        return EXIT_USAGE
    except (LabError, OSError) as error:
        print(f"babble: {error}", file=sys.stderr)
        return EXIT_FAILURE
