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

from babble.bench.configs import PRESETS, harness_config_for_preset
from babble.bench.report import (
    load_results,
    render_markdown_table,
    solved_counts_by_scenario_table,
    summarize_by_preset,
)
from babble.bench.runner import TrialResult, run_matrix, run_trial
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


def scenarios_for_ids(scenario_ids: list[str]) -> list[Scenario]:
    scenarios_by_id = {scenario.id: scenario for scenario in load_all()}
    unknown = [name for name in scenario_ids if name not in scenarios_by_id]
    if unknown:
        raise UsageError(
            f"unknown scenario(s): {', '.join(unknown)}; "
            f"valid scenarios: {', '.join(scenarios_by_id)}"
        )
    return [scenarios_by_id[name] for name in scenario_ids]


def validated_preset_names(preset_names: list[str]) -> list[str]:
    unknown = [name for name in preset_names if name not in PRESETS]
    if unknown:
        raise UsageError(
            f"unknown harness_config_for_preset(s): {', '.join(unknown)}; "
            f"valid presets: {', '.join(PRESETS)}"
        )
    return preset_names


def split_comma_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def default_output_directory() -> Path:
    return Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")


# ------------------------------------------------------------------ lab handling


def bring_lab_to_baseline(manager: LabManager) -> None:
    """Reset the running lab to the baseline saved at deploy time, then fail fast unless it is
    healthy."""
    manager.reset()
    manager.wait_healthy(
        evaluate, timeout=LAB_READY_TIMEOUT_SECONDS, interval=LAB_POLL_INTERVAL_SECONDS
    )


def default_lab_at_baseline() -> tuple[Topology, LabManager]:
    """The default topology and its running lab, verified healthy and recorded as the baseline."""
    topology = load_topology(DEFAULT_TOPOLOGY_NAME)
    manager = build_lab_manager(topology)
    bring_lab_to_baseline(manager)
    return topology, manager


def command_lab_up(args: argparse.Namespace) -> int:
    topology = load_topology(args.topology)
    manager = build_lab_manager(topology)
    manager.deploy()
    manager.wait_healthy(
        evaluate, timeout=LAB_UP_TIMEOUT_SECONDS, interval=LAB_POLL_INTERVAL_SECONDS
    )
    manager.save_baseline()
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
    scenario = scenarios_for_ids([args.scenario_id])[0]
    preset_name = validated_preset_names([args.preset])[0]
    harness_config = harness_config_for_preset(preset_name, model=args.model)
    output_directory = Path(args.output_directory or default_output_directory())
    topology, manager = default_lab_at_baseline()
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
        load_all()
        if args.scenarios == "all"
        else scenarios_for_ids(split_comma_list(args.scenarios))
    )
    preset_names = validated_preset_names(split_comma_list(args.presets))
    output_directory = Path(args.output_directory or default_output_directory())
    topology, manager = default_lab_at_baseline()
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
    expected = _device_and_category(scenario.root_cause.device, scenario.root_cause.category)
    if result.diagnosis is None:
        diagnosed = "no diagnosis"
    else:
        diagnosed = _device_and_category(
            result.diagnosis["root_cause_device"], result.diagnosis["category"]
        )
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
        f"Audit log: {result.trial_file(output_directory, 'audit.jsonl')}",
        f"Messages: {result.trial_file(output_directory, 'messages.json')}",
    ]
    if result.error is not None:
        lines.append(f"Error: {result.error}")
    return "\n".join(lines)


def _device_and_category(device: str, category: str) -> str:
    return f"{device or '-'}/{category}"


def render_report(results: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "## Summary by preset",
            "",
            render_markdown_table(summarize_by_preset(results)),
            "",
            "## Solved per scenario",
            "",
            solved_counts_by_scenario_table(results),
            "",
        ]
    )


# --------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="babble", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    lab_parser = commands.add_parser(
        "lab", help="bring the containerlab topology up, down or check it"
    )
    lab_subcommands = lab_parser.add_subparsers(dest="lab_command", required=True)
    for subcommand_name, handler, help_text in [
        ("up", command_lab_up, "deploy the topology and wait until it is healthy"),
        ("down", command_lab_down, "destroy the topology"),
        ("status", command_lab_status, "print PASS or FAIL for each health check"),
    ]:
        subcommand_parser = lab_subcommands.add_parser(subcommand_name, help=help_text)
        subcommand_parser.add_argument("--topology", default=DEFAULT_TOPOLOGY_NAME)
        subcommand_parser.set_defaults(handler=handler)

    scenarios_parser = commands.add_parser("scenarios", help="list the fault scenarios")
    scenarios_parser.set_defaults(handler=command_scenarios)

    run_parser = commands.add_parser("run", help="run one scenario once against the running lab")
    run_parser.add_argument("scenario_id")
    run_parser.add_argument("--preset", default="full")
    run_parser.add_argument("--model")
    run_parser.add_argument("--approval", choices=APPROVAL_MODES, default="auto_approve")
    run_parser.add_argument("--output-directory")
    run_parser.set_defaults(handler=command_run)

    bench_parser = commands.add_parser("bench", help="run scenarios x presets x trials and report")
    bench_parser.add_argument("--scenarios", default="all", help="comma-separated ids, or all")
    bench_parser.add_argument("--presets", default="full", help="comma-separated preset names")
    bench_parser.add_argument("--trials", type=int, default=3)
    bench_parser.add_argument("--model")
    bench_parser.add_argument("--output-directory")
    bench_parser.set_defaults(handler=command_bench)

    report_parser = commands.add_parser("report", help="print the report for a results.jsonl file")
    report_parser.add_argument("results_jsonl")
    report_parser.set_defaults(handler=command_report)

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
