from __future__ import annotations

import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from babble import cli
from babble.bench.report import load_results
from babble.bench.runner import TrialResult
from babble.bench.scenario import load_all
from babble.lab.scripted import ScriptedLab
from babble.lab.topology import Topology, load_topology
from babble.llm import FakeLLMClient, response, tool_use
from babble.netops import checks
from babble.types import LabError, LLMResponse, PingResult

TOPOLOGY = load_topology("metro")
NO_FAULT_SCENARIO = "no-fault-control"


class HealthyLab(ScriptedLab):
    """Every metro health check passes: every ping succeeds and every BGP session is Established."""

    def __init__(self) -> None:
        super().__init__(devices=list(TOPOLOGY.devices), responses={}, hosts=set(TOPOLOGY.hosts()))

    def ping(self, device: str, target: str) -> PingResult:
        return PingResult(sent=3, received=3, output="")

    def show(self, device: str, command: str) -> str:
        # "show bgp neighbors <peer> json"
        peer = command.split()[3]
        return json.dumps({peer: {"bgpState": "Established"}})


class FakeLabManager:
    """Stands in for LabManager and records the calls the CLI makes to it."""

    def __init__(self, topology: Topology, lab: ScriptedLab | None = None) -> None:
        self.topology = topology
        self.lab = lab if lab is not None else HealthyLab()
        self.health_error: str | None = None
        self.calls: list[str] = []

    def deploy(self) -> None:
        self.calls.append("deploy")

    def destroy(self) -> None:
        self.calls.append("destroy")

    def wait_healthy(self, evaluate: Any, timeout: float = 120, interval: float = 5) -> None:
        self.calls.append(f"wait_healthy({timeout:g})")
        if self.health_error is not None:
            raise LabError(self.health_error)

    def save_baseline(self) -> None:
        self.calls.append("save_baseline")

    def reset(self) -> None:
        self.calls.append("reset")


def no_fault_diagnosis() -> LLMResponse:
    return response(
        tool_use(
            "submit_diagnosis",
            {
                "root_cause_device": "",
                "category": "no_fault",
                "summary": "Nothing is wrong.",
                "evidence": [],
                "fix_applied": False,
            },
        )
    )


def wrong_diagnosis() -> LLMResponse:
    return response(
        tool_use(
            "submit_diagnosis",
            {
                "root_cause_device": "",
                "category": "other",
                "summary": "Guessing.",
                "evidence": [],
                "fix_applied": False,
            },
        )
    )


@pytest.fixture(autouse=True)
def no_real_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the real sleeps in trial settling and the convergence observation window."""

    def no_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(cli, "run_trial", partial(cli.run_trial, sleep=no_sleep))
    monkeypatch.setattr(cli, "run_matrix", partial(cli.run_matrix, sleep=no_sleep))
    monkeypatch.setattr(checks, "MINIMUM_OBSERVATION_SECONDS", 0.0)


@pytest.fixture
def fake_lab(monkeypatch: pytest.MonkeyPatch) -> FakeLabManager:
    fake = FakeLabManager(TOPOLOGY)
    monkeypatch.setattr(cli, "build_lab_manager", lambda topology: fake)
    return fake


@pytest.fixture
def scripted_model(monkeypatch: pytest.MonkeyPatch) -> Callable[[list[LLMResponse]], None]:
    def install(responses: list[LLMResponse]) -> None:
        client = FakeLLMClient(responses)
        monkeypatch.setattr(cli, "build_llm_client", lambda: client)

    return install


@pytest.fixture
def lab_must_not_be_touched(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(topology: Topology) -> None:
        raise AssertionError("the lab must not be touched")

    monkeypatch.setattr(cli, "build_lab_manager", refuse)


# --------------------------------------------------------------- argument parsing


@pytest.mark.parametrize(
    ("argv", "handler"),
    [
        (["lab", "up"], cli.command_lab_up),
        (["lab", "down"], cli.command_lab_down),
        (["lab", "status"], cli.command_lab_status),
        (["scenarios"], cli.command_scenarios),
        (["run", NO_FAULT_SCENARIO], cli.command_run),
        (["bench"], cli.command_bench),
        (["report", "results.jsonl"], cli.command_report),
    ],
)
def test_every_subcommand_routes_to_its_handler(argv: list[str], handler: Callable) -> None:
    assert cli.build_parser().parse_args(argv).handler is handler


@pytest.mark.parametrize("argv", [["lab", "up"], ["lab", "down"], ["lab", "status"]])
def test_lab_commands_default_to_the_metro_topology(argv: list[str]) -> None:
    assert cli.build_parser().parse_args(argv).topology == "metro"


def test_lab_up_accepts_a_topology_name() -> None:
    args = cli.build_parser().parse_args(["lab", "up", "--topology", "other"])
    assert args.topology == "other"


def test_run_defaults_to_full_preset_auto_approval_and_no_output_directory() -> None:
    args = cli.build_parser().parse_args(["run", NO_FAULT_SCENARIO])
    assert args.scenario_id == NO_FAULT_SCENARIO
    assert args.preset == "full"
    assert args.model is None
    assert args.approval == "auto_approve"
    assert args.output_directory is None


def test_run_accepts_every_option() -> None:
    args = cli.build_parser().parse_args(
        [
            "run",
            NO_FAULT_SCENARIO,
            "--preset",
            "no_subagents",
            "--model",
            "claude-haiku-5-5",
            "--approval",
            "auto_deny",
            "--output-directory",
            "out",
        ]
    )
    assert (args.preset, args.model, args.approval, args.output_directory) == (
        "no_subagents",
        "claude-haiku-5-5",
        "auto_deny",
        "out",
    )


def test_run_rejects_an_approval_mode_it_does_not_know() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", NO_FAULT_SCENARIO, "--approval", "maybe"])


def test_bench_defaults_to_every_scenario_full_preset_and_three_trials() -> None:
    args = cli.build_parser().parse_args(["bench"])
    assert args.scenarios == "all"
    assert args.presets == "full"
    assert args.trials == 3
    assert args.model is None
    assert args.output_directory is None


def test_bench_accepts_every_option() -> None:
    args = cli.build_parser().parse_args(
        [
            "bench",
            "--scenarios",
            "a,b",
            "--presets",
            "full,no_subagents",
            "--trials",
            "1",
            "--model",
            "m",
            "--output-directory",
            "out",
        ]
    )
    assert (args.scenarios, args.presets, args.trials, args.model, args.output_directory) == (
        "a,b",
        "full,no_subagents",
        1,
        "m",
        "out",
    )


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


# ------------------------------------------------------------------ unknown names


def test_run_rejects_an_unknown_scenario_before_touching_the_lab(
    lab_must_not_be_touched: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["run", "no-such-scenario"]) == 2

    err = capsys.readouterr().err
    assert "unknown scenario(s): no-such-scenario" in err
    assert NO_FAULT_SCENARIO in err


def test_run_rejects_an_unknown_preset_before_touching_the_lab(
    lab_must_not_be_touched: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["run", NO_FAULT_SCENARIO, "--preset", "zzz"]) == 2

    assert "unknown harness_config_for_preset(s): zzz" in capsys.readouterr().err


def test_bench_rejects_an_unknown_scenario_or_preset_before_touching_the_lab(
    lab_must_not_be_touched: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["bench", "--scenarios", f"{NO_FAULT_SCENARIO},nope"]) == 2
    assert cli.main(["bench", "--presets", "full,zzz"]) == 2

    err = capsys.readouterr().err
    assert "unknown scenario(s): nope" in err
    assert "unknown harness_config_for_preset(s): zzz" in err


# ---------------------------------------------------------------------- lab


def test_lab_up_deploys_waits_for_health_and_reports_it(
    fake_lab: FakeLabManager, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["lab", "up"]) == 0

    assert capsys.readouterr().out.strip() == "lab metro is healthy"
    assert fake_lab.calls == ["deploy", "wait_healthy(150)", "save_baseline"]


def test_lab_up_exits_one_when_the_lab_never_becomes_healthy(
    fake_lab: FakeLabManager, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_lab.health_error = "lab not healthy after 150s; failing: ping ha -> 172.16.0.10"

    assert cli.main(["lab", "up"]) == 1

    assert "lab not healthy after 150s" in capsys.readouterr().err


def test_lab_down_destroys_the_topology(
    fake_lab: FakeLabManager, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["lab", "down"]) == 0

    assert fake_lab.calls == ["destroy"]
    assert "lab metro is destroyed" in capsys.readouterr().out


def test_lab_status_prints_pass_for_every_check_when_healthy(
    fake_lab: FakeLabManager, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["lab", "status"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(TOPOLOGY.health_checks)
    assert all(line.startswith("PASS  ") for line in lines)


def test_lab_status_prints_fail_and_exits_one_when_a_check_fails(
    fake_lab: FakeLabManager, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_lab.lab = ScriptedLab(devices=list(TOPOLOGY.devices), responses={})

    assert cli.main(["lab", "status"]) == 1

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(TOPOLOGY.health_checks)
    assert all(line.startswith("FAIL  ") for line in lines)


# ------------------------------------------------------------------- scenarios


def test_scenarios_lists_each_id_with_difficulty_and_fixable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["scenarios"]) == 0

    rows = capsys.readouterr().out.splitlines()[1:]
    expected = [
        (scenario.id, scenario.difficulty, "yes" if scenario.fixable else "no")
        for scenario in load_all()
    ]
    assert [tuple(row.split()[:3]) for row in rows] == expected
    assert rows[0].split()[0] == "bgp-missing-network"


# ---------------------------------------------------------------------- run


def test_run_solved_exits_zero_and_summarises_the_trial(
    fake_lab: FakeLabManager,
    scripted_model: Callable[[list[LLMResponse]], None],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scripted_model([no_fault_diagnosis()])
    output_directory = tmp_path / "out"

    code = cli.main(["run", NO_FAULT_SCENARIO, "--output-directory", str(output_directory)])

    assert code == 0
    out = capsys.readouterr().out
    assert "Solved: yes" in out
    assert "Root cause: diagnosed -/no_fault, expected -/no_fault (correct)" in out
    assert "Turns: 1" in out
    assert "Changes: applied 0, kept 0, rolled back 0" in out
    assert (output_directory / f"{NO_FAULT_SCENARIO}.full.1.audit.jsonl").is_file()
    assert (output_directory / f"{NO_FAULT_SCENARIO}.full.1.messages.json").is_file()
    assert fake_lab.calls == ["reset", "wait_healthy(30)", "reset"]


def test_run_not_solved_exits_one(
    fake_lab: FakeLabManager,
    scripted_model: Callable[[list[LLMResponse]], None],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scripted_model([wrong_diagnosis()])

    code = cli.main(["run", NO_FAULT_SCENARIO, "--output-directory", str(tmp_path / "out")])

    assert code == 1
    out = capsys.readouterr().out
    assert "Solved: no" in out
    assert "(wrong)" in out


def test_run_fails_fast_with_a_clear_message_when_the_lab_is_unhealthy(
    fake_lab: FakeLabManager,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_lab.health_error = "lab not healthy after 30s; failing: ping ha -> 172.16.0.10"

    def refuse_model() -> None:
        raise AssertionError("no trial may start on an unhealthy lab")

    monkeypatch.setattr(cli, "build_llm_client", refuse_model)

    code = cli.main(["run", NO_FAULT_SCENARIO, "--output-directory", str(tmp_path / "out")])

    assert code == 1
    assert "lab not healthy after 30s" in capsys.readouterr().err
    assert fake_lab.calls == ["reset", "wait_healthy(30)"]


# -------------------------------------------------------------------- bench


def test_bench_runs_the_matrix_and_writes_results_and_report(
    fake_lab: FakeLabManager,
    scripted_model: Callable[[list[LLMResponse]], None],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scripted_model([no_fault_diagnosis(), no_fault_diagnosis()])
    output_directory = tmp_path / "out"

    code = cli.main(
        [
            "bench",
            "--scenarios",
            NO_FAULT_SCENARIO,
            "--presets",
            "full",
            "--trials",
            "2",
            "--output-directory",
            str(output_directory),
        ]
    )

    assert code == 0
    assert len(load_results(output_directory / "results.jsonl")) == 2
    report_text = (output_directory / "report.md").read_text(encoding="utf-8")
    assert "## Summary by preset" in report_text
    assert f"| {NO_FAULT_SCENARIO} | 2/2 |" in report_text
    assert "## Solved per scenario" in capsys.readouterr().out


# ------------------------------------------------------------------- report


def test_report_prints_both_tables_for_an_existing_results_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results_path = tmp_path / "results.jsonl"
    rows = [
        TrialResult("a", "full", 1, "m", fixable=True, solved=True),
        TrialResult("a", "no_subagents", 1, "m", fixable=True, solved=False),
    ]
    results_path.write_text(
        "".join(json.dumps(row.to_dict()) + "\n" for row in rows), encoding="utf-8"
    )

    assert cli.main(["report", str(results_path)]) == 0

    out = capsys.readouterr().out
    assert "## Summary by preset" in out
    assert "| full | 1 |" in out
    assert "| a | 1/1 | 0/1 |" in out


def test_report_exits_one_when_the_results_file_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["report", str(tmp_path / "missing.jsonl")]) == 1

    assert "missing.jsonl" in capsys.readouterr().err
