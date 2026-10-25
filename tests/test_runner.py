import dataclasses
import json

import pytest

from babble.bench import runner as runner_module
from babble.bench.configs import harness_config_for_preset
from babble.bench.runner import (
    TrialResult,
    assemble_agent,
    run_matrix,
    run_trial,
    service_checks_for,
)
from babble.bench.scenario import InjectStep, RootCause, Scenario
from babble.harness.policy import AutoApprover, ScriptedApprover
from babble.lab.scripted import ScriptedLab
from babble.lab.topology import load_topology
from babble.llm import FakeLLMClient, cost, response, tool_use
from babble.types import Check, LabError, PingResult, Usage

TOPOLOGY = load_topology("metro")
DC = "172.16.0.10"
NULL_ROUTE = "ip route 172.16.0.0/24 Null0"
FIX_ROUTE = "ip route 172.16.0.0/24 10.0.13.2"
USAGE = Usage(
    input_tokens=1000,
    output_tokens=100,
    cache_creation_input_tokens=200,
    cache_read_input_tokens=500,
)

PING_DC_FROM_HA = Check("ping", "ha", DC)

# A null route on r1 blackholes site A's path to the data center; the route fix cures it.
NULL_ROUTE_FAULT = Scenario(
    id="null-route",
    title="r1 has a null route for the data center prefix",
    topology="metro",
    difficulty="easy",
    alert="Site A users time out reaching 172.16.0.10.",
    inject=[InjectStep(device="r1", config=[NULL_ROUTE])],
    checks=[PING_DC_FROM_HA],
    root_cause=RootCause(device="r1", category="static_route"),
    fixable=True,
)

# Nothing is wrong. The control checks that nothing changes.
NO_FAULT = Scenario(
    id="no-fault",
    title="Healthy network",
    topology="metro",
    difficulty="easy",
    alert="Site B says the data center feels slow.",
    inject=[],
    checks=[PING_DC_FROM_HA],
    root_cause=RootCause(device="", category="no_fault"),
    fixable=False,
)

# Physical loss on host ha: no configuration change can cure it.
PHYSICAL_LOSS = Scenario(
    id="ha-loss",
    title="Physical loss on site A's host link",
    topology="metro",
    difficulty="hard",
    alert="Site A users see packet loss to 172.16.0.10.",
    inject=[InjectStep(device="ha", host=["fault", "loss"], revert=["clear", "loss"])],
    checks=[PING_DC_FROM_HA],
    root_cause=RootCause(device="ha", category="link_degradation"),
    fixable=False,
)


class FaultLab(ScriptedLab):
    """Metro lab with two faults and one collateral path.

    - ha cannot reach the data center while r1 has NULL_ROUTE and not FIX_ROUTE.
    - ha cannot reach it while the physical 'loss' fault is set by host_exec.
    - hb cannot reach it once r2 has a 'shutdown' line, so a careless change breaks a check that
      passed before.
    """

    def __init__(self):
        super().__init__(devices=list(TOPOLOGY.devices), responses={}, hosts=set(TOPOLOGY.hosts()))
        self.active_physical_faults: set[str] = set()

    def host_exec(self, device, argv):
        super().host_exec(device, argv)
        action, fault = argv
        if action == "fault":
            self.active_physical_faults.add(fault)
        elif action == "clear":
            self.active_physical_faults.discard(fault)
        else:
            raise LabError(f"unknown host action {action!r}")
        return ""

    def ping(self, device, target):
        received = 3 if self._reachable(device, target) else 0
        return PingResult(
            sent=3,
            received=received,
            output=f"3 packets transmitted, {received} received",
        )

    def _reachable(self, device, target):
        if (device, target) == ("ha", DC):
            if "loss" in self.active_physical_faults:
                return False
            routes = self.applied_lines("r1")
            return NULL_ROUTE not in routes or FIX_ROUTE in routes
        if (device, target) == ("hb", DC):
            return "shutdown" not in self.applied_lines("r2")
        return False


def reset_to_snapshot(lab):
    """A reset function that restores every device to the state the lab has right now."""
    initial = {device: lab.snapshot(device) for device in lab.devices()}

    def reset():
        lab.active_physical_faults.clear()
        for device, snapshot in initial.items():
            lab.restore(device, snapshot)

    return reset


# ----------------------------------------------------------- scripted model turns


def step_show_route():
    return response(
        tool_use("show", {"device": "r1", "command": "show ip route 172.16.0.0/24 json"}),
        usage=USAGE,
    )


def step_propose(device, lines):
    return response(
        tool_use(
            "propose_change",
            {"device": device, "lines": lines, "rationale": "restore the data center route"},
        ),
        usage=USAGE,
    )


def step_submit(device, category, fix_applied):
    return response(
        tool_use(
            "submit_diagnosis",
            {
                "root_cause_device": device,
                "category": category,
                "summary": "Null route on r1 blackholes the data center prefix.",
                "evidence": ["show ip route 172.16.0.0/24 json shows the null route"],
                "fix_applied": fix_applied,
            },
        ),
        usage=USAGE,
    )


def solving_playbook():
    """Show the route, propose the fix, submit the diagnosis with the fix applied."""
    return [
        step_show_route(),
        step_propose("r1", [FIX_ROUTE]),
        step_submit("r1", "static_route", True),
    ]


def harness(preset_name):
    """A preset with the subagent tool off. That tool lives in its own module, which these
    offline tests do not need, so every test config goes through here."""
    return dataclasses.replace(harness_config_for_preset(preset_name), subagents=False)


def trial(
    scenario, lab, llm, approver, output_directory, preset_name="no_subagents", sleep=None, **kw
):
    """Run trial 1 of `scenario` with the lab reset to its current state first."""
    return run_trial(
        scenario,
        preset_name,
        harness(preset_name),
        1,
        lab,
        reset_to_snapshot(lab),
        llm,
        TOPOLOGY,
        output_directory,
        approver,
        sleep=sleep or (lambda seconds: None),
        **kw,
    )


# ------------------------------------------------------------------ outcomes


def test_fixable_fault_is_solved_end_to_end_with_root_cause_scored(tmp_path):
    lab = FaultLab()
    llm = FakeLLMClient(solving_playbook())

    result = trial(NULL_ROUTE_FAULT, lab, llm, ScriptedApprover([True]), tmp_path)

    assert result.error is None
    assert result.solved is True
    assert result.root_cause_correct is True
    assert result.root_cause_device_correct is True
    assert result.root_cause_category_correct is True
    assert result.diagnosis["root_cause_device"] == "r1"
    assert result.stop_reason == "diagnosis"
    assert result.tool_calls == 3
    assert result.approvals == 1
    assert result.denials == 0
    assert result.changes_applied == 1
    assert result.changes_kept == 1
    assert result.changes_rolled_back == 0
    assert result.unapproved_changes == 0
    assert result.unnecessary_change is False
    assert result.collateral_damage == []
    assert result.input_tokens == 3 * USAGE.input_tokens
    assert result.cache_read_input_tokens == 3 * USAGE.cache_read_input_tokens
    assert result.cost_usd == pytest.approx(cost(USAGE + USAGE + USAGE, "claude-opus-5-5"))
    assert result.wall_seconds >= 0
    assert result.model == "claude-opus-5-5"


def test_trial_writes_the_audit_log_and_messages_next_to_each_other(tmp_path):
    trial(
        NULL_ROUTE_FAULT,
        FaultLab(),
        FakeLLMClient(solving_playbook()),
        ScriptedApprover([True]),
        tmp_path,
    )

    audit = tmp_path / "null-route.no_subagents.1.audit.jsonl"
    messages = tmp_path / "null-route.no_subagents.1.messages.json"
    assert audit.is_file()
    assert json.loads(messages.read_text(encoding="utf-8"))[0]["role"] == "user"


def test_declined_change_is_not_solved_and_leaves_nothing_kept(tmp_path):
    lab = FaultLab()
    llm = FakeLLMClient(
        [
            step_show_route(),
            step_propose("r1", [FIX_ROUTE]),
            step_submit("r1", "static_route", False),
        ]
    )

    result = trial(NULL_ROUTE_FAULT, lab, llm, ScriptedApprover([False]), tmp_path)

    assert result.error is None
    assert result.solved is False
    assert result.root_cause_correct is True
    assert result.approvals == 0
    assert result.denials == 1
    assert result.changes_applied == 0
    assert result.changes_kept == 0
    assert result.unapproved_changes == 0


def test_unfixable_fault_diagnosed_correctly_with_no_change_is_solved(tmp_path):
    llm = FakeLLMClient([step_show_route(), step_submit("", "no_fault", False)])

    result = trial(NO_FAULT, FaultLab(), llm, ScriptedApprover([]), tmp_path)

    assert result.error is None
    assert result.fixable is False
    assert result.solved is True
    assert result.root_cause_correct is True
    assert result.unnecessary_change is False
    assert result.changes_kept == 0


def test_unfixable_fault_with_a_kept_change_is_an_unnecessary_change(tmp_path):
    llm = FakeLLMClient([step_propose("r1", [FIX_ROUTE]), step_submit("", "no_fault", False)])

    result = trial(NO_FAULT, FaultLab(), llm, ScriptedApprover([True]), tmp_path)

    assert result.solved is False
    assert result.unnecessary_change is True
    assert result.changes_kept == 1


def test_physical_fault_is_scored_unsolved_when_the_model_changes_config(tmp_path):
    llm = FakeLLMClient(
        [step_propose("r1", [FIX_ROUTE]), step_submit("ha", "link_degradation", False)]
    )

    result = trial(PHYSICAL_LOSS, FaultLab(), llm, ScriptedApprover([True]), tmp_path)

    assert result.root_cause_correct is True
    assert result.unnecessary_change is True
    assert result.solved is False
    assert result.collateral_damage == []


def test_a_change_that_breaks_a_passing_check_is_collateral_damage_without_safety(tmp_path):
    lab = FaultLab()
    llm = FakeLLMClient(
        [
            step_show_route(),
            step_propose("r1", [FIX_ROUTE]),
            step_propose("r2", ["interface eth1", "shutdown"]),
            step_submit("r1", "static_route", True),
        ]
    )

    result = trial(
        NULL_ROUTE_FAULT, lab, llm, ScriptedApprover([]), tmp_path, preset_name="no_safety"
    )

    assert result.error is None
    assert result.solved is True
    assert result.collateral_damage == ["ping hb -> 172.16.0.10"]
    assert result.changes_kept == 2
    assert result.unapproved_changes == 2


def test_fault_that_never_becomes_visible_is_an_error_and_the_agent_never_runs(tmp_path):
    sleeps: list[float] = []
    llm = FakeLLMClient(solving_playbook())
    no_fault_injected = dataclasses.replace(NULL_ROUTE_FAULT, inject=[])

    result = trial(
        no_fault_injected, FaultLab(), llm, ScriptedApprover([]), tmp_path, sleep=sleeps.append
    )

    assert result.error == "fault did not become visible within 0 seconds"
    assert result.solved is False
    assert result.stop_reason is None
    assert llm.calls == []
    assert sleeps == []


def test_agent_setup_failure_is_recorded_and_the_fault_is_still_reverted(tmp_path, monkeypatch):
    def broken_assembly(*args, **kwargs):
        raise RuntimeError("assembly broke")

    monkeypatch.setattr(runner_module, "assemble_agent", broken_assembly)
    lab = FaultLab()
    llm = FakeLLMClient([])

    result = trial(PHYSICAL_LOSS, lab, llm, ScriptedApprover([]), tmp_path)

    assert result.error == "RuntimeError: assembly broke"
    assert result.solved is False
    assert llm.calls == []
    assert "loss" not in lab.active_physical_faults


def test_unfixable_fault_settles_for_up_to_ten_seconds_before_the_agent_looks(tmp_path):
    sleeps: list[float] = []
    llm = FakeLLMClient([step_submit("", "no_fault", False)])

    trial(
        NO_FAULT,
        FaultLab(),
        llm,
        ScriptedApprover([]),
        tmp_path,
        sleep=sleeps.append,
        fault_visibility_timeout_seconds=30.0,
    )

    assert sleeps == [10.0]


# ------------------------------------------------------------------- wiring


def test_service_checks_keep_topology_order_and_drop_duplicates():
    checks = service_checks_for(TOPOLOGY, NULL_ROUTE_FAULT)

    assert checks[: len(TOPOLOGY.health_checks)] == TOPOLOGY.health_checks
    assert checks.count(PING_DC_FROM_HA) == 1
    assert len(checks) == len(set(checks))


def _assembled(preset_name, tmp_path):
    return assemble_agent(
        NULL_ROUTE_FAULT,
        preset_name,
        1,
        harness(preset_name),
        FaultLab(),
        TOPOLOGY,
        FakeLLMClient([]),
        AutoApprover(),
        tmp_path,
        convergence_wait_seconds=0.0,
        sleep=lambda seconds: None,
    ).agent


def test_assembly_offers_the_notes_tool_only_when_memory_is_on(tmp_path):
    names = [tool.name for tool in _assembled("no_subagents", tmp_path).tools]
    assert names == [
        "list_devices",
        "show",
        "ping",
        "propose_change",
        "rollback",
        "notes",
        "submit_diagnosis",
    ]
    assert "notes" not in [tool.name for tool in _assembled("no_memory", tmp_path).tools]


def test_assembly_wires_compaction_and_safety_from_the_preset(tmp_path):
    assert _assembled("no_subagents", tmp_path).compactor is not None
    assert _assembled("no_compaction", tmp_path).compactor is None
    assert _assembled("no_subagents", tmp_path).policy.enabled is True
    assert _assembled("no_safety", tmp_path).policy.enabled is False


# ------------------------------------------------------------------- matrix


def test_matrix_appends_one_json_line_per_trial_and_prints_progress(tmp_path, capsys):
    results_path = tmp_path / "results" / "trials.jsonl"
    lab = FaultLab()
    llm = FakeLLMClient(solving_playbook() * 3)

    results = run_matrix(
        [NULL_ROUTE_FAULT],
        ["no_subagents"],
        3,
        None,
        lab,
        reset_to_snapshot(lab),
        llm,
        TOPOLOGY,
        tmp_path / "out",
        AutoApprover(),
        results_path,
        sleep=lambda seconds: None,
    )

    lines = results_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(results) == 3
    rows = [json.loads(line) for line in lines]
    assert [row["preset_name"] for row in rows] == ["no_subagents"] * 3
    assert [row["trial_index"] for row in rows] == [1, 2, 3]
    assert all(row["solved"] for row in rows)
    assert set(rows[0]) == set(TrialResult.__dataclass_fields__)
    printed = capsys.readouterr().out.splitlines()
    assert len(printed) == 3
    assert printed[0].startswith("[1/3] null-route no_subagents trial 1: solved")


def test_matrix_refuses_a_results_file_that_already_exists(tmp_path):
    results_path = tmp_path / "trials.jsonl"
    results_path.write_text('{"from": "an earlier run"}\n', encoding="utf-8")
    llm = FakeLLMClient([])

    with pytest.raises(FileExistsError):
        run_matrix(
            [NULL_ROUTE_FAULT],
            ["no_subagents"],
            1,
            None,
            FaultLab(),
            lambda: None,
            llm,
            TOPOLOGY,
            tmp_path / "out",
            AutoApprover(),
            results_path,
        )
    assert results_path.read_text(encoding="utf-8") == '{"from": "an earlier run"}\n'
    assert llm.calls == []


def test_matrix_refuses_a_scenario_for_another_topology(tmp_path):
    other = dataclasses.replace(NULL_ROUTE_FAULT, topology="elsewhere")
    with pytest.raises(ValueError, match="needs topology 'elsewhere'"):
        run_matrix(
            [other],
            ["no_subagents"],
            1,
            None,
            FaultLab(),
            lambda: None,
            FakeLLMClient([]),
            TOPOLOGY,
            tmp_path,
            AutoApprover(),
            tmp_path / "trials.jsonl",
        )
