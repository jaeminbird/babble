from __future__ import annotations

import ipaddress
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

import babbler.bench.scenario
from babbler.bench.scenario import (
    SCENARIOS_DIR,
    InjectStep,
    RootCause,
    Scenario,
    inject,
    load_all,
    load_scenario,
    revert_host,
)
from babbler.lab.scripted import ScriptedLab
from babbler.lab.topology import Topology, load_topology
from babbler.types import FAULT_CATEGORIES

EXPECTED_IDS = [
    "bgp-missing-network",
    "bgp-outbound-filter",
    "bgp-wrong-neighbor-ip",
    "bgp-wrong-remote-as",
    "edge-link-shutdown",
    "ip-misconfig",
    "link-degradation",
    "next-hop-self-missing",
    "no-fault-control",
    "ospf-loopback-missing",
    "ospf-timer-mismatch",
    "static-blackhole",
]

# Category words the alert must never leak, e.g. "bgp", "neighbor", "misconfig" for
# bgp_neighbor_misconfig. Generic words that appear in ordinary prose are excluded.
GENERIC_WORDS = {"or", "no", "other"}


@pytest.fixture(scope="module")
def topo() -> Topology:
    return load_topology("metro")


@pytest.fixture(scope="module")
def scenarios() -> dict[str, Scenario]:
    return {s.id: s for s in load_all()}


def _lab(topo: Topology) -> ScriptedLab:
    return ScriptedLab(devices=list(topo.devices), responses={}, hosts=set(topo.hosts()))


# ------------------------------------------------------------- bundled scenarios


def test_loads_all_twelve_in_id_order() -> None:
    scenarios = load_all()
    assert [s.id for s in scenarios] == EXPECTED_IDS


def test_default_dir_is_the_package_scenarios_dir() -> None:
    assert SCENARIOS_DIR == Path(babbler.bench.scenario.__file__).resolve().parent / "scenarios"
    assert SCENARIOS_DIR.is_dir()
    assert [s.id for s in load_all(SCENARIOS_DIR)] == EXPECTED_IDS


def test_id_matches_file_stem() -> None:
    for path in sorted(SCENARIOS_DIR.glob("*.yaml")):
        assert load_scenario(path).id == path.stem, path.name


def test_metro_topology_and_known_devices(topo: Topology, scenarios: dict[str, Scenario]) -> None:
    for scenario in scenarios.values():
        assert scenario.topology == "metro", scenario.id
        for step in scenario.inject:
            assert step.device in topo.devices, f"{scenario.id}: inject device {step.device}"
        for check in scenario.checks:
            assert check.device in topo.devices, f"{scenario.id}: check device {check.device}"
        if scenario.root_cause.device:
            assert scenario.root_cause.device in topo.devices, scenario.id


def test_config_steps_target_routers_and_host_steps_target_real_devices(
    topo: Topology, scenarios: dict[str, Scenario]
) -> None:
    for scenario in scenarios.values():
        for step in scenario.inject:
            if step.config is not None:
                assert step.device in topo.routers(), f"{scenario.id}: config on non-router"
            else:
                assert step.device in topo.devices, scenario.id


def test_non_ping_checks_target_routers(topo: Topology, scenarios: dict[str, Scenario]) -> None:
    for scenario in scenarios.values():
        for check in scenario.checks:
            if check.kind != "ping":
                assert check.device in topo.routers(), f"{scenario.id}: {check.describe()}"


def test_check_targets_are_valid_addresses_or_prefixes(scenarios: dict[str, Scenario]) -> None:
    for scenario in scenarios.values():
        for check in scenario.checks:
            if check.kind == "route_present":
                ipaddress.ip_network(check.target)
            else:
                ipaddress.ip_address(check.target)


def test_checks_are_nonempty_and_use_known_kinds(scenarios: dict[str, Scenario]) -> None:
    kinds = {"ping", "bgp_established", "route_present", "ospf_full"}
    for scenario in scenarios.values():
        assert scenario.checks, scenario.id
        assert {c.kind for c in scenario.checks} <= kinds, scenario.id


def test_fixable_scenarios_inject_something_and_no_fault_injects_nothing(
    scenarios: dict[str, Scenario],
) -> None:
    for scenario in scenarios.values():
        if scenario.fixable:
            assert scenario.inject, f"{scenario.id} is fixable but injects nothing"
    assert scenarios["no-fault-control"].inject == []
    assert scenarios["no-fault-control"].root_cause.category == "no_fault"


def test_only_link_degradation_is_unfixable_by_config(scenarios: dict[str, Scenario]) -> None:
    unfixable = sorted(s.id for s in scenarios.values() if not s.fixable)
    assert unfixable == ["link-degradation", "no-fault-control"]
    assert scenarios["link-degradation"].inject[0].host is not None


def test_difficulty_and_categories_are_valid(scenarios: dict[str, Scenario]) -> None:
    for scenario in scenarios.values():
        assert scenario.difficulty in ("easy", "medium", "hard"), scenario.id
        assert scenario.root_cause.category in FAULT_CATEGORIES, scenario.id


def test_root_causes_match_the_plan(scenarios: dict[str, Scenario]) -> None:
    expected = {
        "edge-link-shutdown": ("r3", "interface_down", "easy"),
        "bgp-wrong-remote-as": ("rb", "bgp_neighbor_misconfig", "easy"),
        "bgp-wrong-neighbor-ip": ("r3", "bgp_neighbor_misconfig", "medium"),
        "bgp-outbound-filter": ("ra", "bgp_policy", "medium"),
        "bgp-missing-network": ("rdc", "bgp_policy", "easy"),
        "ospf-loopback-missing": ("r3", "ospf_misconfig", "hard"),
        "ospf-timer-mismatch": ("r1", "ospf_misconfig", "medium"),
        "ip-misconfig": ("r2", "ip_misconfig", "medium"),
        "static-blackhole": ("r1", "static_route", "medium"),
        "next-hop-self-missing": ("r3", "bgp_neighbor_misconfig", "hard"),
        "link-degradation": ("r3", "link_degradation", "hard"),
        "no-fault-control": ("", "no_fault", "medium"),
    }
    for sid, (device, category, difficulty) in expected.items():
        s = scenarios[sid]
        assert (s.root_cause.device, s.root_cause.category, s.difficulty) == (
            device,
            category,
            difficulty,
        ), sid


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in GENERIC_WORDS}


def test_alert_never_names_the_root_cause(scenarios: dict[str, Scenario]) -> None:
    for scenario in scenarios.values():
        alert = scenario.alert.lower()
        device = scenario.root_cause.device
        if device:
            assert not re.search(rf"\b{re.escape(device.lower())}\b", alert), (
                f"{scenario.id}: alert names device {device}"
            )
        category_words = _words(scenario.root_cause.category.replace("_", " "))
        leaked = category_words & _words(scenario.alert)
        assert not leaked, f"{scenario.id}: alert contains category words {sorted(leaked)}"


def test_every_scenario_has_a_title_and_alert(scenarios: dict[str, Scenario]) -> None:
    for scenario in scenarios.values():
        assert scenario.title.strip(), scenario.id
        assert scenario.alert.strip(), scenario.id


# ------------------------------------------------------------ inject and revert


def test_inject_config_steps_apply_in_order(scenarios: dict[str, Scenario], topo: Topology) -> None:
    lab = _lab(topo)
    inject(scenarios["bgp-wrong-neighbor-ip"], lab)
    assert lab.config_calls == [
        (
            "r3",
            [
                "router bgp 65000",
                "no neighbor 10.100.0.2",
                "neighbor 10.100.0.6 remote-as 65100",
            ],
        )
    ]
    assert lab.host_calls == []


def test_inject_multiple_steps_in_declared_order(topo: Topology) -> None:
    scenario = Scenario(
        id="ordered",
        title="Two steps",
        topology="metro",
        difficulty="easy",
        alert="Users cannot reach the data center server 172.16.0.10.",
        inject=[
            InjectStep(device="r2", config=["interface eth1", "shutdown"]),
            InjectStep(device="r1", config=["interface eth1", "shutdown"]),
        ],
        checks=[],
        root_cause=RootCause(device="r1", category="interface_down"),
    )
    lab = _lab(topo)
    inject(scenario, lab)
    assert [device for device, _ in lab.config_calls] == ["r2", "r1"]


def test_inject_host_step_goes_to_host_exec_and_revert_undoes_it(
    scenarios: dict[str, Scenario], topo: Topology
) -> None:
    lab = _lab(topo)
    degraded = scenarios["link-degradation"]
    inject(degraded, lab)
    assert lab.config_calls == []
    assert lab.host_calls == [
        ("r3", ["tc", "qdisc", "add", "dev", "eth3", "root", "netem", "loss", "40%"])
    ]

    revert_host(degraded, lab)
    assert lab.host_calls[-1] == ("r3", ["tc", "qdisc", "del", "dev", "eth3", "root"])
    assert len(lab.host_calls) == 2


def test_revert_host_skips_config_only_scenarios(
    scenarios: dict[str, Scenario], topo: Topology
) -> None:
    lab = _lab(topo)
    inject(scenarios["bgp-wrong-remote-as"], lab)
    revert_host(scenarios["bgp-wrong-remote-as"], lab)
    assert lab.host_calls == []


def test_inject_no_fault_control_touches_nothing(
    scenarios: dict[str, Scenario], topo: Topology
) -> None:
    lab = _lab(topo)
    inject(scenarios["no-fault-control"], lab)
    revert_host(scenarios["no-fault-control"], lab)
    assert lab.config_calls == []
    assert lab.host_calls == []
    assert lab.state == "broken"  # no fix table, so the lab never reports fixed


def test_injecting_a_config_fault_leaves_lab_state_changed(
    scenarios: dict[str, Scenario], topo: Topology
) -> None:
    lab = ScriptedLab(
        devices=list(topo.devices),
        responses={},
        hosts=set(topo.hosts()),
        fix={"rb": ["neighbor 10.2.0.1 remote-as 65000"]},
    )
    inject(scenarios["bgp-wrong-remote-as"], lab)
    assert lab.applied_lines("rb") == ["router bgp 65002", "neighbor 10.2.0.1 remote-as 65010"]
    assert lab.state == "broken"


# ------------------------------------------------------------ validation errors


def _base() -> dict[str, Any]:
    return {
        "id": "bad",
        "title": "A bad scenario",
        "topology": "metro",
        "difficulty": "easy",
        "fixable": True,
        "alert": "Users at site B cannot reach the data center server 172.16.0.10.",
        "inject": [{"device": "rb", "config": ["router bgp 65002"]}],
        "checks": [{"kind": "ping", "device": "hb", "target": "172.16.0.10"}],
        "root_cause": {"device": "rb", "category": "bgp_neighbor_misconfig"},
    }


def _load_raw(tmp_path: Path, data: Any, name: str = "bad.yaml") -> Scenario:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_scenario(path)


def test_valid_base_loads(tmp_path: Path) -> None:
    scenario = _load_raw(tmp_path, _base())
    assert scenario.id == "bad"
    assert scenario.checks[0].kind == "ping"


@pytest.mark.parametrize(
    "mutate, problem",
    [
        (lambda d: d.pop("alert"), "missing keys"),
        (lambda d: d.update(difficulty="extreme"), "difficulty"),
        (lambda d: d.update(root_cause={"device": "rb", "category": "cosmic_rays"}), "category"),
        (lambda d: d.update(fixable="yes"), "fixable"),
        (lambda d: d.update(extra=1), "unknown keys"),
        (lambda d: d["inject"][0].update(host=["ls"]), "exactly one of config or host"),
        (lambda d: d["inject"][0].pop("config"), "exactly one of config or host"),
        (lambda d: d["inject"][0].update(revert=["ls"]), "revert is only allowed with host"),
        (lambda d: d["inject"][0].update(surprise=True), "unknown keys"),
        (lambda d: d["checks"][0].update(kind="traceroute"), "check kind"),
        (lambda d: d["checks"][0].update(target="not-an-address"), "not a valid address"),
        (
            lambda d: d["checks"][0].update(kind="route_present", target="172.16.0.0/33"),
            "not a valid address",
        ),
        (lambda d: d.update(checks=["hb"]), "each check must be a mapping"),
        (lambda d: d.update(difficulty=None), "difficulty"),
        (lambda d: d["root_cause"].update(device=""), "may be empty only for category no_fault"),
        (lambda d: d.update(inject="rb"), "inject must be a list"),
    ],
)
def test_malformed_scenarios_are_rejected_with_file_name(
    tmp_path: Path, mutate: Any, problem: str
) -> None:
    data = _base()
    mutate(data)
    path = tmp_path / "broken-case.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(problem)) as info:
        load_scenario(path)
    assert "broken-case.yaml" in str(info.value)


def test_top_level_must_be_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top level must be a mapping"):
        load_scenario(path)


def test_invalid_yaml_is_reported_with_file_name(tmp_path: Path) -> None:
    path = tmp_path / "syntax.yaml"
    path.write_text("id: [unclosed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="syntax.yaml"):
        load_scenario(path)


def test_load_all_on_a_directory_with_a_bad_file_names_the_file(tmp_path: Path) -> None:
    _load_raw(tmp_path, _base(), name="good-one.yaml")
    bad = _base()
    bad["difficulty"] = "impossible"
    (tmp_path / "bad-one.yaml").write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="bad-one.yaml"):
        load_all(tmp_path)


def test_no_fault_may_have_an_empty_root_device(tmp_path: Path) -> None:
    data = _base()
    data["root_cause"] = {"device": "", "category": "no_fault"}
    data["inject"] = []
    data["fixable"] = False
    scenario = _load_raw(tmp_path, data)
    assert scenario.root_cause.device == ""
    assert scenario.inject == []
