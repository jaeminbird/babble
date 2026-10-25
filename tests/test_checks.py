import json

import pytest

from babble.lab.scripted import ScriptedLab
from babble.netops.checks import evaluate, evaluate_all, evaluate_all_after_convergence
from babble.types import Check, LabError, PingResult

BGP_CMD = "show bgp neighbors 10.2.0.1 json"
ROUTE_CMD = "show ip route 172.16.0.0/24 json"
OSPF_CMD = "show ip ospf neighbor json"

BGP_UP = json.dumps(
    {"10.2.0.1": {"remoteAs": 65002, "bgpState": "Established", "bgpTimerUpMsec": 1000}}
)
BGP_DOWN = json.dumps({"10.2.0.1": {"remoteAs": 65002, "bgpState": "Active"}})
ROUTE_UP = json.dumps(
    {
        "172.16.0.0/24": [
            {"prefix": "172.16.0.0/24", "protocol": "bgp", "selected": True, "nexthops": []}
        ]
    }
)
OSPF_FULL = json.dumps(
    {
        "neighbors": {
            "10.255.0.2": [
                {"nbrState": "Full/DR", "ifaceName": "eth1", "ifaceAddress": "10.0.12.2"}
            ]
        }
    }
)
OSPF_INIT = json.dumps(
    {"neighbors": {"10.255.0.2": [{"nbrState": "Init/DROther", "ifaceName": "eth1"}]}}
)


def lab_with(output_by_command, device="r1", **kwargs):
    """A ScriptedLab where `device` answers each command with a fixed output."""
    responses = {(device, cmd): out for cmd, out in output_by_command.items()}
    return ScriptedLab(devices=[device, "ha"], responses=responses, hosts={"ha"}, **kwargs)


# ---------------------------------------------------------- bgp_established


def test_bgp_established_passes_on_established_session():
    lab = lab_with({BGP_CMD: BGP_UP})
    assert evaluate(Check("bgp_established", "r1", "10.2.0.1"), lab) is True


def test_bgp_established_fails_on_other_state():
    lab = lab_with({BGP_CMD: BGP_DOWN})
    assert evaluate(Check("bgp_established", "r1", "10.2.0.1"), lab) is False


def test_bgp_established_fails_when_neighbor_missing():
    lab = lab_with({BGP_CMD: "{}"})
    assert evaluate(Check("bgp_established", "r1", "10.2.0.1"), lab) is False


def test_bgp_established_fails_on_garbage_output():
    lab = lab_with({BGP_CMD: "% No such neighbor"})
    assert evaluate(Check("bgp_established", "r1", "10.2.0.1"), lab) is False


def test_bgp_established_fails_when_output_is_a_list():
    lab = lab_with({BGP_CMD: "[1, 2, 3]"})
    assert evaluate(Check("bgp_established", "r1", "10.2.0.1"), lab) is False


# ----------------------------------------------------------- route_present


def test_route_present_passes_with_nonempty_route_list():
    lab = lab_with({ROUTE_CMD: ROUTE_UP})
    assert evaluate(Check("route_present", "r1", "172.16.0.0/24"), lab) is True


def test_route_present_fails_on_empty_object():
    lab = lab_with({ROUTE_CMD: "{}"})
    assert evaluate(Check("route_present", "r1", "172.16.0.0/24"), lab) is False


def test_route_present_fails_on_empty_route_list():
    lab = lab_with({ROUTE_CMD: json.dumps({"172.16.0.0/24": []})})
    assert evaluate(Check("route_present", "r1", "172.16.0.0/24"), lab) is False


def test_route_present_fails_on_non_json_output():
    lab = lab_with({ROUTE_CMD: "% Network not in table"})
    assert evaluate(Check("route_present", "r1", "172.16.0.0/24"), lab) is False


# --------------------------------------------------------------- ospf_full


def test_ospf_full_passes_on_full_neighbor():
    lab = lab_with({OSPF_CMD: OSPF_FULL})
    assert evaluate(Check("ospf_full", "r1", "10.255.0.2"), lab) is True


def test_ospf_full_fails_on_init_neighbor():
    lab = lab_with({OSPF_CMD: OSPF_INIT})
    assert evaluate(Check("ospf_full", "r1", "10.255.0.2"), lab) is False


def test_ospf_full_fails_when_router_id_absent():
    lab = lab_with({OSPF_CMD: OSPF_FULL})
    assert evaluate(Check("ospf_full", "r1", "10.255.0.3"), lab) is False


def test_ospf_full_accepts_state_or_converged_fields():
    by_state = json.dumps({"neighbors": {"10.255.0.2": [{"state": "Full"}]}})
    by_converged = json.dumps({"neighbors": {"10.255.0.2": [{"converged": "Full"}]}})
    check = Check("ospf_full", "r1", "10.255.0.2")
    assert evaluate(check, lab_with({OSPF_CMD: by_state})) is True
    assert evaluate(check, lab_with({OSPF_CMD: by_converged})) is True


def test_ospf_full_any_entry_is_enough():
    out = json.dumps(
        {
            "neighbors": {
                "10.255.0.2": [
                    {"nbrState": "Init/DROther"},
                    "not an object",
                    {"nbrState": "Full/Backup"},
                ]
            }
        }
    )
    assert evaluate(Check("ospf_full", "r1", "10.255.0.2"), lab_with({OSPF_CMD: out})) is True


@pytest.mark.parametrize(
    "garbage",
    [
        "",
        "not json",
        "[]",
        json.dumps({"neighbors": []}),
        json.dumps({"neighbors": {"10.255.0.2": "Full"}}),
        json.dumps({"neighbors": {"10.255.0.2": [{"nbrState": 7}]}}),
        json.dumps({"other": {}}),
    ],
)
def test_ospf_full_garbage_fails_without_raising(garbage):
    lab = lab_with({OSPF_CMD: garbage})
    assert evaluate(Check("ospf_full", "r1", "10.255.0.2"), lab) is False


# -------------------------------------------------------------------- ping


def test_ping_passes_when_lab_replies():
    lab = ScriptedLab(devices=["ha"], responses={}, pings={("ha", "172.16.0.10"): True})
    assert evaluate(Check("ping", "ha", "172.16.0.10"), lab) is True


def test_ping_fails_when_lab_does_not_reply():
    lab = ScriptedLab(devices=["ha"], responses={}, pings={("ha", "172.16.0.10"): False})
    assert evaluate(Check("ping", "ha", "172.16.0.10"), lab) is False


def test_ping_follows_lab_fix_state():
    lab = ScriptedLab(
        devices=["r1", "ha"],
        responses={},
        fix={"r1": ["ip route 172.16.0.0/24 10.0.13.2"]},
        pings={("ha", "172.16.0.10"): False},
        fixed_pings={("ha", "172.16.0.10"): True},
        hosts={"ha"},
    )
    check = Check("ping", "ha", "172.16.0.10")
    assert evaluate(check, lab) is False
    lab.configure("r1", ["ip route 172.16.0.0/24 10.0.13.2"])
    assert evaluate(check, lab) is True


# ------------------------------------------------------- never raises


class _ExplodingLab:
    """A backend whose every method fails in a different way."""

    def show(self, device, command):
        raise LabError("container down")

    def ping(self, device, target, count=3):
        raise RuntimeError("boom")


class _BadPingLab:
    def ping(self, device, target, count=3):
        return PingResult(sent=3, received=0, output="")

    def show(self, device, command):
        return None


@pytest.mark.parametrize(
    "check",
    [
        Check("ping", "r1", "10.0.0.1"),
        Check("bgp_established", "r1", "10.2.0.1"),
        Check("route_present", "r1", "172.16.0.0/24"),
        Check("ospf_full", "r1", "10.255.0.2"),
    ],
)
def test_evaluate_never_raises_on_lab_errors(check):
    assert evaluate(check, _ExplodingLab()) is False


def test_evaluate_handles_non_string_show_output():
    check = Check("bgp_established", "r1", "10.2.0.1")
    assert evaluate(check, _BadPingLab()) is False


def test_evaluate_unknown_device_is_false():
    lab = lab_with({BGP_CMD: BGP_UP})
    assert evaluate(Check("bgp_established", "r9", "10.2.0.1"), lab) is False


def test_evaluate_unknown_kind_is_false():
    check = Check("teleport", "r1", "10.0.0.1")  # type: ignore[arg-type]
    assert evaluate(check, lab_with({})) is False


def test_evaluate_on_host_is_false_for_show_checks():
    lab = lab_with({BGP_CMD: BGP_UP})
    assert evaluate(Check("bgp_established", "ha", "10.2.0.1"), lab) is False


# --------------------------------------------------------- evaluate_all


def test_evaluate_all_maps_each_check_to_its_result():
    lab = lab_with({BGP_CMD: BGP_UP, ROUTE_CMD: "{}"})
    up = Check("bgp_established", "r1", "10.2.0.1")
    missing = Check("route_present", "r1", "172.16.0.0/24")
    assert evaluate_all([up, missing], lab) == {up: True, missing: False}


def test_evaluate_all_empty_list():
    assert evaluate_all([], lab_with({})) == {}


def test_evaluate_all_collapses_duplicates_without_error():
    lab = lab_with({BGP_CMD: BGP_UP})
    check = Check("bgp_established", "r1", "10.2.0.1")
    assert evaluate_all([check, check], lab) == {check: True}


def test_after_convergence_with_no_wait_evaluates_once_and_never_sleeps():
    lab = lab_with({BGP_CMD: BGP_DOWN})
    check = Check("bgp_established", "r1", "10.2.0.1")
    sleeps = []
    result = evaluate_all_after_convergence(
        [check], lab, convergence_wait_seconds=0.0, poll_interval_seconds=1.0, sleep=sleeps.append
    )
    assert result == {check: False}
    assert sleeps == []


def test_after_convergence_polls_until_the_checks_pass():
    class ConvergingLab(ScriptedLab):
        """Reports the BGP session down for the first two reads, then up."""

        def __init__(self):
            super().__init__(devices=["r1"], responses={})
            self.reads = 0

        def show(self, device, command):
            self.reads += 1
            return BGP_UP if self.reads > 2 else BGP_DOWN

    lab = ConvergingLab()
    check = Check("bgp_established", "r1", "10.2.0.1")
    sleeps = []
    # A short wait keeps the minimum observation window (also capped by the wait) fast.
    result = evaluate_all_after_convergence(
        [check], lab, convergence_wait_seconds=0.05, poll_interval_seconds=0.01, sleep=sleeps.append
    )
    assert result == {check: True}
    assert lab.reads >= 3
    assert sleeps[:2] == [0.01, 0.01]


def test_after_convergence_returns_the_last_evaluation_when_the_wait_runs_out():
    lab = lab_with({BGP_CMD: BGP_DOWN})
    check = Check("bgp_established", "r1", "10.2.0.1")
    sleeps = []
    result = evaluate_all_after_convergence(
        [check], lab, convergence_wait_seconds=0.05, poll_interval_seconds=0.01, sleep=sleeps.append
    )
    assert result == {check: False}
    assert sleeps
