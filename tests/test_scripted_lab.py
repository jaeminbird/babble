import pytest

from babble.lab.scripted import ScriptedLab
from babble.types import LabError

ROUTE = "show ip route 172.16.0.0/24 json"
ROUTE_JSON = '{"172.16.0.0/24": [{"prefix": "172.16.0.0/24", "protocol": "bgp", "selected": true}]}'


def make_lab(**overrides):
    kwargs = {
        "devices": ["r1", "r2", "ha"],
        "responses": {("r1", ROUTE): "{}", ("r2", "show version"): "FRR 10.7.1"},
        "fixed_responses": {("r1", ROUTE): ROUTE_JSON},
        "fix": {"r1": ["ip route 172.16.0.0/24 10.0.13.2"]},
        "pings": {("ha", "172.16.0.10"): False, ("r1", "10.0.13.2"): True},
        "fixed_pings": {("ha", "172.16.0.10"): True},
        "hosts": {"ha"},
    }
    kwargs.update(overrides)
    return ScriptedLab(**kwargs)


# ------------------------------------------------------------------ state


def test_starts_broken_and_shows_broken_output():
    lab = make_lab()
    assert lab.state == "broken"
    assert lab.show("r1", ROUTE) == "{}"


def test_fixed_after_every_required_line_is_applied():
    lab = make_lab()
    lab.configure("r1", ["ip route 172.16.0.0/24 10.0.13.2"])
    assert lab.state == "fixed"
    assert lab.show("r1", ROUTE) == ROUTE_JSON


def test_fix_is_cumulative_across_calls_and_devices():
    lab = make_lab(
        fix={"r1": ["line a", "line b"], "r2": ["line c"]},
        responses={},
        fixed_responses={},
    )
    lab.configure("r1", ["line a"])
    assert lab.state == "broken"
    lab.configure("r2", ["line c"])
    assert lab.state == "broken"  # r1 still missing "line b"
    lab.configure("r1", ["line b"])
    assert lab.state == "fixed"


def test_fix_lines_compare_after_whitespace_normalization():
    lab = make_lab()
    lab.configure("r1", ["   ip   route  172.16.0.0/24\t10.0.13.2  "])
    assert lab.state == "fixed"


def test_extra_lines_do_not_break_the_fix():
    lab = make_lab()
    lab.configure("r1", ["hostname r1", "ip route 172.16.0.0/24 10.0.13.2"])
    assert lab.state == "fixed"


def test_blank_config_lines_are_dropped():
    lab = make_lab()
    lab.configure("r1", ["", "   "])
    assert lab.applied_lines("r1") == []
    assert lab.config_calls == [("r1", [])]


def test_lab_without_fix_is_never_fixed():
    lab = make_lab(fix={}, fixed_responses={("r1", ROUTE): ROUTE_JSON})
    lab.configure("r1", ["anything"])
    assert lab.state == "broken"
    assert lab.show("r1", ROUTE) == "{}"


def test_repeated_fix_lines_are_applied_once():
    lab = make_lab()
    lab.configure("r1", ["ip route 172.16.0.0/24 10.0.13.2"])
    lab.configure("r1", ["ip route 172.16.0.0/24 10.0.13.2"])
    assert lab.applied_lines("r1") == ["ip route 172.16.0.0/24 10.0.13.2"]


def test_configure_records_normalized_calls_in_order():
    lab = make_lab()
    lab.configure("r2", ["  router   bgp 65001 "])
    lab.configure("r1", ["ip route 172.16.0.0/24 10.0.13.2"])
    assert lab.config_calls == [
        ("r2", ["router bgp 65001"]),
        ("r1", ["ip route 172.16.0.0/24 10.0.13.2"]),
    ]


def test_configure_returns_empty_output():
    assert make_lab().configure("r2", ["x"]) == ""


# ------------------------------------------------------------------- show


def test_show_normalizes_command_whitespace_and_table_keys():
    lab = ScriptedLab(
        devices=["r1"],
        responses={("r1", "  show   ip  route 10.0.0.0/8  "): "ok"},
    )
    assert lab.show("r1", "show ip route 10.0.0.0/8") == "ok"
    assert lab.show("r1", "show  ip route   10.0.0.0/8 ") == "ok"


def test_show_unknown_command_returns_unknown_command_output():
    lab = make_lab()
    out = lab.show("r2", "show ip bgp")
    assert out.startswith("% Unknown command")


def test_show_fixed_table_falls_back_to_responses():
    lab = make_lab()
    lab.configure("r1", ["ip route 172.16.0.0/24 10.0.13.2"])
    assert lab.show("r2", "show version") == "FRR 10.7.1"


def test_show_unknown_device_raises():
    with pytest.raises(LabError, match="unknown device"):
        make_lab().show("r9", ROUTE)


def test_show_on_host_raises_not_a_router():
    with pytest.raises(LabError, match="not a router"):
        make_lab().show("ha", ROUTE)


def test_configure_on_host_raises():
    lab = make_lab()
    with pytest.raises(LabError, match="not a router"):
        lab.configure("ha", ["x"])
    assert lab.config_calls == []


def test_configure_unknown_device_raises():
    with pytest.raises(LabError, match="unknown device"):
        make_lab().configure("r9", ["x"])


# ------------------------------------------------------- snapshot / restore


def test_snapshot_restore_round_trip_moves_back_to_broken():
    lab = make_lab()
    snap = lab.snapshot("r1")
    lab.configure("r1", ["ip route 172.16.0.0/24 10.0.13.2"])
    assert lab.state == "fixed"

    lab.restore("r1", snap)
    assert lab.state == "broken"
    assert lab.applied_lines("r1") == []
    assert lab.show("r1", ROUTE) == "{}"


def test_restore_keeps_other_devices_changes():
    lab = make_lab(fix={})
    lab.configure("r2", ["hostname-like change"])
    snap = lab.snapshot("r1")
    lab.configure("r1", ["ip route 172.16.0.0/24 10.0.13.2"])
    lab.restore("r1", snap)
    assert lab.applied_lines("r2") == ["hostname-like change"]
    assert lab.applied_lines("r1") == []


def test_snapshot_is_a_string_and_stable_until_changed():
    lab = make_lab()
    first = lab.snapshot("r1")
    assert isinstance(first, str)
    assert lab.snapshot("r1") == first
    lab.configure("r1", ["ip route 172.16.0.0/24 10.0.13.2"])
    assert lab.snapshot("r1") != first


def test_restore_rejects_garbage_and_other_devices_snapshots():
    lab = make_lab()
    with pytest.raises(LabError):
        lab.restore("r1", "not json")
    with pytest.raises(LabError):
        lab.restore("r1", "[1, 2]")
    with pytest.raises(LabError):
        lab.restore("r1", lab.snapshot("r2"))


def test_restore_unknown_device_raises():
    with pytest.raises(LabError, match="unknown device"):
        make_lab().restore("r9", "{}")


# ------------------------------------------------------------------- ping


def test_ping_uses_pings_table_and_defaults_to_no_reply():
    lab = make_lab()
    missing_reply = lab.ping("ha", "172.16.0.10")
    assert missing_reply.ok is False
    assert (missing_reply.sent, missing_reply.received) == (3, 0)
    assert lab.ping("ha", "192.168.9.9").ok is False  # missing entry


def test_ping_switches_to_fixed_table_when_fixed():
    lab = make_lab()
    lab.configure("r1", ["ip route 172.16.0.0/24 10.0.13.2"])
    result = lab.ping("ha", "172.16.0.10", count=5)
    assert result.ok is True
    assert (result.sent, result.received) == (5, 5)
    assert "5 packets transmitted" in result.output


def test_ping_from_router_uses_pings_table():
    assert make_lab().ping("r1", "10.0.13.2").ok is True


def test_ping_unknown_device_raises():
    with pytest.raises(LabError):
        make_lab().ping("r9", "10.0.0.1")


# ------------------------------------------------------------- host exec


def test_host_exec_is_recorded_and_changes_nothing():
    lab = make_lab()
    assert lab.host_exec("ha", ["ip", "link", "set", "eth1", "down"]) == ""
    assert lab.host_calls == [("ha", ["ip", "link", "set", "eth1", "down"])]
    assert lab.state == "broken"


def test_host_exec_unknown_device_raises():
    with pytest.raises(LabError):
        make_lab().host_exec("nope", ["ls"])


# ------------------------------------------------------------ construction


def test_devices_returns_declared_order():
    assert make_lab().devices() == ["r1", "r2", "ha"]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"devices": ["r1", "r1"]}, "duplicate"),
        ({"hosts": {"zz"}}, "hosts not in devices"),
        ({"responses": {("zz", "show x"): "y"}}, "unknown device"),
        ({"fixed_responses": {("zz", "show x"): "y"}}, "unknown device"),
        ({"pings": {("zz", "1.1.1.1"): True}}, "unknown device"),
        ({"fix": {"zz": ["line"]}}, "unknown device"),
    ],
)
def test_bad_construction_fails_fast(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_lab(**overrides)


def test_fix_with_an_empty_line_list_for_a_device_raises() -> None:
    with pytest.raises(ValueError, match="fix for 'r1' is empty"):
        make_lab(fix={"r1": []})
