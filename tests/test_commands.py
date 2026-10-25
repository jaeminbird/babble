import pytest

from faultline.netops.commands import MAX_SHOW_LENGTH, blocked_config, validate_show


@pytest.mark.parametrize(
    "command",
    [
        "show ip bgp summary json",
        "  show ip route 172.16.0.0/24 json  ",
        "SHOW ip ospf neighbor json",
        "Show running-config",
        "show bgp neighbors 10.0.12.2 json",
    ],
)
def test_valid_show_commands(command):
    assert validate_show(command) is None


@pytest.mark.parametrize(
    "command, fragment",
    [
        ("", "must start with 'show '"),
        ("   ", "must start with 'show '"),
        ("show", "must start with 'show '"),
        ("configure terminal", "must start with 'show '"),
        ("ip route 0.0.0.0/0 10.0.0.1", "must start with 'show '"),
        ("showip route", "must start with 'show '"),
        ("show ip route | include x", "forbidden character '|'"),
        ("show ip route; reload", "forbidden character ';'"),
        ("show ip route && reload", "forbidden character '&'"),
        ("show ip route $HOME", "forbidden character '$'"),
        ("show ip route `id`", "forbidden character '`'"),
        ("show ip route > /tmp/x", "forbidden character '>'"),
        ("show ip route < /etc/passwd", "forbidden character '<'"),
        ("show ip route\nconfigure terminal", "forbidden character '\\n'"),
        ("show ip route\rconfigure terminal", "forbidden character '\\r'"),
        ("show ip route ../../etc", "forbidden sequence '..'"),
        ("show " + "a" * MAX_SHOW_LENGTH, "longer than"),
    ],
)
def test_rejected_show_commands(command, fragment):
    error = validate_show(command)
    assert error is not None
    assert fragment in error


def test_show_at_length_limit_is_allowed():
    command = "show " + "a" * (MAX_SHOW_LENGTH - len("show "))
    assert len(command) == MAX_SHOW_LENGTH
    assert validate_show(command) is None


@pytest.mark.parametrize(
    "line",
    [
        "no router bgp 65000",
        "no router ospf",
        "no interface eth1",
        "hostname r9",
        "write memory",
        "copy running-config startup-config",
        "erase startup-config",
        "reload",
        "no ip forwarding",
        "password hunter2",
        "enable password hunter2",
        "line vty",
        "username admin privilege 15",
        "  NO ROUTER BGP 65000  ",
        "  Hostname r9",
    ],
)
def test_blocked_config_lines(line):
    reasons = blocked_config([line])
    assert len(reasons) == 1
    assert line.strip() in reasons[0]


@pytest.mark.parametrize(
    "line",
    [
        "router bgp 65000",
        "neighbor 10.0.12.2 remote-as 65000",
        "interface eth1",
        "ip route 172.16.0.0/24 10.0.12.2",
        "no shutdown",
        "ip address 10.0.12.1/30",
        "router ospf",
        "network 10.0.12.0/30 area 0",
        "",
    ],
)
def test_allowed_config_lines(line):
    assert blocked_config([line]) == []


def test_blocked_config_empty_input():
    assert blocked_config([]) == []


def test_blocked_config_one_reason_per_blocked_line():
    lines = [
        "router bgp 65000",
        "no router bgp 65000",
        "neighbor 10.0.12.2 remote-as 65000",
        "hostname r9",
        "no shutdown",
    ]
    reasons = blocked_config(lines)
    assert len(reasons) == 2
    assert "no router bgp 65000" in reasons[0]
    assert "hostname r9" in reasons[1]


def test_blocked_config_does_not_match_mid_line():
    # Patterns are anchored: a dangerous word later in the line is not a match.
    assert blocked_config(["description do not reload"]) == []
    assert blocked_config(["neighbor 1.1.1.1 description hostname"]) == []
