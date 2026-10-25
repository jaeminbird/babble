from __future__ import annotations

import re
from pathlib import Path

import pytest

from babble.lab.topology import DEFAULT_LABS_DIR, Topology, load_topology
from babble.types import Check

LAB = "metro"


@pytest.fixture(scope="module")
def topo() -> Topology:
    return load_topology(LAB)


def test_loads_metro_from_repo_labs_dir(topo: Topology) -> None:
    assert topo.name == "metro"
    assert topo.dir == DEFAULT_LABS_DIR / "metro"
    assert topo.clab_file == topo.dir / "metro.clab.yml"
    assert topo.clab_file.is_file()


def test_device_roles(topo: Topology) -> None:
    assert topo.routers() == ["r1", "r2", "r3", "ra", "rb", "rdc"]
    assert topo.hosts() == ["ha", "hb", "hdc"]


def test_every_link_endpoint_is_a_device(topo: Topology) -> None:
    for link in topo.links:
        for end in (link["a"], link["b"]):
            device = end.split(":")[0]
            assert device in topo.devices, f"{end} is not a device"


def test_every_router_has_a_config_file(topo: Topology) -> None:
    for router in topo.routers():
        assert (topo.dir / "configs" / f"{router}.conf").is_file(), router


def test_every_router_config_contains_its_loopback(topo: Topology) -> None:
    for router in topo.routers():
        loopback = topo.devices[router]["loopback"]
        text = (topo.dir / "configs" / f"{router}.conf").read_text(encoding="utf-8")
        assert f"ip address {loopback}/32" in text, router
        assert f"hostname {router}" in text, router


def test_clab_file_declares_every_device(topo: Topology) -> None:
    text = topo.clab_file.read_text(encoding="utf-8")
    nodes = re.findall(r"^    (\w+):$", text, flags=re.MULTILINE)
    assert sorted(nodes) == sorted(topo.devices)


def test_every_link_subnet_appears_once(topo: Topology) -> None:
    subnets = [link["subnet"] for link in topo.links]
    assert len(subnets) == len(set(subnets))


def test_health_checks_reference_devices(topo: Topology) -> None:
    assert topo.health_checks, "metro needs baseline health checks"
    for check in topo.health_checks:
        assert check.device in topo.devices
        assert check.kind in ("ping", "bgp_established", "route_present", "ospf_full")


def test_health_checks_cover_baseline_and_sessions(topo: Topology) -> None:
    checks = set(topo.health_checks)
    assert Check("ping", "ha", "172.16.0.10") in checks
    assert Check("ping", "hb", "172.16.0.10") in checks
    assert Check("ping", "ha", "192.168.2.10") in checks
    # Every eBGP session, both ends.
    for device, neighbor in [
        ("r1", "10.1.0.2"),
        ("ra", "10.1.0.1"),
        ("r2", "10.2.0.2"),
        ("rb", "10.2.0.1"),
        ("r3", "10.100.0.2"),
        ("rdc", "10.100.0.1"),
    ]:
        assert Check("bgp_established", device, neighbor) in checks
    # Every iBGP session, both ends.
    core_loopbacks = {"r1": "10.255.0.1", "r2": "10.255.0.2", "r3": "10.255.0.3"}
    for router in core_loopbacks:
        for peer, peer_loopback in core_loopbacks.items():
            if router != peer:
                assert Check("bgp_established", router, peer_loopback) in checks


def test_routers_and_hosts_helpers_on_synthetic_topology(tmp_path: Path) -> None:
    lab = tmp_path / "tiny"
    lab.mkdir()
    (lab / "meta.yaml").write_text(
        "name: tiny\n"
        "devices:\n"
        "  r1: {kind: router, role: core, asn: 65000, loopback: 10.255.0.1}\n"
        "  h1: {kind: host, role: host, ip: 10.0.0.2}\n"
        "links:\n"
        "  - {a: 'r1:eth1', b: 'h1:eth1', subnet: '10.0.0.0/24'}\n"
        "health_checks:\n"
        "  - {kind: ping, device: h1, target: 10.0.0.1}\n",
        encoding="utf-8",
    )
    tiny_topology = load_topology("tiny", labs_dir=tmp_path)
    assert tiny_topology.routers() == ["r1"]
    assert tiny_topology.hosts() == ["h1"]
    assert tiny_topology.health_checks == [Check("ping", "h1", "10.0.0.1")]
    text = tiny_topology.describe()
    assert "r1 (core, AS 65000, loopback 10.255.0.1)" in text
    assert "h1 (host, ip 10.0.0.2)" in text
    assert "r1:eth1 <-> h1:eth1  10.0.0.0/24" in text
    assert "ping h1 -> 10.0.0.1" in text


def test_describe_lists_every_device_and_link(topo: Topology) -> None:
    text = topo.describe()
    for device in topo.devices:
        assert device in text
    assert "r1:eth1 <-> r2:eth1  10.0.12.0/30" in text


def test_missing_lab_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_topology("nope", labs_dir=tmp_path)


@pytest.mark.parametrize("meta_text", ["", "- just\n- a list\n", "scalar\n"])
def test_meta_yaml_that_is_empty_or_not_a_mapping_raises(tmp_path: Path, meta_text: str) -> None:
    lab = tmp_path / "broken"
    lab.mkdir()
    (lab / "meta.yaml").write_text(meta_text, encoding="utf-8")
    with pytest.raises(ValueError, match="broken.*meta.yaml must be a non-empty mapping"):
        load_topology("broken", labs_dir=tmp_path)
