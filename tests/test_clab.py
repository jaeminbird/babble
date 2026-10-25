from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from faultline.lab import clab
from faultline.lab.clab import RESTORE_FILE, ClabLab, CommandRunner
from faultline.lab.topology import load_topology
from faultline.types import LabError


def completed(
    argv: list[str] | None = None, rc: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv or [], rc, stdout, stderr)


class FakeRunner(CommandRunner):
    """Records every argv it is asked to run. `handler(argv, input)` supplies the result."""

    def __init__(
        self,
        handler: Callable[[list[str], str | None], subprocess.CompletedProcess[str]] | None = None,
        prefix: list[str] | None = None,
    ) -> None:
        super().__init__(prefix)
        self.handler = handler
        self.calls: list[list[str]] = []
        self.inputs: list[str | None] = []

    def run(
        self, argv: list[str], input: str | None = None, timeout: float = 60
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        self.inputs.append(input)
        if self.handler is None:
            return completed(argv)
        return self.handler(list(argv), input)


# ----------------------------------------------------------------- CommandRunner


def test_from_env_without_distro_has_no_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAULTLINE_WSL_DISTRO", raising=False)
    assert CommandRunner.from_env().prefix == []


def test_from_env_with_distro_wraps_in_wsl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAULTLINE_WSL_DISTRO", "Ubuntu-24.04")
    assert CommandRunner.from_env().prefix == [
        "wsl",
        "-d",
        "Ubuntu-24.04",
        "-u",
        "root",
        "--",
    ]


def test_run_prepends_prefix_and_passes_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen["argv"] = argv
        seen.update(kwargs)
        return completed(argv, stdout="ok")

    monkeypatch.setattr(clab.subprocess, "run", fake_run)
    runner = CommandRunner(["wsl", "--"])
    result = runner.run(["docker", "exec", "-i", "c", "cat"], input="hello", timeout=5)
    assert result.stdout == "ok"
    assert seen["argv"] == ["wsl", "--", "docker", "exec", "-i", "c", "cat"]
    assert seen["input"] == "hello"
    assert seen["timeout"] == 5
    assert seen["capture_output"] is True


def test_path_without_prefix_is_unchanged() -> None:
    p = Path("E:/faultline-wt/clab-backend/labs/metro/metro.clab.yml")
    assert CommandRunner().path(p) == str(p)


def test_path_under_wsl_maps_drive_to_mnt() -> None:
    p = Path("E:/faultline-wt/clab-backend/labs/metro/metro.clab.yml")
    runner = CommandRunner(["wsl", "--"])
    assert runner.path(p) == "/mnt/e/faultline-wt/clab-backend/labs/metro/metro.clab.yml"


def test_path_under_wsl_keeps_posix_paths() -> None:
    assert CommandRunner(["wsl", "--"]).path(Path("/home/jae/lab.yml")) == "/home/jae/lab.yml"


# ------------------------------------------------------------------------ ClabLab


@pytest.fixture
def topo():
    return load_topology("metro")


def make_lab(topo, handler=None) -> tuple[ClabLab, FakeRunner]:
    runner = FakeRunner(handler)
    return ClabLab(topo, runner), runner


def test_devices_are_routers(topo) -> None:
    lab, _ = make_lab(topo)
    assert lab.devices() == ["r1", "r2", "r3", "ra", "rb", "rdc"]


def test_unknown_device_raises(topo) -> None:
    lab, runner = make_lab(topo)
    with pytest.raises(LabError, match="unknown device"):
        lab.show("nope", "show version")
    assert runner.calls == []


def test_show_runs_vtysh_in_container(topo) -> None:
    lab, runner = make_lab(topo, lambda argv, _in: completed(argv, stdout="summary"))
    out = lab.show("r1", "show ip bgp summary json")
    assert out == "summary"
    assert runner.calls == [
        ["docker", "exec", "clab-metro-r1", "vtysh", "-c", "show ip bgp summary json"]
    ]


def test_show_rejects_non_show_commands(topo) -> None:
    lab, runner = make_lab(topo)
    with pytest.raises(LabError, match="read-only"):
        lab.show("r1", "configure terminal")
    assert runner.calls == []


def test_show_nonzero_exit_raises_with_stderr(topo) -> None:
    lab, _ = make_lab(topo, lambda argv, _in: completed(argv, rc=1, stderr="boom"))
    with pytest.raises(LabError, match="boom"):
        lab.show("r1", "show version")


def test_configure_builds_argv_and_ends_configure_mode(topo) -> None:
    lab, runner = make_lab(topo)
    lab.configure("r2", ["router bgp 65000", " neighbor 10.255.0.3 remote-as 65000"])
    assert runner.calls == [
        [
            "docker", "exec", "clab-metro-r2", "vtysh",
            "-c", "configure terminal",
            "-c", "router bgp 65000",
            "-c", "neighbor 10.255.0.3 remote-as 65000",
            "-c", "end",
        ]
    ]  # fmt: skip


def test_configure_success_with_notes_is_not_an_error(topo) -> None:
    notes = "% Configuration applied with notes:\n% No changes found to be committed!\n"
    lab, _ = make_lab(topo, lambda argv, _in: completed(argv, rc=0, stdout=notes))
    assert "No changes found" in lab.configure("r1", ["hostname r1"])


def test_configure_nonzero_exit_raises_with_output(topo) -> None:
    err = "% Unknown command: bogus\n"
    lab, _ = make_lab(topo, lambda argv, _in: completed(argv, rc=1, stdout=err))
    with pytest.raises(LabError, match="Unknown command"):
        lab.configure("r1", ["bogus"])


def test_snapshot_strips_header_before_frr_version(topo) -> None:
    raw = (
        "Building configuration...\n\nCurrent configuration:\n!\n"
        "frr version 10.7.1\nfrr defaults traditional\nhostname r1\n!\nend\n"
    )
    lab, runner = make_lab(topo, lambda argv, _in: completed(argv, stdout=raw))
    snap = lab.snapshot("r1")
    assert snap.startswith("frr version 10.7.1\n")
    assert "Building configuration" not in snap
    assert runner.calls[0] == [
        "docker", "exec", "clab-metro-r1", "vtysh", "-c", "show running-config",
    ]  # fmt: skip


def test_snapshot_without_header_returns_output_unchanged(topo) -> None:
    lab, _ = make_lab(topo, lambda argv, _in: completed(argv, stdout="hostname r1\n"))
    assert lab.snapshot("r1") == "hostname r1\n"


def test_restore_writes_file_then_reloads(topo) -> None:
    lab, runner = make_lab(topo)
    lab.restore("r3", "frr version 10.7.1\nhostname r3\n")
    assert runner.calls == [
        ["docker", "exec", "clab-metro-r3", "sh", "-c", f"cat > {RESTORE_FILE}"],
        [
            "docker", "exec", "clab-metro-r3",
            "/usr/lib/frr/frr-reload.py", "--reload", RESTORE_FILE,
        ],
    ]  # fmt: skip
    assert runner.inputs[0] == "frr version 10.7.1\nhostname r3\n"


def test_restore_stops_if_write_fails(topo) -> None:
    lab, runner = make_lab(topo, lambda argv, _in: completed(argv, rc=1, stderr="no space"))
    with pytest.raises(LabError, match="no space"):
        lab.restore("r3", "x")
    assert len(runner.calls) == 1


def test_ping_parses_statistics(topo) -> None:
    out = "3 packets transmitted, 3 packets received, 0% packet loss\n"
    lab, runner = make_lab(topo, lambda argv, _in: completed(argv, stdout=out))
    result = lab.ping("ha", "172.16.0.10")
    assert (result.sent, result.received, result.ok) == (3, 3, True)
    assert runner.calls[0] == [
        "docker", "exec", "clab-metro-ha",
        "ping", "-c", "3", "-W", "1", "-i", "0.2", "172.16.0.10",
    ]  # fmt: skip


def test_ping_loss_is_a_result_not_an_error(topo) -> None:
    out = "3 packets transmitted, 0 received, 100% packet loss\n"
    lab, _ = make_lab(topo, lambda argv, _in: completed(argv, rc=1, stdout=out))
    result = lab.ping("ha", "192.168.2.10")
    assert (result.sent, result.received, result.ok) == (3, 0, False)


def test_ping_without_statistics_raises(topo) -> None:
    lab, _ = make_lab(topo, lambda argv, _in: completed(argv, rc=2, stderr="unknown host"))
    with pytest.raises(LabError, match="unknown host"):
        lab.ping("ha", "not-a-host")


def test_host_exec_runs_raw_argv(topo) -> None:
    lab, runner = make_lab(topo, lambda argv, _in: completed(argv, stdout="eth1 UP"))
    assert lab.host_exec("ha", ["ip", "-br", "link"]) == "eth1 UP"
    assert runner.calls == [["docker", "exec", "clab-metro-ha", "ip", "-br", "link"]]


def test_docker_missing_becomes_lab_error(topo) -> None:
    class Broken(CommandRunner):
        def run(self, argv, input=None, timeout=60):
            raise FileNotFoundError("docker")

    lab = ClabLab(topo, Broken())
    with pytest.raises(LabError, match="could not run"):
        lab.show("r1", "show version")
