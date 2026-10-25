from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from babble.lab import clab
from babble.lab.clab import RESTORE_FILE, ClabLab, CommandRunner, LabManager
from babble.lab.topology import load_topology
from babble.types import LabError


def completed(
    argv: list[str] | None = None, rc: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv or [], rc, stdout, stderr)


class FakeRunner(CommandRunner):
    """Records every argv it is asked to run. `handler(argv, stdin_text)` supplies the result."""

    def __init__(
        self,
        handler: Callable[[list[str], str | None], subprocess.CompletedProcess[str]] | None = None,
        prefix: list[str] | None = None,
    ) -> None:
        super().__init__(prefix)
        self.handler = handler
        self.calls: list[list[str]] = []
        self.stdin_texts: list[str | None] = []

    def run(
        self, argv: list[str], stdin_text: str | None = None, timeout: float = 60
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        self.stdin_texts.append(stdin_text)
        if self.handler is None:
            return completed(argv)
        return self.handler(list(argv), stdin_text)


# ----------------------------------------------------------------- CommandRunner


def test_from_env_without_distro_has_no_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BABBLE_WSL_DISTRO", raising=False)
    assert CommandRunner.from_env().prefix == []


def test_from_env_with_distro_wraps_in_wsl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BABBLE_WSL_DISTRO", "Ubuntu-24.04")
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
    result = runner.run(["docker", "exec", "-i", "c", "cat"], stdin_text="hello", timeout=5)
    assert result.stdout == "ok"
    assert seen["argv"] == ["wsl", "--", "docker", "exec", "-i", "c", "cat"]
    assert seen["input"] == "hello"
    assert seen["timeout"] == 5
    assert seen["capture_output"] is True


def test_path_without_prefix_is_unchanged() -> None:
    host_path = Path("E:/babble-wt/clab-backend/labs/metro/metro.clab.yml")
    assert CommandRunner().path(host_path) == str(host_path)


def test_path_under_wsl_maps_drive_to_mnt() -> None:
    host_path = Path("E:/babble-wt/clab-backend/labs/metro/metro.clab.yml")
    runner = CommandRunner(["wsl", "--"])
    assert runner.path(host_path) == "/mnt/e/babble-wt/clab-backend/labs/metro/metro.clab.yml"


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
    lab, runner = make_lab(topo, lambda argv, _stdin: completed(argv, stdout="summary"))
    summary_output = lab.show("r1", "show ip bgp summary json")
    assert summary_output == "summary"
    assert runner.calls == [
        ["docker", "exec", "clab-metro-r1", "vtysh", "-c", "show ip bgp summary json"]
    ]


def test_show_rejects_non_show_commands(topo) -> None:
    lab, runner = make_lab(topo)
    with pytest.raises(LabError, match="read-only"):
        lab.show("r1", "configure terminal")
    assert runner.calls == []


def test_show_nonzero_exit_raises_with_stderr(topo) -> None:
    lab, _ = make_lab(topo, lambda argv, _stdin: completed(argv, rc=1, stderr="boom"))
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
    lab, _ = make_lab(topo, lambda argv, _stdin: completed(argv, rc=0, stdout=notes))
    assert "No changes found" in lab.configure("r1", ["hostname r1"])


def test_configure_nonzero_exit_raises_with_output(topo) -> None:
    unknown_command_output = "% Unknown command: bogus\n"
    failing = lambda argv, _stdin: completed(argv, rc=1, stdout=unknown_command_output)
    lab, _ = make_lab(topo, failing)
    with pytest.raises(LabError, match="Unknown command"):
        lab.configure("r1", ["bogus"])


def test_snapshot_strips_header_before_frr_version(topo) -> None:
    raw_running_config = (
        "Building configuration...\n\nCurrent configuration:\n!\n"
        "frr version 10.7.1\nfrr defaults traditional\nhostname r1\n!\nend\n"
    )
    lab, runner = make_lab(topo, lambda argv, _stdin: completed(argv, stdout=raw_running_config))
    snapshot_text = lab.snapshot("r1")
    assert snapshot_text.startswith("frr version 10.7.1\n")
    assert "Building configuration" not in snapshot_text
    assert runner.calls[0] == [
        "docker", "exec", "clab-metro-r1", "vtysh", "-c", "show running-config",
    ]  # fmt: skip


def test_snapshot_without_frr_version_line_raises_naming_the_device(topo) -> None:
    lab, _ = make_lab(topo, lambda argv, _stdin: completed(argv, stdout="hostname r1\n"))
    with pytest.raises(LabError, match="r1.*frr version"):
        lab.snapshot("r1")


def test_restore_writes_file_then_reloads(topo) -> None:
    lab, runner = make_lab(topo)
    lab.restore("r3", "frr version 10.7.1\nhostname r3\n")
    assert runner.calls == [
        ["docker", "exec", "-i", "clab-metro-r3", "tee", RESTORE_FILE],
        [
            "docker", "exec", "clab-metro-r3",
            "/usr/lib/frr/frr-reload.py", "--reload", RESTORE_FILE,
        ],
    ]  # fmt: skip
    assert runner.stdin_texts[0] == "frr version 10.7.1\nhostname r3\n"


def test_restore_stops_if_write_fails(topo) -> None:
    lab, runner = make_lab(topo, lambda argv, _stdin: completed(argv, rc=1, stderr="no space"))
    with pytest.raises(LabError, match="no space"):
        lab.restore("r3", "x")
    assert len(runner.calls) == 1


def test_ping_parses_statistics(topo) -> None:
    summary_output = "3 packets transmitted, 3 packets received, 0% packet loss\n"
    lab, runner = make_lab(topo, lambda argv, _stdin: completed(argv, stdout=summary_output))
    result = lab.ping("ha", "172.16.0.10")
    assert (result.sent, result.received, result.ok) == (3, 3, True)
    assert runner.calls[0] == [
        "docker", "exec", "clab-metro-ha",
        "ping", "-c", "3", "-W", "1", "-i", "0.2", "172.16.0.10",
    ]  # fmt: skip


def test_ping_loss_is_a_result_not_an_error(topo) -> None:
    summary_output = "3 packets transmitted, 0 received, 100% packet loss\n"
    lab, _ = make_lab(topo, lambda argv, _stdin: completed(argv, rc=1, stdout=summary_output))
    result = lab.ping("ha", "192.168.2.10")
    assert (result.sent, result.received, result.ok) == (3, 0, False)


def test_ping_without_statistics_raises(topo) -> None:
    lab, _ = make_lab(topo, lambda argv, _stdin: completed(argv, rc=2, stderr="unknown host"))
    with pytest.raises(LabError, match="unknown host"):
        lab.ping("ha", "not-a-host")


def test_host_exec_runs_raw_argv(topo) -> None:
    lab, runner = make_lab(topo, lambda argv, _stdin: completed(argv, stdout="eth1 UP"))
    assert lab.host_exec("ha", ["ip", "-br", "link"]) == "eth1 UP"
    assert runner.calls == [["docker", "exec", "clab-metro-ha", "ip", "-br", "link"]]


def test_docker_missing_becomes_lab_error(topo) -> None:
    class Broken(CommandRunner):
        def run(self, argv, stdin_text=None, timeout=60):
            raise FileNotFoundError("docker")

    lab = ClabLab(topo, Broken())
    with pytest.raises(LabError, match="could not run"):
        lab.show("r1", "show version")


# -------------------------------------------------------------------- LabManager


def make_manager(topo, handler=None) -> tuple[LabManager, FakeRunner]:
    runner = FakeRunner(handler)
    return LabManager(topo, runner), runner


def test_deploy_runs_containerlab_deploy(topo) -> None:
    manager, runner = make_manager(topo)
    manager.deploy()
    assert runner.calls == [["containerlab", "deploy", "-t", str(topo.clab_file), "--reconfigure"]]


def test_destroy_runs_containerlab_destroy(topo) -> None:
    manager, runner = make_manager(topo)
    manager.destroy()
    assert runner.calls == [["containerlab", "destroy", "-t", str(topo.clab_file), "--cleanup"]]


def test_deploy_failure_raises_with_stderr(topo) -> None:
    manager, _ = make_manager(topo, lambda argv, _stdin: completed(argv, rc=1, stderr="no docker"))
    with pytest.raises(LabError, match="no docker"):
        manager.deploy()


def test_take_baseline_then_reset_restores_each_router(topo) -> None:
    def handler(argv, _stdin):
        if "running-config" in argv[-1]:
            device = argv[argv.index("exec") + 1].removeprefix("clab-metro-")
            return completed(argv, stdout=f"frr version 10.7.1\nhostname {device}\n")
        return completed(argv)

    manager, runner = make_manager(topo, handler)
    manager.take_baseline()
    snapshot_calls = [call for call in runner.calls if "running-config" in call[-1]]
    assert len(snapshot_calls) == len(topo.routers())

    runner.calls.clear()
    runner.stdin_texts.clear()
    manager.reset()
    restore_inputs = [
        config_input for config_input in runner.stdin_texts if config_input is not None
    ]
    assert restore_inputs == [
        f"frr version 10.7.1\nhostname {router}\n" for router in topo.routers()
    ]


def test_reset_without_baseline_raises(topo) -> None:
    manager, runner = make_manager(topo)
    with pytest.raises(LabError, match="baseline"):
        manager.reset()
    assert runner.calls == []


def test_wait_healthy_returns_when_all_checks_pass(topo) -> None:
    manager, _ = make_manager(topo)
    seen: list[object] = []

    def evaluate(check, lab):
        seen.append((lab, check))
        return True

    manager.wait_healthy(timeout=0, evaluate=evaluate)
    assert len(seen) == len(topo.health_checks)
    assert all(lab is manager.lab for lab, _ in seen)


def test_wait_healthy_retries_until_checks_pass(topo) -> None:
    manager, _ = make_manager(topo)
    calls = {"evaluations": 0}

    def evaluate(check, lab):
        calls["evaluations"] += 1
        return calls["evaluations"] > len(topo.health_checks)  # first pass fails, second passes

    manager.wait_healthy(timeout=5, interval=0, evaluate=evaluate)
    assert calls["evaluations"] == 2 * len(topo.health_checks)


def test_wait_healthy_treats_lab_errors_as_not_yet_healthy(topo) -> None:
    manager, _ = make_manager(topo)
    calls = {"evaluations": 0}

    def evaluate(check, lab):
        calls["evaluations"] += 1
        if calls["evaluations"] == 1:
            raise LabError("container not up")
        return True

    manager.wait_healthy(timeout=5, interval=0, evaluate=evaluate)
    assert calls["evaluations"] == 2 * len(topo.health_checks)


def test_wait_healthy_times_out_naming_failing_checks(topo) -> None:
    manager, _ = make_manager(topo)
    with pytest.raises(LabError, match="ping ha -> 172.16.0.10"):
        manager.wait_healthy(timeout=0, evaluate=lambda check, lab: False)


