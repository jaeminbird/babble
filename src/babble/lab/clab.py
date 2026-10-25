"""Drive a containerlab lab through docker and containerlab CLIs.

Everything here shells out. Tests substitute a fake CommandRunner.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

from babble.lab.topology import Topology
from babble.polling import poll_until
from babble.types import PING_PACKET_COUNT, Check, LabBackend, LabError, PingResult

# Decides whether one health check passes on a running lab. Matches the shape of
# babble.netops.checks.evaluate.
Evaluate = Callable[[Check, LabBackend], bool]

WSL_PREFIX_ENV = "BABBLE_WSL_DISTRO"
_DRIVE_PATH = re.compile(r"^([A-Za-z]):/(.*)$")
RESTORE_FILE = "/tmp/babble-restore.conf"
_PING_STATISTICS = re.compile(r"(\d+) packets transmitted, (\d+) (?:packets )?received")


class CommandRunner:
    """Runs argv lists, optionally through a prefix such as `wsl -d <distro> -u root --`."""

    def __init__(self, prefix: list[str] | None = None) -> None:
        self.prefix: list[str] = list(prefix or [])

    @classmethod
    def from_env(cls) -> CommandRunner:
        distro = os.environ.get(WSL_PREFIX_ENV)
        if not distro:
            return cls()
        return cls(["wsl", "-d", distro, "-u", "root", "--"])

    def run(
        self, argv: list[str], stdin_text: str | None = None, timeout: float = 60
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*self.prefix, *argv],
            input=stdin_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )

    def path(self, host_path: Path) -> str:
        """Path as the runner's process sees it. Under WSL, a drive path such as E:/x/y
        becomes /mnt/e/x/y."""
        path_text = str(host_path)
        if not self.prefix:
            return path_text
        path_text = path_text.replace("\\", "/")
        match = _DRIVE_PATH.match(path_text)
        if match:
            return f"/mnt/{match.group(1).lower()}/{match.group(2)}"
        return path_text


def _error_text(completed_process: subprocess.CompletedProcess[str], fallback: str) -> str:
    """Stderr, else stdout, else the fallback: the most useful text a failed command gave."""
    return completed_process.stderr.strip() or completed_process.stdout.strip() or fallback


class ClabLab:
    """A running containerlab topology, seen through `docker exec` into each device."""

    def __init__(self, topology: Topology, runner: CommandRunner) -> None:
        self.topology = topology
        self.runner = runner

    def _container_name(self, device: str) -> str:
        if device not in self.topology.devices:
            raise LabError(f"unknown device {device!r} in lab {self.topology.name!r}")
        return f"clab-{self.topology.name}-{device}"

    def _run_command(
        self, argv: list[str], stdin_text: str | None = None, timeout: float = 60
    ) -> subprocess.CompletedProcess[str]:
        try:
            return self.runner.run(argv, stdin_text=stdin_text, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LabError(f"could not run {argv[0]}: {exc}") from exc

    def _run_in_container(
        self, device: str, argv: list[str], stdin_text: str | None = None, timeout: float = 60
    ) -> subprocess.CompletedProcess[str]:
        """Run argv inside the device container with docker exec. `-i` is added when stdin is
        given. A non-zero exit is returned, not raised."""
        interactive_flags = ["-i"] if stdin_text is not None else []
        return self._run_command(
            ["docker", "exec", *interactive_flags, self._container_name(device), *argv],
            stdin_text=stdin_text,
            timeout=timeout,
        )

    def _exec_in_container(
        self, device: str, argv: list[str], stdin_text: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Like _run_in_container, but a non-zero exit raises LabError."""
        completed_process = self._run_in_container(device, argv, stdin_text=stdin_text)
        if completed_process.returncode != 0:
            raise LabError(_error_text(completed_process, f"exit {completed_process.returncode}"))
        return completed_process

    def devices(self) -> list[str]:
        return self.topology.routers()

    def show(self, device: str, command: str) -> str:
        if not command.strip().startswith("show "):
            raise LabError(f"show only runs read-only 'show' commands, got {command!r}")
        return self._exec_in_container(device, ["vtysh", "-c", command]).stdout

    def configure(self, device: str, lines: list[str]) -> str:
        # vtysh's exit code is the error signal. Its '%' notes on success are not errors.
        argv = ["vtysh", "-c", "configure terminal"]
        for line in lines:
            argv += ["-c", line.strip()]
        argv += ["-c", "end"]
        completed_process = self._run_in_container(device, argv)
        output = completed_process.stdout + completed_process.stderr
        if completed_process.returncode != 0:
            raise LabError(output.strip() or f"vtysh exit {completed_process.returncode}")
        return output

    def snapshot(self, device: str) -> str:
        output = self._exec_in_container(device, ["vtysh", "-c", "show running-config"]).stdout
        # Drop the "Building configuration..." header. frr-reload.py only accepts
        # a config that starts at 'frr version'.
        lines = output.splitlines(keepends=True)
        for index, line in enumerate(lines):
            if line.startswith("frr version"):
                return "".join(lines[index:])
        raise LabError(
            f"{device} running config has no 'frr version' line, so it cannot be snapshotted "
            "for frr-reload."
        )

    def restore(self, device: str, config: str) -> None:
        # tee, not a shell redirect: no quoting to survive through a wsl.exe prefix.
        self._exec_in_container(device, ["tee", RESTORE_FILE], stdin_text=config)
        self._exec_in_container(device, ["/usr/lib/frr/frr-reload.py", "--reload", RESTORE_FILE])

    def ping(self, device: str, target: str) -> PingResult:
        # Loss is not an error: ping exits non-zero when packets are lost.
        argv = ["ping", "-c", str(PING_PACKET_COUNT), "-W", "1", "-i", "0.2", target]
        completed_process = self._run_in_container(
            device, argv, timeout=PING_PACKET_COUNT * 2 + 10
        )
        output = completed_process.stdout + completed_process.stderr
        statistics_match = _PING_STATISTICS.search(output)
        if statistics_match is None:
            raise LabError(_error_text(completed_process, "ping gave no statistics"))
        return PingResult(
            sent=int(statistics_match.group(1)),
            received=int(statistics_match.group(2)),
            output=output,
        )

    def host_exec(self, device: str, argv: list[str]) -> str:
        return self._exec_in_container(device, argv).stdout


class LabManager:
    """Deploys and destroys a topology, and snapshots and resets its routers."""

    def __init__(self, topology: Topology, runner: CommandRunner) -> None:
        self.topology = topology
        self.runner = runner
        self.lab = ClabLab(topology, runner)
        self._baseline_snapshots_by_router: dict[str, str] = {}

    def deploy(self) -> None:
        clab_file_path = self.runner.path(self.topology.clab_file)
        self._run_containerlab(["deploy", "-t", clab_file_path, "--reconfigure"], timeout=900)

    def destroy(self) -> None:
        clab_file_path = self.runner.path(self.topology.clab_file)
        self._run_containerlab(["destroy", "-t", clab_file_path, "--cleanup"], timeout=600)

    def _run_containerlab(self, args: list[str], timeout: float) -> None:
        try:
            completed_process = self.runner.run(["containerlab", *args], timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LabError(f"could not run containerlab: {exc}") from exc
        if completed_process.returncode != 0:
            detail = _error_text(completed_process, f"exit {completed_process.returncode}")
            raise LabError(f"containerlab {args[0]} failed: {detail}")

    def wait_healthy(self, evaluate: Evaluate, timeout: float = 120, interval: float = 5) -> None:
        """Poll every health check until all pass. Raises LabError when time runs out."""
        failing_checks_at_last_poll: list[Check] = []

        def all_health_checks_pass() -> bool:
            nonlocal failing_checks_at_last_poll
            failing_checks_at_last_poll = [
                check
                for check in self.topology.health_checks
                if not _check_is_passing(evaluate, self.lab, check)
            ]
            return not failing_checks_at_last_poll

        if not poll_until(all_health_checks_pass, timeout, interval):
            names = ", ".join(check.describe() for check in failing_checks_at_last_poll)
            raise LabError(f"lab not healthy after {timeout}s; failing: {names}")

    def take_baseline(self) -> None:
        self._baseline_snapshots_by_router = {
            router_name: self.lab.snapshot(router_name)
            for router_name in self.topology.routers()
        }

    def reset(self) -> None:
        if not self._baseline_snapshots_by_router:
            raise LabError("no baseline to reset to; call take_baseline() first")
        for router_name, snapshot_config in self._baseline_snapshots_by_router.items():
            self.lab.restore(router_name, snapshot_config)


def _check_is_passing(evaluate: Evaluate, lab: LabBackend, check: Check) -> bool:
    try:
        return bool(evaluate(check, lab))
    except LabError:
        # A container that is not up yet is simply not healthy yet.
        return False
