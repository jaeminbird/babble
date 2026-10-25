"""Drive a containerlab lab through docker and containerlab CLIs.

Everything here shells out. Tests substitute a fake CommandRunner.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from faultline.lab.topology import Topology
from faultline.types import Check, LabBackend, LabError, PingResult

# Decides whether one health check passes on a running lab. Matches the shape of
# faultline.netops.checks.evaluate, which is imported lazily when no callable is given.
Evaluate = Callable[[LabBackend, Check], bool]

WSL_PREFIX_ENV = "FAULTLINE_WSL_DISTRO"
_DRIVE = re.compile(r"^([A-Za-z]):/(.*)$")
RESTORE_FILE = "/tmp/faultline-restore.conf"
_PING_STATS = re.compile(r"(\d+) packets transmitted, (\d+) (?:packets )?received")


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
        self, argv: list[str], input: str | None = None, timeout: float = 60
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*self.prefix, *argv],
            input=input,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )

    def path(self, p: Path) -> str:
        """Path as the runner's process sees it. Under WSL, a drive path such as E:/x/y
        becomes /mnt/e/x/y."""
        text = str(p)
        if not self.prefix:
            return text
        text = text.replace("\\", "/")
        match = _DRIVE.match(text)
        if match:
            return f"/mnt/{match.group(1).lower()}/{match.group(2)}"
        return text


class ClabLab:
    """A running containerlab topology, seen through `docker exec` into each device."""

    def __init__(self, topology: Topology, runner: CommandRunner) -> None:
        self.topology = topology
        self.runner = runner

    def _container(self, device: str) -> str:
        if device not in self.topology.devices:
            raise LabError(f"unknown device {device!r} in lab {self.topology.name!r}")
        return f"clab-{self.topology.name}-{device}"

    def _run(
        self, argv: list[str], input: str | None = None, timeout: float = 60
    ) -> subprocess.CompletedProcess[str]:
        try:
            return self.runner.run(argv, input=input, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LabError(f"could not run {argv[0]}: {exc}") from exc

    def _exec(
        self, device: str, argv: list[str], input: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Run argv inside the device container. Non-zero exit raises LabError."""
        proc = self._run(["docker", "exec", *self._exec_args(device), *argv], input=input)
        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
            raise LabError(detail)
        return proc

    def _exec_args(self, device: str) -> list[str]:
        return [self._container(device)]

    def devices(self) -> list[str]:
        return self.topology.routers()

    def show(self, device: str, command: str) -> str:
        if not command.strip().startswith("show "):
            raise LabError(f"show only runs read-only 'show' commands, got {command!r}")
        return self._exec(device, ["vtysh", "-c", command]).stdout

    def configure(self, device: str, lines: list[str]) -> str:
        # vtysh's exit code is the error signal. Its '%' notes on success are not errors.
        argv = ["vtysh", "-c", "configure terminal"]
        for line in lines:
            argv += ["-c", line.strip()]
        argv += ["-c", "end"]
        proc = self._run(["docker", "exec", *self._exec_args(device), *argv])
        output = proc.stdout + proc.stderr
        if proc.returncode != 0:
            raise LabError(output.strip() or f"vtysh exit {proc.returncode}")
        return output

    def snapshot(self, device: str) -> str:
        output = self._exec(device, ["vtysh", "-c", "show running-config"]).stdout
        # Drop the "Building configuration..." header; frr-reload.py only accepts from 'frr version'.
        lines = output.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if line.startswith("frr version"):
                return "".join(lines[i:])
        return output

    def restore(self, device: str, config: str) -> None:
        self._exec(device, ["sh", "-c", f"cat > {RESTORE_FILE}"], input=config)
        self._exec(device, ["/usr/lib/frr/frr-reload.py", "--reload", RESTORE_FILE])

    def ping(self, device: str, target: str, count: int = 3) -> PingResult:
        # Loss is not an error: ping exits non-zero when packets are lost.
        argv = ["ping", "-c", str(count), "-W", "1", "-i", "0.2", target]
        proc = self._run(
            ["docker", "exec", *self._exec_args(device), *argv], timeout=count * 2 + 10
        )
        output = proc.stdout + proc.stderr
        match = _PING_STATS.search(output)
        if match is None:
            raise LabError(proc.stderr.strip() or proc.stdout.strip() or "ping gave no statistics")
        return PingResult(sent=int(match.group(1)), received=int(match.group(2)), output=output)

    def host_exec(self, device: str, argv: list[str]) -> str:
        return self._exec(device, argv).stdout


class LabManager:
    """Deploys and destroys a topology, and snapshots and resets its routers."""

    def __init__(self, topology: Topology, runner: CommandRunner) -> None:
        self.topology = topology
        self.runner = runner
        self.lab = ClabLab(topology, runner)
        self._baseline: dict[str, str] = {}

    def deploy(self) -> None:
        path = self.runner.path(self.topology.clab_file)
        self._clab(["deploy", "-t", path, "--reconfigure"], timeout=900)

    def destroy(self) -> None:
        path = self.runner.path(self.topology.clab_file)
        self._clab(["destroy", "-t", path, "--cleanup"], timeout=600)

    def _clab(self, args: list[str], timeout: float) -> None:
        try:
            proc = self.runner.run(["containerlab", *args], timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LabError(f"could not run containerlab: {exc}") from exc
        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
            raise LabError(f"containerlab {args[0]} failed: {detail}")

    def wait_healthy(
        self,
        timeout: float = 120,
        interval: float = 5,
        evaluate: Evaluate | None = None,
    ) -> None:
        """Poll every health check until all pass. Raises LabError when time runs out."""
        if evaluate is None:
            # Lazy: netops.checks is written separately and may not exist on every branch.
            from faultline.netops.checks import evaluate as default_evaluate

            evaluate = default_evaluate
        deadline = time.monotonic() + timeout
        while True:
            failing = [
                check
                for check in self.topology.health_checks
                if not _passes(evaluate, self.lab, check)
            ]
            if not failing:
                return
            if time.monotonic() >= deadline:
                names = ", ".join(c.describe() for c in failing)
                raise LabError(f"lab not healthy after {timeout}s; failing: {names}")
            time.sleep(interval)

    def take_baseline(self) -> None:
        self._baseline = {r: self.lab.snapshot(r) for r in self.topology.routers()}

    def reset(self) -> None:
        if not self._baseline:
            raise LabError("no baseline to reset to; call take_baseline() first")
        for router, config in self._baseline.items():
            self.lab.restore(router, config)


def _passes(evaluate: Evaluate, lab: LabBackend, check: Check) -> bool:
    try:
        return bool(evaluate(lab, check))
    except LabError:
        # A container that is not up yet is simply not healthy yet.
        return False
