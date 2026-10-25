from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from faultline.lab import clab
from faultline.lab.clab import CommandRunner


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
