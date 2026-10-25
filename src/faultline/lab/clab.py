"""Drive a containerlab lab through docker and containerlab CLIs.

Everything here shells out. Tests substitute a fake CommandRunner.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

WSL_PREFIX_ENV = "FAULTLINE_WSL_DISTRO"
_DRIVE = re.compile(r"^([A-Za-z]):/(.*)$")


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
