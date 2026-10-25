"""ScriptedLab: an in-memory LabBackend. The fake network every offline test uses.

Each device answers `show` commands from a table of canned output. The lab starts "broken"
and becomes "fixed" once the config lines in `fix` have all been applied, so a test can
watch an agent's change move the lab from one state to the other without Docker.

Rules, so tests stay predictable:

- Commands, config lines and ping targets are normalized by stripping them and collapsing
  internal whitespace. Table keys are normalized the same way when the lab is built.
- The state is derived from the applied config lines, never stored separately. The lab is
  "fixed" when every device named in `fix` has received all of its lines (cumulatively).
  With an empty `fix` the lab is never fixed.
- `show` uses `fixed_responses` first while fixed, then `responses`. A command with no entry
  returns a "% Unknown command" line, as a real vtysh would.
- `ping` works the same way with `fixed_pings` and `pings`. A missing entry means no reply.
- `snapshot` captures one device's applied config lines. `restore` puts them back and
  recomputes the state, so restoring a device can move the whole lab back to "broken".
- `host_exec` is recorded only. It never changes state, which keeps fault injection
  visible to tests through `host_calls`.

Usage:

    lab = ScriptedLab(
        devices=["r1", "ha"],
        hosts={"ha"},
        responses={("r1", "show ip route 172.16.0.0/24 json"): "{}"},
        fixed_responses={("r1", "show ip route 172.16.0.0/24 json"): ROUTE_JSON},
        fix={"r1": ["ip route 172.16.0.0/24 10.0.13.2"]},
        pings={("ha", "172.16.0.10"): False},
    )
    lab.state                                   # "broken"
    lab.configure("r1", ["ip route  172.16.0.0/24 10.0.13.2"])
    lab.state                                   # "fixed"
    lab.show("r1", "show ip route 172.16.0.0/24 json")   # ROUTE_JSON
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, TypeVar

from babble.types import LabError, PingResult

_V = TypeVar("_V")


def _norm(text: str) -> str:
    """Strip and collapse internal whitespace."""
    return " ".join(text.split())


def _normalize_table(
    table: Mapping[tuple[str, str], _V], known: set[str], label: str
) -> dict[tuple[str, str], _V]:
    out: dict[tuple[str, str], _V] = {}
    for (device, text), value in table.items():
        if device not in known:
            raise ValueError(f"{label} refers to unknown device {device!r}")
        out[(device, _norm(text))] = value
    return out


class ScriptedLab:
    """In-memory LabBackend. See the module docstring for the rules."""

    def __init__(
        self,
        devices: list[str],
        responses: dict[tuple[str, str], str],
        fixed_responses: dict[tuple[str, str], str] | None = None,
        fix: dict[str, list[str]] | None = None,
        pings: dict[tuple[str, str], bool] | None = None,
        fixed_pings: dict[tuple[str, str], bool] | None = None,
        hosts: set[str] | None = None,
    ) -> None:
        if len(set(devices)) != len(devices):
            raise ValueError("duplicate device names")
        self._devices = list(devices)
        known = set(self._devices)

        self._hosts = set(hosts or ())
        if not self._hosts <= known:
            raise ValueError(f"hosts not in devices: {sorted(self._hosts - known)}")

        self._responses = _normalize_table(responses, known, "responses")
        self._fixed_responses = _normalize_table(fixed_responses or {}, known, "fixed_responses")
        self._pings = _normalize_table(pings or {}, known, "pings")
        self._fixed_pings = _normalize_table(fixed_pings or {}, known, "fixed_pings")

        self._fix: dict[str, set[str]] = {}
        for device, lines in (fix or {}).items():
            if device not in known:
                raise ValueError(f"fix refers to unknown device {device!r}")
            self._fix[device] = {_norm(line) for line in lines} - {""}

        # Applied config lines per device, in the order they were first applied.
        self._applied: dict[str, list[str]] = {device: [] for device in self._devices}

        # Accepted configure calls, with normalized lines, in call order.
        self.config_calls: list[tuple[str, list[str]]] = []
        # Host exec calls as (device, argv), in call order.
        self.host_calls: list[tuple[str, list[str]]] = []

    # ----------------------------------------------------------------- state

    @property
    def state(self) -> str:
        """Either "fixed" (every device in `fix` has all its lines applied) or "broken"."""
        return "fixed" if self._is_fixed() else "broken"

    def _is_fixed(self) -> bool:
        if not self._fix:
            return False
        return all(required <= set(self._applied[device]) for device, required in self._fix.items())

    def applied_lines(self, device: str) -> list[str]:
        """Normalized config lines applied to `device` so far, in first-applied order."""
        self._require_device(device)
        return list(self._applied[device])

    # ------------------------------------------------------------- LabBackend

    def devices(self) -> list[str]:
        return list(self._devices)

    def show(self, device: str, command: str) -> str:
        self._require_router(device)
        key = (device, _norm(command))
        if self._is_fixed() and key in self._fixed_responses:
            return self._fixed_responses[key]
        if key in self._responses:
            return self._responses[key]
        return f"% Unknown command: {key[1]}"

    def configure(self, device: str, lines: list[str]) -> str:
        self._require_router(device)
        normalized = [n for n in (_norm(line) for line in lines) if n]
        self.config_calls.append((device, list(normalized)))
        applied = self._applied[device]
        for line in normalized:
            if line not in applied:
                applied.append(line)
        return ""

    def snapshot(self, device: str) -> str:
        self._require_device(device)
        return json.dumps({"device": device, "applied": self._applied[device]})

    def restore(self, device: str, config: str) -> None:
        self._require_device(device)
        try:
            data: Any = json.loads(config)
        except (TypeError, ValueError) as exc:
            raise LabError(f"not a snapshot for {device}: {exc}") from exc
        if (
            not isinstance(data, dict)
            or data.get("device") != device
            or not isinstance(data.get("applied"), list)
        ):
            raise LabError(f"not a snapshot for {device}")
        self._applied[device] = [str(line) for line in data["applied"]]

    def ping(self, device: str, target: str, count: int = 3) -> PingResult:
        self._require_device(device)
        key = (device, _norm(target))
        if self._is_fixed() and key in self._fixed_pings:
            ok = self._fixed_pings[key]
        else:
            ok = self._pings.get(key, False)
        received = count if ok else 0
        return PingResult(
            sent=count,
            received=received,
            output=f"{count} packets transmitted, {received} received",
        )

    def host_exec(self, device: str, argv: list[str]) -> str:
        self._require_device(device)
        self.host_calls.append((device, list(argv)))
        return ""

    # ---------------------------------------------------------------- helpers

    def _require_device(self, device: str) -> None:
        if device not in self._devices:
            raise LabError(f"unknown device: {device!r}")

    def _require_router(self, device: str) -> None:
        self._require_device(device)
        if device in self._hosts:
            raise LabError(f"{device} is not a router")


__all__ = ["ScriptedLab"]
