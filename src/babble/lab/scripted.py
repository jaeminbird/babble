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
  visible to tests through `host_exec_calls`.

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

from babble.types import PING_PACKET_COUNT, LabError, PingResult

_TableValue = TypeVar("_TableValue")


def _normalize_whitespace(text: str) -> str:
    """Strip and collapse internal whitespace."""
    return " ".join(text.split())


def _normalize_table(
    table: Mapping[tuple[str, str], _TableValue],
    known_device_names: set[str],
    table_label: str,
) -> dict[tuple[str, str], _TableValue]:
    normalized_table: dict[tuple[str, str], _TableValue] = {}
    for (device, raw_text), value in table.items():
        if device not in known_device_names:
            raise ValueError(f"{table_label} refers to unknown device {device!r}")
        normalized_table[(device, _normalize_whitespace(raw_text))] = value
    return normalized_table


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
        self._device_names = list(devices)
        known_device_names = set(self._device_names)

        self._host_names = set(hosts or ())
        if not self._host_names <= known_device_names:
            unknown_hosts = sorted(self._host_names - known_device_names)
            raise ValueError(f"hosts not in devices: {unknown_hosts}")

        self._broken_show_responses = _normalize_table(
            responses, known_device_names, "responses"
        )
        self._fixed_show_responses = _normalize_table(
            fixed_responses or {}, known_device_names, "fixed_responses"
        )
        self._broken_ping_results = _normalize_table(pings or {}, known_device_names, "pings")
        self._fixed_ping_results = _normalize_table(
            fixed_pings or {}, known_device_names, "fixed_pings"
        )

        self._required_config_lines: dict[str, set[str]] = {}
        for device, lines in (fix or {}).items():
            if device not in known_device_names:
                raise ValueError(f"fix refers to unknown device {device!r}")
            normalized_lines = {_normalize_whitespace(line) for line in lines} - {""}
            if not normalized_lines:
                raise ValueError(
                    f"fix for {device!r} is empty, so the device would be trivially fixed"
                )
            self._required_config_lines[device] = normalized_lines

        # Applied config lines per device, in the order they were first applied.
        self._applied_config_lines: dict[str, list[str]] = {
            device: [] for device in self._device_names
        }

        # Accepted configure calls, with normalized lines, in call order.
        self.configure_calls: list[tuple[str, list[str]]] = []
        # Host exec calls as (device, argv), in call order.
        self.host_exec_calls: list[tuple[str, list[str]]] = []

    # ----------------------------------------------------------------- state

    @property
    def state(self) -> str:
        """Either "fixed" (every device in `fix` has all its lines applied) or "broken"."""
        return "fixed" if self._is_fixed() else "broken"

    def _is_fixed(self) -> bool:
        if not self._required_config_lines:
            return False
        return all(
            required_lines <= set(self._applied_config_lines[device])
            for device, required_lines in self._required_config_lines.items()
        )

    def applied_lines(self, device: str) -> list[str]:
        """Normalized config lines applied to `device` so far, in first-applied order."""
        self._require_device(device)
        return list(self._applied_config_lines[device])

    # ------------------------------------------------------------- LabBackend

    def devices(self) -> list[str]:
        return list(self._device_names)

    def show(self, device: str, command: str) -> str:
        self._require_router(device)
        key = (device, _normalize_whitespace(command))
        output = self._lookup_for_current_state(
            self._fixed_show_responses, self._broken_show_responses, key
        )
        if output is None:
            return f"% Unknown command: {key[1]}"
        return output

    def configure(self, device: str, lines: list[str]) -> str:
        self._require_router(device)
        normalized_lines = [line for line in map(_normalize_whitespace, lines) if line]
        self.configure_calls.append((device, normalized_lines))
        applied_lines = self._applied_config_lines[device]
        for normalized_line in normalized_lines:
            if normalized_line not in applied_lines:
                applied_lines.append(normalized_line)
        return ""

    def snapshot(self, device: str) -> str:
        self._require_device(device)
        return json.dumps({"device": device, "applied": self._applied_config_lines[device]})

    def restore(self, device: str, config: str) -> None:
        self._require_device(device)
        try:
            parsed_snapshot: Any = json.loads(config)
        except (TypeError, ValueError) as exc:
            raise LabError(f"not a snapshot for {device}: {exc}") from exc
        if (
            not isinstance(parsed_snapshot, dict)
            or parsed_snapshot.get("device") != device
            or not isinstance(parsed_snapshot.get("applied"), list)
        ):
            raise LabError(f"not a snapshot for {device}")
        self._applied_config_lines[device] = [str(line) for line in parsed_snapshot["applied"]]

    def ping(self, device: str, target: str) -> PingResult:
        self._require_device(device)
        key = (device, _normalize_whitespace(target))
        # A missing entry means no reply.
        reply_received = bool(
            self._lookup_for_current_state(self._fixed_ping_results, self._broken_ping_results, key)
        )
        received = PING_PACKET_COUNT if reply_received else 0
        return PingResult(
            sent=PING_PACKET_COUNT,
            received=received,
            output=f"{PING_PACKET_COUNT} packets transmitted, {received} received",
        )

    def host_exec(self, device: str, argv: list[str]) -> str:
        self._require_device(device)
        self.host_exec_calls.append((device, list(argv)))
        return ""

    # ---------------------------------------------------------------- helpers

    def _lookup_for_current_state(
        self,
        fixed_table: dict[tuple[str, str], _TableValue],
        broken_table: dict[tuple[str, str], _TableValue],
        key: tuple[str, str],
    ) -> _TableValue | None:
        """The fixed table's entry while the lab is fixed and has one, else the broken table's."""
        if self._is_fixed() and key in fixed_table:
            return fixed_table[key]
        return broken_table.get(key)

    def _require_device(self, device: str) -> None:
        if device not in self._device_names:
            raise LabError(f"unknown device: {device!r}")

    def _require_router(self, device: str) -> None:
        self._require_device(device)
        if device in self._host_names:
            raise LabError(f"{device} is not a router")


__all__ = ["ScriptedLab"]
