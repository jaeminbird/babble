"""Fault scenarios: a YAML file describes one fault to inject into a lab, the checks it breaks,
and the root cause a diagnosis must name. The bench loads these; the agent never sees them."""

from __future__ import annotations

import ipaddress
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

import yaml

from babble.types import FAULT_CATEGORIES, Check, CheckKind, LabBackend

Failer = Callable[[str], ValueError]

SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"
DIFFICULTIES = ("easy", "medium", "hard")
CHECK_KINDS: tuple[str, ...] = get_args(CheckKind)

_TOP_LEVEL_KEYS = {
    "id",
    "title",
    "topology",
    "difficulty",
    "fixable",
    "alert",
    "inject",
    "checks",
    "root_cause",
}
_INJECT_STEP_KEYS = {"device", "config", "host", "revert"}
_SCENARIO_CHECK_KEYS = {"kind", "device", "target"}
_ROOT_CAUSE_KEYS = {"device", "category"}


@dataclass
class InjectStep:
    device: str
    config: list[str] | None = None
    host: list[str] | None = None
    revert: list[str] | None = None


@dataclass
class RootCause:
    device: str
    category: str


@dataclass
class Scenario:
    id: str
    title: str
    topology: str
    difficulty: str
    alert: str
    inject: list[InjectStep]
    checks: list[Check]
    root_cause: RootCause
    fixable: bool


# ----------------------------------------------------------------------- loading


def load_scenario(path: Path | str) -> Scenario:
    """Parse and validate one scenario file. Raises ValueError naming the file and the problem."""
    path = Path(path)
    file_name = path.name

    def build_error(problem: str) -> ValueError:
        return ValueError(f"{file_name}: {problem}")

    try:
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise build_error(f"cannot read YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise build_error("top level must be a mapping")

    missing = _TOP_LEVEL_KEYS - data.keys()
    if missing:
        raise build_error(f"missing keys: {sorted(missing)}")
    unknown = data.keys() - _TOP_LEVEL_KEYS
    if unknown:
        raise build_error(f"unknown keys: {sorted(unknown)}")

    scenario_id = _required_string(data, "id", build_error)
    title = _required_string(data, "title", build_error)
    topology = _required_string(data, "topology", build_error)
    alert = _required_string(data, "alert", build_error)

    difficulty = _required_string(data, "difficulty", build_error)
    if difficulty not in DIFFICULTIES:
        raise build_error(f"difficulty {difficulty!r} must be one of {DIFFICULTIES}")

    fixable = data["fixable"]
    if not isinstance(fixable, bool):
        raise build_error("fixable must be true or false")

    inject_steps = [
        _load_inject_step(step, build_error) for step in _required_list(data, "inject", build_error)
    ]
    scenario_checks = [
        _load_scenario_check(check, build_error)
        for check in _required_list(data, "checks", build_error)
    ]

    root_cause_data = data["root_cause"]
    if not isinstance(root_cause_data, dict):
        raise build_error("root_cause must be a mapping")
    if set(root_cause_data) != _ROOT_CAUSE_KEYS:
        raise build_error(f"root_cause keys must be exactly {sorted(_ROOT_CAUSE_KEYS)}")
    category = _required_string(root_cause_data, "category", build_error)
    if category not in FAULT_CATEGORIES:
        raise build_error(f"root_cause category {category!r} is not one of {FAULT_CATEGORIES}")
    root_device = root_cause_data["device"]
    if not isinstance(root_device, str):
        raise build_error("root_cause device must be a string")
    if not root_device and category != "no_fault":
        raise build_error("root_cause device may be empty only for category no_fault")

    if scenario_id != path.stem:
        raise build_error(f"id {scenario_id!r} must match the file name {path.stem!r}")

    return Scenario(
        id=scenario_id,
        title=title,
        topology=topology,
        difficulty=difficulty,
        alert=alert,
        inject=inject_steps,
        checks=scenario_checks,
        root_cause=RootCause(device=root_device, category=category),
        fixable=fixable,
    )


def load_all(scenarios_directory: Path | str | None = None) -> list[Scenario]:
    """Every *.yaml scenario in `scenarios_directory` (default: the bundled scenarios), sorted by id. Raises if the
    directory is missing. Ids are unique because each must match its file name."""
    scenarios_dir = Path(scenarios_directory) if scenarios_directory is not None else SCENARIOS_DIR
    if not scenarios_dir.is_dir():
        raise ValueError(f"scenarios directory does not exist: {scenarios_dir}")
    scenarios = [load_scenario(p) for p in sorted(scenarios_dir.glob("*.yaml"))]
    return sorted(scenarios, key=lambda s: s.id)


def _load_inject_step(raw: Any, build_error: Failer) -> InjectStep:
    if not isinstance(raw, dict):
        raise build_error("each inject step must be a mapping")
    unknown = raw.keys() - _INJECT_STEP_KEYS
    if unknown:
        raise build_error(f"inject step has unknown keys: {sorted(unknown)}")
    device = _required_string(raw, "device", build_error)
    config = _optional_string_list(raw, "config", build_error)
    host = _optional_string_list(raw, "host", build_error)
    revert = _optional_string_list(raw, "revert", build_error)
    if (config is None) == (host is None):
        raise build_error(f"inject step on {device} needs exactly one of config or host")
    if revert is not None and host is None:
        raise build_error(f"revert is only allowed with host, not config (step on {device})")
    if host is not None and not host:
        raise build_error(f"host argv on {device} must not be empty")
    return InjectStep(device=device, config=config, host=host, revert=revert)


def _load_scenario_check(raw: Any, build_error: Failer) -> Check:
    if not isinstance(raw, dict):
        raise build_error("each check must be a mapping")
    if set(raw) != _SCENARIO_CHECK_KEYS:
        raise build_error(
            f"check keys must be exactly {sorted(_SCENARIO_CHECK_KEYS)}, got {sorted(raw)}"
        )
    kind = _required_string(raw, "kind", build_error)
    if kind not in CHECK_KINDS:
        raise build_error(f"check kind {kind!r} is not one of {CHECK_KINDS}")
    device = _required_string(raw, "device", build_error)
    target = _required_string(raw, "target", build_error)
    try:
        if kind == "route_present":
            ipaddress.ip_network(target, strict=True)
        else:
            ipaddress.ip_address(target)
    except ValueError as exc:
        raise build_error(f"check target {target!r} is not a valid address for {kind}") from exc
    return Check(kind=kind, device=device, target=target)  # type: ignore[arg-type]


def _required_string(data: dict[str, Any], key: str, build_error: Failer) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise build_error(f"{key} must be a non-empty string")
    return value


def _required_list(data: dict[str, Any], key: str, build_error: Failer) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise build_error(f"{key} must be a list")
    return value


def _optional_string_list(data: dict[str, Any], key: str, build_error: Failer) -> list[str] | None:
    if key not in data:
        return None
    value = data[key]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise build_error(f"{key} must be a list of strings")
    return list(value)


# --------------------------------------------------------------------- applying


def inject_fault(scenario: Scenario, lab: LabBackend) -> None:
    """Apply the fault's steps in order: config lines through vtysh, host commands raw."""
    for step in scenario.inject:
        if step.config is not None:
            lab.configure(step.device, list(step.config))
        else:
            assert step.host is not None
            lab.host_exec(step.device, list(step.host))


def revert_host_side_fault(scenario: Scenario, lab: LabBackend) -> None:
    """Undo host-level steps that declare a revert, last step first. Config faults are undone
    by the baseline restore, not here."""
    for step in reversed(scenario.inject):
        if step.host is not None and step.revert is not None:
            lab.host_exec(step.device, list(step.revert))
