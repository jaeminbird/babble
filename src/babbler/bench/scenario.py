"""Fault scenarios: a YAML file describes one fault to inject into a lab, the checks it breaks,
and the root cause a diagnosis must name. The bench loads these; the agent never sees them."""

from __future__ import annotations

import ipaddress
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args

import yaml

from babbler.types import FAULT_CATEGORIES, Check, CheckKind, LabBackend

Failer = Callable[[str], ValueError]

SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"
DIFFICULTIES = ("easy", "medium", "hard")
CHECK_KINDS: tuple[str, ...] = get_args(CheckKind)

_SCENARIO_KEYS = {
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
_STEP_KEYS = {"device", "config", "host", "revert"}
_CHECK_KEYS = {"kind", "device", "target"}
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
    inject: list[InjectStep] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    root_cause: RootCause = field(default_factory=lambda: RootCause("", "other"))
    fixable: bool = True


# ----------------------------------------------------------------------- loading


def load_scenario(path: Path | str) -> Scenario:
    """Parse and validate one scenario file. Raises ValueError naming the file and the problem."""
    path = Path(path)
    name = path.name

    def fail(problem: str) -> ValueError:
        return ValueError(f"{name}: {problem}")

    try:
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise fail(f"cannot read YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise fail("top level must be a mapping")

    missing = _SCENARIO_KEYS - data.keys()
    if missing:
        raise fail(f"missing keys: {sorted(missing)}")
    unknown = data.keys() - _SCENARIO_KEYS
    if unknown:
        raise fail(f"unknown keys: {sorted(unknown)}")

    scenario_id = _str(data, "id", fail)
    title = _str(data, "title", fail)
    topology = _str(data, "topology", fail)
    alert = _str(data, "alert", fail)

    difficulty = _str(data, "difficulty", fail)
    if difficulty not in DIFFICULTIES:
        raise fail(f"difficulty {difficulty!r} must be one of {DIFFICULTIES}")

    fixable = data["fixable"]
    if not isinstance(fixable, bool):
        raise fail("fixable must be true or false")

    inject = [_load_step(step, fail) for step in _list(data, "inject", fail)]
    checks = [_load_check(check, fail) for check in _list(data, "checks", fail)]

    root = data["root_cause"]
    if not isinstance(root, dict):
        raise fail("root_cause must be a mapping")
    if set(root) != _ROOT_CAUSE_KEYS:
        raise fail(f"root_cause keys must be exactly {sorted(_ROOT_CAUSE_KEYS)}")
    category = _str(root, "category", fail)
    if category not in FAULT_CATEGORIES:
        raise fail(f"root_cause category {category!r} is not one of {FAULT_CATEGORIES}")
    root_device = root["device"]
    if not isinstance(root_device, str):
        raise fail("root_cause device must be a string")
    if not root_device and category != "no_fault":
        raise fail("root_cause device may be empty only for category no_fault")

    return Scenario(
        id=scenario_id,
        title=title,
        topology=topology,
        difficulty=difficulty,
        alert=alert,
        inject=inject,
        checks=checks,
        root_cause=RootCause(device=root_device, category=category),
        fixable=fixable,
    )


def load_all(dir: Path | str | None = None) -> list[Scenario]:
    """Every *.yaml scenario in `dir` (default: the bundled scenarios), sorted by id."""
    root = Path(dir) if dir is not None else SCENARIOS_DIR
    scenarios = [load_scenario(p) for p in sorted(root.glob("*.yaml"))]
    return sorted(scenarios, key=lambda s: s.id)


def _load_step(raw: Any, fail: Failer) -> InjectStep:
    if not isinstance(raw, dict):
        raise fail("each inject step must be a mapping")
    unknown = raw.keys() - _STEP_KEYS
    if unknown:
        raise fail(f"inject step has unknown keys: {sorted(unknown)}")
    device = _str(raw, "device", fail)
    config = _opt_list(raw, "config", fail)
    host = _opt_list(raw, "host", fail)
    revert = _opt_list(raw, "revert", fail)
    if (config is None) == (host is None):
        raise fail(f"inject step on {device} needs exactly one of config or host")
    if revert is not None and host is None:
        raise fail(f"revert is only allowed with host, not config (step on {device})")
    if host is not None and not host:
        raise fail(f"host argv on {device} must not be empty")
    return InjectStep(device=device, config=config, host=host, revert=revert)


def _load_check(raw: Any, fail: Failer) -> Check:
    if not isinstance(raw, dict):
        raise fail("each check must be a mapping")
    if set(raw) != _CHECK_KEYS:
        raise fail(f"check keys must be exactly {sorted(_CHECK_KEYS)}, got {sorted(raw)}")
    kind = _str(raw, "kind", fail)
    if kind not in CHECK_KINDS:
        raise fail(f"check kind {kind!r} is not one of {CHECK_KINDS}")
    device = _str(raw, "device", fail)
    target = _str(raw, "target", fail)
    try:
        if kind == "route_present":
            ipaddress.ip_network(target, strict=True)
        else:
            ipaddress.ip_address(target)
    except ValueError as exc:
        raise fail(f"check target {target!r} is not a valid address for {kind}") from exc
    return Check(kind=kind, device=device, target=target)  # type: ignore[arg-type]


def _str(data: dict[str, Any], key: str, fail: Failer) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise fail(f"{key} must be a non-empty string")
    return value


def _list(data: dict[str, Any], key: str, fail: Failer) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise fail(f"{key} must be a list")
    return value


def _opt_list(data: dict[str, Any], key: str, fail: Failer) -> list[str] | None:
    if key not in data:
        return None
    value = data[key]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise fail(f"{key} must be a list of strings")
    return list(value)


# --------------------------------------------------------------------- applying


def inject(scenario: Scenario, lab: LabBackend) -> None:
    """Apply the fault's steps in order: config lines through vtysh, host commands raw."""
    for step in scenario.inject:
        if step.config is not None:
            lab.configure(step.device, list(step.config))
        else:
            assert step.host is not None
            lab.host_exec(step.device, list(step.host))


def revert_host(scenario: Scenario, lab: LabBackend) -> None:
    """Undo host-level steps that declare a revert, last step first. Config faults are undone
    by the baseline restore, not here."""
    for step in reversed(scenario.inject):
        if step.host is not None and step.revert is not None:
            lab.host_exec(step.device, list(step.revert))
