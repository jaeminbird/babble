"""Load a lab topology from labs/<name>/meta.yaml. The containerlab file sits beside it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from babble.types import Check

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LABS_DIR = REPO_ROOT / "labs"


@dataclass
class Topology:
    name: str
    lab_directory: Path
    devices: dict[str, dict[str, Any]]
    links: list[dict[str, Any]]
    health_checks: list[Check]

    @property
    def clab_file(self) -> Path:
        return self.lab_directory / f"{self.name}.clab.yml"

    def routers(self) -> list[str]:
        return self._device_names_of_kind("router")

    def hosts(self) -> list[str]:
        return self._device_names_of_kind("host")

    def _device_names_of_kind(self, kind: str) -> list[str]:
        return [
            device_name
            for device_name, device_attributes in self.devices.items()
            if device_attributes.get("kind") == kind
        ]

    def describe(self) -> str:
        lines = [f"Topology {self.name}", "Routers:"]
        for router_name in self.routers():
            device_attributes = self.devices[router_name]
            lines.append(
                f"  {router_name} ({device_attributes.get('role', '-')}, "
                f"AS {device_attributes.get('asn', '-')}, "
                f"loopback {device_attributes.get('loopback', '-')})"
            )
        lines.append("Hosts:")
        for host_name in self.hosts():
            device_attributes = self.devices[host_name]
            lines.append(
                f"  {host_name} ({device_attributes.get('role', '-')}, "
                f"ip {device_attributes.get('ip', '-')})"
            )
        lines.append("Links:")
        for link in self.links:
            lines.append(f"  {link['a']} <-> {link['b']}  {link['subnet']}")
        lines.append("Health checks:")
        for check in self.health_checks:
            lines.append(f"  {check.describe()}")
        return "\n".join(lines)


def load_topology(name: str, labs_dir: Path | None = None) -> Topology:
    lab_dir = (labs_dir or DEFAULT_LABS_DIR) / name
    meta_path = lab_dir / "meta.yaml"
    if not meta_path.is_file():
        raise FileNotFoundError(f"no lab named {name!r}: {meta_path} missing")
    meta_yaml = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta_yaml, dict) or not meta_yaml:
        raise ValueError(f"{meta_path}: meta.yaml must be a non-empty mapping")
    return Topology(
        name=meta_yaml.get("name", name),
        lab_directory=lab_dir,
        devices=dict(meta_yaml.get("devices") or {}),
        links=list(meta_yaml.get("links") or []),
        health_checks=[
            Check(kind=check_spec["kind"], device=check_spec["device"], target=check_spec["target"])
            for check_spec in meta_yaml.get("health_checks") or []
        ],
    )
