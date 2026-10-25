"""Load a lab topology from labs/<name>/meta.yaml. The containerlab file sits beside it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from faultline.types import Check

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LABS_DIR = REPO_ROOT / "labs"


@dataclass
class Topology:
    name: str
    dir: Path
    devices: dict[str, dict[str, Any]]
    links: list[dict[str, Any]]
    health_checks: list[Check]

    @property
    def clab_file(self) -> Path:
        return self.dir / f"{self.name}.clab.yml"

    def routers(self) -> list[str]:
        return [n for n, d in self.devices.items() if d.get("kind") == "router"]

    def hosts(self) -> list[str]:
        return [n for n, d in self.devices.items() if d.get("kind") == "host"]

    def describe(self) -> str:
        lines = [f"Topology {self.name}", "Routers:"]
        for name in self.routers():
            d = self.devices[name]
            lines.append(
                f"  {name} ({d.get('role', '-')}, AS {d.get('asn', '-')}, "
                f"loopback {d.get('loopback', '-')})"
            )
        lines.append("Hosts:")
        for name in self.hosts():
            d = self.devices[name]
            lines.append(f"  {name} ({d.get('role', '-')}, ip {d.get('ip', '-')})")
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
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    return Topology(
        name=meta.get("name", name),
        dir=lab_dir,
        devices=dict(meta.get("devices") or {}),
        links=list(meta.get("links") or []),
        health_checks=[
            Check(kind=h["kind"], device=h["device"], target=h["target"])
            for h in meta.get("health_checks") or []
        ],
    )
