"""Service-level checks, evaluated against a LabBackend using FRR's JSON output.

`evaluate` is the only entry point the harness and the bench need. It never raises: a lab
error, unparseable output or an unexpected shape all count as the check failing.

Usage:

    from faultline.netops.checks import evaluate, evaluate_all, evaluate_all_after_convergence
    from faultline.types import Check

    check = Check("bgp_established", "r1", "10.2.0.1")
    evaluate(check, lab)                       # True when r1 has an Established session
    evaluate_all([check, ...], lab)            # {Check: bool, ...}
    evaluate_all_after_convergence([check, ...], lab, 30, 3)  # waits for convergence first
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from faultline.types import Check, LabBackend

# Even when every check passes early, keep watching this long: a bad change can take a few
# seconds to tear down a session.
MINIMUM_OBSERVATION_SECONDS = 10.0


def evaluate(check: Check, lab: LabBackend) -> bool:
    """True when `check` passes on `lab`. Never raises."""
    try:
        return bool(_EVALUATORS[check.kind](check, lab))
    except Exception:  # noqa: BLE001 - evaluate must never raise; any failure is a failed check
        return False


def evaluate_all(checks: list[Check], lab: LabBackend) -> dict[Check, bool]:
    """Evaluate each check once. Duplicate checks share one result."""
    return {check: evaluate(check, lab) for check in checks}


def evaluate_all_after_convergence(
    checks: list[Check],
    lab: LabBackend,
    convergence_wait_seconds: float,
    poll_interval_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[Check, bool]:
    """Poll the checks until they all pass (after a minimum observation window) or the
    convergence wait runs out, and return the last evaluation. With a zero wait, evaluates once."""
    started = time.monotonic()
    minimum = min(MINIMUM_OBSERVATION_SECONDS, convergence_wait_seconds)
    while True:
        results = evaluate_all(checks, lab)
        elapsed = time.monotonic() - started
        if elapsed >= convergence_wait_seconds:
            return results
        if all(results.values()) and elapsed >= minimum:
            return results
        sleep(poll_interval_seconds)


def _ping(check: Check, lab: LabBackend) -> bool:
    return lab.ping(check.device, check.target).ok


def _bgp_established(check: Check, lab: LabBackend) -> bool:
    # Real output: {"10.2.0.1": {"remoteAs": 65002, "bgpState": "Established", ...}}
    data = _load_json(lab.show(check.device, f"show bgp neighbors {check.target} json"))
    entry = data.get(check.target)
    return isinstance(entry, dict) and entry.get("bgpState") == "Established"


def _route_present(check: Check, lab: LabBackend) -> bool:
    # Real output: {"172.16.0.0/24": [{"prefix": ..., "protocol": "bgp", ...}]}; empty if absent.
    data = _load_json(lab.show(check.device, f"show ip route {check.target} json"))
    routes = data.get(check.target)
    return isinstance(routes, list) and len(routes) > 0


_FULL_FIELDS = ("nbrState", "state", "converged")


def _ospf_full(check: Check, lab: LabBackend) -> bool:
    # Real output: {"neighbors": {"10.255.0.2": [{"nbrState": "Full/DR", "ifaceName": ...}]}}
    data = _load_json(lab.show(check.device, "show ip ospf neighbor json"))
    neighbors = data["neighbors"]
    entries = neighbors.get(check.target)
    if not isinstance(entries, list):
        return False
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for field in _FULL_FIELDS:
            value = entry.get(field)
            if isinstance(value, str) and value.startswith("Full"):
                return True
    return False


def _load_json(text: str) -> dict[str, Any]:
    """Parse command output. Anything that is not a JSON object raises, so evaluate fails."""
    data = json.loads(text)
    if not isinstance(data, dict):
        raise TypeError("expected a JSON object")
    return data


_EVALUATORS = {
    "ping": _ping,
    "bgp_established": _bgp_established,
    "route_present": _route_present,
    "ospf_full": _ospf_full,
}
