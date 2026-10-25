"""Service-level checks, evaluated against a LabBackend using FRR's JSON output.

`evaluate` is the only entry point the harness and the bench need. A check that cannot be
judged on a real lab counts as failing: a LabError, unparseable output (ValueError), a missing
key (KeyError) or an unexpected JSON shape (TypeError). Anything else propagates, and an unknown
check kind raises ValueError, because those are bugs rather than failed checks.

Usage:

    from babble.netops.checks import evaluate, evaluate_all, evaluate_all_after_convergence
    from babble.types import Check

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

from babble.types import Check, LabBackend, LabError

# Even when every check passes early, keep watching this long: a bad change can take a few
# seconds to tear down a session.
MINIMUM_OBSERVATION_SECONDS = 10.0


def evaluate(check: Check, lab: LabBackend) -> bool:
    """True when `check` passes on `lab`. False when the lab could not show it passing (see the
    module docstring). Raises ValueError for an unknown check kind."""
    evaluator = _EVALUATORS.get(check.kind)
    if evaluator is None:
        raise ValueError(f"unknown check kind: {check.kind!r}")
    try:
        return bool(evaluator(check, lab))
    except _FAILED_CHECK_EXCEPTIONS:
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
    polling_started_at = time.monotonic()
    minimum_observation = min(MINIMUM_OBSERVATION_SECONDS, convergence_wait_seconds)
    while True:
        latest_results = evaluate_all(checks, lab)
        seconds_since_polling_started = time.monotonic() - polling_started_at
        convergence_wait_is_over = seconds_since_polling_started >= convergence_wait_seconds
        all_checks_passing = all(latest_results.values())
        if convergence_wait_is_over or (
            all_checks_passing and seconds_since_polling_started >= minimum_observation
        ):
            return latest_results
        sleep(poll_interval_seconds)


def _ping(check: Check, lab: LabBackend) -> bool:
    return lab.ping(check.device, check.target).ok


def _bgp_established(check: Check, lab: LabBackend) -> bool:
    # Real output: {"10.2.0.1": {"remoteAs": 65002, "bgpState": "Established", ...}}
    parsed_output = _load_json(lab.show(check.device, f"show bgp neighbors {check.target} json"))
    neighbor_entry = parsed_output.get(check.target)
    return isinstance(neighbor_entry, dict) and neighbor_entry.get("bgpState") == "Established"


def _route_present(check: Check, lab: LabBackend) -> bool:
    # Real output: {"172.16.0.0/24": [{"prefix": ..., "protocol": "bgp", ...}]}; empty if absent.
    parsed_output = _load_json(lab.show(check.device, f"show ip route {check.target} json"))
    routes_for_prefix = parsed_output.get(check.target)
    return isinstance(routes_for_prefix, list) and len(routes_for_prefix) > 0


# What a real lab raises when a check cannot be judged: LabError from the backend, ValueError
# from JSON decoding, KeyError and TypeError from an unexpected output shape.
_FAILED_CHECK_EXCEPTIONS = (LabError, ValueError, KeyError, TypeError)

_OSPF_STATE_FIELDS = ("nbrState", "state", "converged")


def _ospf_full(check: Check, lab: LabBackend) -> bool:
    # Real output: {"neighbors": {"10.255.0.2": [{"nbrState": "Full/DR", "ifaceName": ...}]}}
    parsed_output = _load_json(lab.show(check.device, "show ip ospf neighbor json"))
    neighbor_entries_by_router_id = parsed_output["neighbors"]
    if not isinstance(neighbor_entries_by_router_id, dict):
        raise TypeError("expected 'neighbors' to be a JSON object")
    neighbor_entries = neighbor_entries_by_router_id.get(check.target)
    if not isinstance(neighbor_entries, list):
        return False
    for neighbor_entry in neighbor_entries:
        if not isinstance(neighbor_entry, dict):
            continue
        for state_field in _OSPF_STATE_FIELDS:
            state_value = neighbor_entry.get(state_field)
            if isinstance(state_value, str) and state_value.startswith("Full"):
                return True
    return False


def _load_json(command_output: str) -> dict[str, Any]:
    """Parse command output. Anything that is not a JSON object raises, so evaluate fails."""
    parsed = json.loads(command_output)
    if not isinstance(parsed, dict):
        raise TypeError("expected a JSON object")
    return parsed


_EVALUATORS = {
    "ping": _ping,
    "bgp_established": _bgp_established,
    "route_present": _route_present,
    "ospf_full": _ospf_full,
}
