"""Live: the shared-prefix-list trap breaks site B when fixed the obvious way, and the change
manager rolls that fix back on real FRR routers. Needs the metro lab running."""

from __future__ import annotations

import pytest

from babble.bench.runner import service_checks_for
from babble.bench.scenario import SCENARIOS_DIR, inject_fault, load_scenario
from babble.cli import CONVERGENCE_WAIT_SECONDS, default_lab_at_baseline
from babble.harness.audit import AuditLog
from babble.netops.change import ChangeManager
from babble.netops.checks import evaluate
from babble.polling import poll_until
from babble.types import Check, HarnessConfig

pytestmark = pytest.mark.live

SITE_A_TO_DC = Check(kind="ping", device="ha", target="172.16.0.10")
SITE_B_TO_DC = Check(kind="ping", device="hb", target="172.16.0.10")
OBVIOUS_FIX = ["ip prefix-list CORE-ALLOWED seq 10 permit 172.16.0.0/24"]


def test_obvious_fix_for_the_trap_is_rolled_back_on_live_routers() -> None:
    topology, manager = default_lab_at_baseline()
    scenario = load_scenario(SCENARIOS_DIR / "shared-prefix-list-trap.yaml")
    try:
        inject_fault(scenario, manager.lab)
        assert poll_until(lambda: not evaluate(SITE_A_TO_DC, manager.lab), 60, 3)
        assert evaluate(SITE_B_TO_DC, manager.lab)
        audit = AuditLog()
        change_manager = ChangeManager(
            manager.lab,
            service_checks_for(topology, scenario),
            HarnessConfig(),
            audit,
            convergence_wait_seconds=CONVERGENCE_WAIT_SECONDS,
        )

        result = change_manager.apply("r3", OBVIOUS_FIX, "add the DC prefix to the allow-list")

        assert result.is_error, result.content
        record = change_manager.change_records[-1]
        assert record["rolled_back"] is True
        assert SITE_B_TO_DC.describe() in record["regressions"]
        assert evaluate(SITE_B_TO_DC, manager.lab)
        assert not evaluate(SITE_A_TO_DC, manager.lab)
    finally:
        manager.reset()
