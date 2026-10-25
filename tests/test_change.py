from babble.harness.audit import AuditLog
from babble.lab.scripted import ScriptedLab
from babble.netops.change import ChangeManager
from babble.types import Check, HarnessConfig, LabError

FIX = ["interface eth3", "no shutdown"]
RATIONALE = "eth3 is administratively down"
PING_DC = Check("ping", "ha", "172.16.0.10")
PING_B = Check("ping", "hb", "192.168.2.10")


class CountingLab(ScriptedLab):
    """ScriptedLab that counts ping calls, to prove checks were (not) evaluated."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pings = 0

    def ping(self, device, target):
        self.pings += 1
        return super().ping(device, target)


class PartialFailLab(ScriptedLab):
    """Applies the lines, then fails, as a vtysh session can after a partial apply."""

    def configure(self, device, lines):
        super().configure(device, lines)
        raise LabError("% Unknown command: bogus")


class BrokenRestoreLab(PartialFailLab):
    def restore(self, device, config):
        raise LabError("restore broke")


def make_lab(**overrides):
    kwargs = {
        "devices": ["r3", "ha", "hb"],
        "hosts": {"ha", "hb"},
        "responses": {},
        "fixed_responses": {},
        "fix": {"r3": FIX},
        "pings": {("ha", "172.16.0.10"): False, ("hb", "192.168.2.10"): True},
        "fixed_pings": {("ha", "172.16.0.10"): True},
    }
    kwargs.update(overrides)
    return kwargs


def make_manager_with_lab(checks, lab_cls=ScriptedLab, config=None, lab_overrides=None):
    lab = lab_cls(**make_lab(**(lab_overrides or {})))
    audit = AuditLog()
    manager = ChangeManager(lab, checks, config or HarnessConfig(), audit)
    return manager, lab, audit


def _event_types(audit):
    return [e.type for e in audit.events]


# ------------------------------------------------------------- applying


def test_applies_and_reports_every_check_passing():
    manager, lab, audit = make_manager_with_lab([PING_DC])
    result = manager.apply("r3", FIX, RATIONALE)

    assert not result.is_error
    assert result.content == "Applied 2 lines on r3. Service checks: 1/1 passing."
    assert lab.state == "fixed"
    applied = audit.of_type("change_applied")
    assert len(applied) == 1
    assert applied[0].data == {
        "device": "r3",
        "lines": FIX,
        "rationale": RATIONALE,
        "checks_passing": 1,
        "checks_total": 1,
    }
    assert manager.change_records == [
        {
            "device": "r3",
            "lines": FIX,
            "rationale": RATIONALE,
            "rolled_back": False,
            "rollback_failed": False,
            "regressions": [],
        }
    ]


def test_lists_still_failing_checks():
    unreachable = Check("ping", "hb", "10.99.0.1")  # not in the lab's ping table, so it fails
    manager, _, audit = make_manager_with_lab([PING_DC, unreachable])
    result = manager.apply("r3", FIX, RATIONALE)

    assert result.content == (
        "Applied 2 lines on r3. Service checks: 1/2 passing. Still failing: ping hb -> 10.99.0.1."
    )
    assert audit.of_type("change_applied")[0].data["checks_passing"] == 1


def test_caps_listed_checks_at_eight_then_counts_the_rest():
    checks = [Check("ping", "ha", f"10.9.9.{i}") for i in range(1, 11)]
    manager, _, audit = make_manager_with_lab(checks)
    result = manager.apply("r3", FIX, RATIONALE)

    assert result.content.startswith("Applied 2 lines on r3. Service checks: 0/10 passing.")
    assert "10.9.9.8, " in result.content
    assert "10.9.9.9" not in result.content
    assert result.content.endswith("and 2 more.")
    assert audit.of_type("change_applied")[0].data["checks_total"] == 10


def test_empty_check_list_is_safe_and_never_regresses():
    manager, lab, audit = make_manager_with_lab(
        [], lab_overrides={"pings": {("ha", "172.16.0.10"): True}}
    )
    result = manager.apply("r3", FIX, RATIONALE)

    assert not result.is_error
    assert result.content == (
        "Applied 2 lines on r3. No service checks are configured, so nothing was verified."
    )
    assert audit.of_type("change_applied")[0].data["checks_total"] == 0
    assert audit.of_type("rollback") == []
    assert lab.applied_lines("r3") == FIX


def test_verify_off_applies_without_evaluating_checks():
    config = HarnessConfig(verify_after_change=False)
    manager, lab, audit = make_manager_with_lab([PING_DC], lab_cls=CountingLab, config=config)
    result = manager.apply("r3", FIX, RATIONALE)

    assert not result.is_error
    assert lab.pings == 0
    assert "not evaluated because verify_after_change is off" in result.content
    assert audit.of_type("change_applied")[0].data["checks_passing"] is None
    assert lab.state == "fixed"


# ------------------------------------------------------------ rejections


def test_blocked_lines_are_rejected_and_nothing_is_applied():
    manager, lab, audit = make_manager_with_lab([PING_DC])
    result = manager.apply("r3", ["interface eth3", "hostname evil"], RATIONALE)

    assert result.is_error
    assert "hostname evil" in result.content
    assert "Nothing was changed" in result.content
    assert lab.configure_calls == []
    assert lab.applied_lines("r3") == []
    assert _event_types(audit) == ["change_rejected"]
    assert audit.of_type("change_rejected")[0].data["device"] == "r3"
    assert manager.change_records == []


def test_empty_line_list_is_rejected():
    manager, lab, _ = make_manager_with_lab([PING_DC])
    result = manager.apply("r3", [], RATIONALE)

    assert result.is_error
    assert lab.configure_calls == []


def test_unknown_device_is_an_error_result():
    manager, _, audit = make_manager_with_lab([PING_DC])
    result = manager.apply("nope", FIX, RATIONALE)

    assert result.is_error
    assert "unknown device" in result.content
    assert audit.events == []


def test_cli_error_restores_the_snapshot_and_reports_it():
    manager, lab, audit = make_manager_with_lab([PING_DC], lab_cls=PartialFailLab)
    result = manager.apply("r3", FIX, RATIONALE)

    assert result.is_error
    assert "% Unknown command: bogus" in result.content
    assert "restored to its previous state" in result.content
    assert lab.applied_lines("r3") == []
    assert _event_types(audit) == ["change_rejected"]
    assert manager.change_records == []


def test_cli_error_reports_when_restore_also_fails():
    manager, _, _ = make_manager_with_lab([PING_DC], lab_cls=BrokenRestoreLab)
    result = manager.apply("r3", FIX, RATIONALE)

    assert result.is_error
    assert "Restoring the previous config also failed: restore broke" in result.content


# ------------------------------------------------------- regressions


def _regression_overrides():
    # ping hb passes while the lab is broken and fails once the fix lands.
    return {
        "pings": {("ha", "172.16.0.10"): False, ("hb", "192.168.2.10"): True},
        "fixed_pings": {("hb", "192.168.2.10"): False},
    }


def test_regression_rolls_back_automatically():
    manager, lab, audit = make_manager_with_lab([PING_B], lab_overrides=_regression_overrides())
    result = manager.apply("r3", FIX, RATIONALE)

    assert result.is_error
    assert result.content == (
        "Change on r3 was rolled back automatically because it broke: "
        "ping hb -> 192.168.2.10. The device is back to its previous config."
    )
    assert lab.applied_lines("r3") == []
    assert lab.state == "broken"
    assert _event_types(audit) == ["rollback"]
    assert audit.of_type("rollback")[0].data == {
        "device": "r3",
        "reason": "service checks regressed",
        "regressions": ["ping hb -> 192.168.2.10"],
    }
    assert manager.change_records[0]["rolled_back"] is True
    assert manager.change_records[0]["regressions"] == ["ping hb -> 192.168.2.10"]


def test_regression_is_kept_when_auto_rollback_is_off():
    config = HarnessConfig(auto_rollback=False)
    manager, lab, audit = make_manager_with_lab(
        [PING_B, PING_DC], config=config, lab_overrides=_regression_overrides()
    )
    result = manager.apply("r3", FIX, RATIONALE)

    assert not result.is_error
    assert "Regressed, kept because auto_rollback is off: ping hb -> 192.168.2.10" in result.content
    assert "Still failing: ping hb -> 192.168.2.10" in result.content
    assert lab.applied_lines("r3") == FIX
    assert _event_types(audit) == ["change_applied"]
    assert manager.change_records[0]["rolled_back"] is False
    assert manager.change_records[0]["regressions"] == ["ping hb -> 192.168.2.10"]


def test_failing_before_and_after_is_not_a_regression():
    manager, _lab, audit = make_manager_with_lab([PING_B], lab_overrides={"pings": {}})
    result = manager.apply("r3", FIX, RATIONALE)

    assert not result.is_error
    assert audit.of_type("rollback") == []
    assert manager.change_records[0]["regressions"] == []


class RestoreAlwaysFailsLab(ScriptedLab):
    def restore(self, device, config):
        raise LabError("restore broke")


class RestoreFailsOnceLab(ScriptedLab):
    """The first restore fails (the automatic rollback); later restores succeed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.restores_attempted = 0

    def restore(self, device, config):
        self.restores_attempted += 1
        if self.restores_attempted == 1:
            raise LabError("restore broke")
        super().restore(device, config)


def test_failed_automatic_rollback_is_recorded_as_failed_not_rolled_back():
    manager, _, audit = make_manager_with_lab(
        [PING_B],
        lab_cls=RestoreAlwaysFailsLab,
        lab_overrides=_regression_overrides(),
    )
    result = manager.apply("r3", FIX, RATIONALE)

    assert result.is_error
    assert "automatic rollback failed: restore broke" in result.content
    assert manager.change_records == [
        {
            "device": "r3",
            "lines": FIX,
            "rationale": RATIONALE,
            "rolled_back": False,
            "rollback_failed": True,
            "regressions": ["ping hb -> 192.168.2.10"],
        }
    ]
    assert _event_types(audit) == ["rollback_failed"]
    assert audit.of_type("rollback_failed")[0].data == {
        "device": "r3",
        "regressions": ["ping hb -> 192.168.2.10"],
        "restore_error": "restore broke",
    }


# ------------------------------------------------------------- rollback


def test_rollback_restores_the_config_before_the_last_kept_change():
    manager, lab, audit = make_manager_with_lab([PING_DC])
    manager.apply("r3", FIX, RATIONALE)
    assert lab.state == "fixed"

    result = manager.rollback("r3")

    assert not result.is_error
    assert lab.applied_lines("r3") == []
    assert lab.state == "broken"
    assert manager.change_records[0]["rolled_back"] is True
    assert audit.of_type("rollback")[0].data["reason"] == "rolled back on request"


def test_rollback_walks_back_one_change_at_a_time():
    manager, lab, _ = make_manager_with_lab([])
    manager.apply("r3", ["interface eth3"], RATIONALE)
    manager.apply("r3", ["no shutdown"], RATIONALE)
    assert lab.applied_lines("r3") == ["interface eth3", "no shutdown"]

    manager.rollback("r3")
    assert lab.applied_lines("r3") == ["interface eth3"]
    manager.rollback("r3")
    assert lab.applied_lines("r3") == []

    result = manager.rollback("r3")
    assert result.is_error
    assert result.content == "No kept change on r3 to roll back."


def test_rollback_without_a_change_is_an_error():
    manager, _, _ = make_manager_with_lab([PING_DC])
    result = manager.rollback("r3")

    assert result.is_error
    assert "No kept change on r3" in result.content


def test_rollback_is_not_recorded_for_a_change_that_auto_rolled_back():
    manager, _, _ = make_manager_with_lab([PING_B], lab_overrides=_regression_overrides())
    manager.apply("r3", FIX, RATIONALE)

    result = manager.rollback("r3")
    assert result.is_error
    assert "No kept change" in result.content


class SlowRegressionLab(ScriptedLab):
    """A change that looks fine at first: site B's ping only fails from the third evaluation on,
    the way a BGP session takes a few seconds to drop."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.site_b_pings_after_change = 0
        self.change_applied = False

    def configure(self, device, lines):
        self.change_applied = True
        return super().configure(device, lines)

    def restore(self, device, config):
        self.change_applied = False
        return super().restore(device, config)

    def ping(self, device, target):
        if self.change_applied and device == "hb":
            self.site_b_pings_after_change += 1
            if self.site_b_pings_after_change >= 3:
                return super().ping(device, "unreachable")
        return super().ping(device, target)


def test_regression_that_appears_after_the_change_is_caught_and_rolled_back():
    lab = SlowRegressionLab(
        devices=["r3", "hb"],
        hosts={"hb"},
        responses={},
        pings={("hb", "192.168.2.10"): True},
    )
    manager = ChangeManager(
        lab,
        [PING_B],
        HarnessConfig(),
        AuditLog(),
        convergence_wait_seconds=0.05,
        poll_interval_seconds=0.01,
    )
    result = manager.apply("r3", ["router bgp 65000"], "test")
    assert result.is_error
    assert "rolled back" in result.content
    assert manager.change_records[-1]["rolled_back"] is True


def test_convergence_wait_returns_once_checks_pass_after_minimum_observation():
    sleeps = []
    lab = ScriptedLab(**make_lab())
    manager = ChangeManager(
        lab,
        [PING_DC],
        HarnessConfig(),
        AuditLog(),
        convergence_wait_seconds=0.0,
        sleep=sleeps.append,
    )
    result = manager.apply("r3", FIX, RATIONALE)
    assert not result.is_error
    assert sleeps == []


def test_a_failed_automatic_rollback_can_be_retried_with_rollback():
    manager, lab, _ = make_manager_with_lab(
        [PING_B],
        lab_cls=RestoreFailsOnceLab,
        lab_overrides=_regression_overrides(),
    )
    failed = manager.apply("r3", FIX, RATIONALE)
    assert failed.is_error
    assert lab.applied_lines("r3") == FIX  # the regressing change is still on the device

    retried = manager.rollback("r3")

    assert not retried.is_error
    assert lab.applied_lines("r3") == []
    assert manager.change_records[0]["rolled_back"] is True
    assert manager.rollback("r3").is_error  # nothing left to undo
