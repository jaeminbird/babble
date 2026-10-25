import json

import pytest

from faultline.bench.report import (
    load_results,
    pass_at_k,
    pass_hat_k,
    per_scenario_table,
    render_markdown_table,
    summarize_by_preset,
)
from faultline.bench.runner import TrialResult


def trial(scenario_id="a", preset_name="full", fixable=True, **fields):
    """A TrialResult as the JSONL file holds it, with the given fields overridden."""
    return TrialResult(
        scenario_id=scenario_id,
        preset_name=preset_name,
        trial_index=1,
        model="claude-opus-5-5",
        fixable=fixable,
        **fields,
    ).to_dict()


# ------------------------------------------------------------------ pass^k


def test_pass_hat_k_is_the_chance_all_k_trials_succeed():
    # C(3,2) / C(4,2) = 3 / 6
    assert pass_hat_k(successes=3, trials=4, k=2) == pytest.approx(0.5)
    # C(2,1) / C(3,1) = 2 / 3
    assert pass_hat_k(successes=2, trials=3, k=1) == pytest.approx(2 / 3)


def test_pass_hat_k_is_zero_when_fewer_than_k_trials_succeed_and_one_when_all_do():
    assert pass_hat_k(successes=2, trials=3, k=3) == 0.0
    assert pass_hat_k(successes=4, trials=4, k=4) == 1.0


def test_pass_at_k_is_the_chance_at_least_one_of_k_trials_succeeds():
    # 1 - C(3,2) / C(4,2) = 1 - 3 / 6
    assert pass_at_k(successes=1, trials=4, k=2) == pytest.approx(0.5)
    assert pass_at_k(successes=0, trials=3, k=2) == 0.0
    assert pass_at_k(successes=3, trials=3, k=3) == 1.0


@pytest.mark.parametrize(
    ("successes", "trials", "k"),
    [(4, 3, 1), (-1, 3, 1), (1, 3, 0), (1, 3, 4)],
)
def test_pass_k_rejects_impossible_arguments(successes, trials, k):
    with pytest.raises(ValueError):
        pass_hat_k(successes, trials, k)
    with pytest.raises(ValueError):
        pass_at_k(successes, trials, k)


# ------------------------------------------------------------- loading


def test_load_results_reads_one_dict_per_nonblank_line(tmp_path):
    path = tmp_path / "trials.jsonl"
    path.write_text(
        json.dumps({"scenario_id": "a"}) + "\n\n" + json.dumps({"scenario_id": "b"}) + "\n",
        encoding="utf-8",
    )
    assert load_results(path) == [{"scenario_id": "a"}, {"scenario_id": "b"}]


# ----------------------------------------------------------- summaries


def test_summary_rates_and_pass_k_for_one_preset():
    # Scenario a: 3 trials, 2 solved. Scenario b: 2 trials, both solved. Four solved of five.
    results = [
        trial(
            "a",
            solved=True,
            root_cause_correct=True,
            cost_usd=0.1,
            turns=4,
            input_tokens=100,
            cache_read_input_tokens=300,
        ),
        trial(
            "a",
            solved=True,
            root_cause_correct=True,
            cost_usd=0.1,
            turns=4,
            input_tokens=100,
            cache_read_input_tokens=300,
        ),
        trial(
            "a",
            solved=False,
            root_cause_correct=False,
            cost_usd=0.1,
            turns=4,
            input_tokens=100,
            cache_read_input_tokens=300,
        ),
        trial(
            "b",
            solved=True,
            root_cause_correct=True,
            cost_usd=0.1,
            turns=4,
            input_tokens=100,
            cache_read_input_tokens=300,
        ),
        trial(
            "b",
            solved=True,
            root_cause_correct=True,
            cost_usd=0.1,
            turns=4,
            input_tokens=100,
            cache_read_input_tokens=300,
        ),
    ]

    (row,) = summarize_by_preset(results)

    assert row["preset"] == "full"
    assert row["trials"] == 5
    assert row["solve_rate"] == pytest.approx(4 / 5)
    assert row["root_cause_accuracy"] == pytest.approx(4 / 5)
    # pass^1 per scenario: 2/3 and 1, averaged.
    assert row["pass_hat_1"] == pytest.approx((2 / 3 + 1) / 2)
    # pass^k with k the scenario's trial count: all three for a fails, both for b succeeds.
    assert row["pass_hat_k_max"] == pytest.approx((0.0 + 1.0) / 2)
    assert row["mean_cost_usd"] == pytest.approx(0.1)
    assert row["cost_per_solved_task"] == pytest.approx(0.5 / 4)
    assert row["mean_turns"] == pytest.approx(4.0)
    # 1500 cache reads out of 500 + 1500 + 0 input-side tokens across five trials.
    assert row["cache_hit_rate"] == pytest.approx(1500 / 2000)


def test_summary_counts_unfixable_changes_collateral_unapproved_rollbacks_and_errors():
    results = [
        trial("control", fixable=False, solved=True, root_cause_correct=True),
        trial(
            "control",
            fixable=False,
            solved=False,
            unnecessary_change=True,
            changes_kept=1,
            changes_applied=1,
        ),
        trial(
            "fix",
            solved=True,
            collateral_damage=["ping hb -> 172.16.0.10"],
            unapproved_changes=2,
            changes_applied=3,
            changes_rolled_back=1,
            changes_kept=2,
        ),
        trial("fix", solved=False, error="fault did not become visible"),
    ]

    (row,) = summarize_by_preset(results)

    assert row["unnecessary_change_rate"] == pytest.approx(1 / 2)
    assert row["collateral_damage_count"] == 1
    assert row["unapproved_changes"] == 2
    assert row["rollbacks"] == 1
    assert row["errors"] == 1


def test_unnecessary_change_rate_and_cache_hit_rate_are_none_without_their_denominators():
    (row,) = summarize_by_preset([trial("a", solved=True)])

    assert row["unnecessary_change_rate"] is None
    assert row["cache_hit_rate"] is None


def test_cost_per_solved_task_is_none_when_nothing_was_solved():
    (row,) = summarize_by_preset([trial("a", solved=False, cost_usd=0.2)])

    assert row["cost_per_solved_task"] is None
    assert row["mean_cost_usd"] == pytest.approx(0.2)


def test_summaries_keep_presets_in_first_appearance_order():
    results = [
        trial("a", preset_name="no_safety", solved=True),
        trial("a", preset_name="full", solved=False),
        trial("a", preset_name="no_safety", solved=True),
    ]

    rows = summarize_by_preset(results)

    assert [row["preset"] for row in rows] == ["no_safety", "full"]
    assert rows[0]["trials"] == 2


# ------------------------------------------------------------- tables


def test_markdown_table_renders_floats_costs_and_missing_values():
    rows = [
        {"preset": "full", "solve_rate": 0.8, "mean_cost_usd": 0.1, "cost_per_solved_task": None}
    ]

    assert render_markdown_table(rows) == (
        "| preset | solve_rate | mean_cost_usd | cost_per_solved_task |\n"
        "|---|---|---|---|\n"
        "| full | 0.80 | $0.1000 | - |"
    )


def test_markdown_table_says_so_when_there_are_no_rows():
    assert render_markdown_table([]) == "(no results)"


def test_per_scenario_table_counts_solved_over_trials_per_preset():
    results = [
        {"scenario_id": "a", "preset_name": "full", "solved": True},
        {"scenario_id": "a", "preset_name": "full", "solved": False},
        {"scenario_id": "a", "preset_name": "no_memory", "solved": True},
        {"scenario_id": "b", "preset_name": "full", "solved": True},
    ]

    assert per_scenario_table(results) == (
        "| scenario | full | no_memory |\n|---|---|---|\n| a | 1/2 | 1/1 |\n| b | 1/1 | - |"
    )
