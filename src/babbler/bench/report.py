"""Summaries of bench results: per-preset rates, pass^k and pass@k, and markdown tables.

Input is the JSONL file `run_matrix` writes: one TrialResult dict per line.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from math import comb
from pathlib import Path
from typing import Any


def load_results(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as results_file:
        return [json.loads(line) for line in results_file if line.strip()]


def pass_hat_k(successes: int, trials: int, k: int) -> float:
    """tau-bench pass^k: the chance that all k trials drawn from `trials` succeed, C(c,k)/C(n,k)."""
    _check_pass_arguments(successes, trials, k)
    return comb(successes, k) / comb(trials, k)


def pass_at_k(successes: int, trials: int, k: int) -> float:
    """pass@k: the chance that at least one of k trials drawn from `trials` succeeds."""
    _check_pass_arguments(successes, trials, k)
    return 1 - comb(trials - successes, k) / comb(trials, k)


def _check_pass_arguments(successes: int, trials: int, k: int) -> None:
    if not 0 <= successes <= trials:
        raise ValueError(f"successes must be between 0 and trials ({trials}), got {successes}")
    if not 1 <= k <= trials:
        raise ValueError(f"k must be between 1 and trials ({trials}), got {k}")


def summarize_by_preset(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One summary row per preset, in first-appearance order."""
    by_preset = _group(results, "preset_name")
    return [_summarize(preset_name, rows) for preset_name, rows in by_preset.items()]


def _summarize(preset_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    trials = len(rows)
    solved = _count(rows, "solved")
    total_cost = sum(row["cost_usd"] for row in rows)

    # pass^k per scenario, with k the number of trials that scenario got in this preset, then the
    # mean over scenarios.
    scenarios = _group(rows, "scenario_id").values()
    pass_hat_1 = _mean(pass_hat_k(_count(s, "solved"), len(s), 1) for s in scenarios)
    pass_hat_k_max = _mean(pass_hat_k(_count(s, "solved"), len(s), len(s)) for s in scenarios)

    input_tokens = sum(row["input_tokens"] for row in rows)
    cache_read = sum(row["cache_read_input_tokens"] for row in rows)
    cache_creation = sum(row["cache_creation_input_tokens"] for row in rows)
    cache_denominator = input_tokens + cache_read + cache_creation

    unfixable = [row for row in rows if not row["fixable"]]

    return {
        "preset": preset_name,
        "trials": trials,
        "solve_rate": solved / trials,
        "root_cause_accuracy": _count(rows, "root_cause_correct") / trials,
        "pass_hat_1": pass_hat_1,
        "pass_hat_k_max": pass_hat_k_max,
        "mean_cost_usd": total_cost / trials,
        "cost_per_solved_task": total_cost / solved if solved else None,
        "mean_turns": sum(row["turns"] for row in rows) / trials,
        "cache_hit_rate": cache_read / cache_denominator if cache_denominator else None,
        "unnecessary_change_rate": (
            _count(unfixable, "unnecessary_change") / len(unfixable) if unfixable else None
        ),
        "collateral_damage_count": sum(len(row["collateral_damage"]) for row in rows),
        "unapproved_changes": sum(row["unapproved_changes"] for row in rows),
        "rollbacks": sum(row["changes_rolled_back"] for row in rows),
        "errors": sum(1 for row in rows if row["error"] is not None),
    }


def render_markdown_table(rows: Sequence[dict[str, Any]]) -> str:
    """A markdown table with one column per key of the first row. None renders as a dash."""
    if not rows:
        return "(no results)"
    columns = list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(column, row[column]) for column in columns) + " |")
    return "\n".join(lines)


def per_scenario_table(results: Sequence[dict[str, Any]]) -> str:
    """Rows are scenarios, columns are presets, and each cell is "solved/trials"."""
    presets = list(_group(results, "preset_name"))
    rows: list[dict[str, Any]] = []
    for scenario_id, scenario_rows in _group(results, "scenario_id").items():
        row: dict[str, Any] = {"scenario": scenario_id}
        for preset_name in presets:
            in_preset = [r for r in scenario_rows if r["preset_name"] == preset_name]
            row[preset_name] = (
                f"{_count(in_preset, 'solved')}/{len(in_preset)}" if in_preset else "-"
            )
        rows.append(row)
    return render_markdown_table(rows)


def _cell(column: str, value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if "cost" in column:
            return f"${value:.4f}"
        return f"{value:.2f}"
    return str(value)


def _group(rows: Sequence[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)
    return groups


def _count(rows: Sequence[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row[key])


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected)
