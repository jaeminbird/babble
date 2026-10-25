"""Ablation presets: each one is the full harness with one feature turned off."""

from __future__ import annotations

import dataclasses

from babbler.types import HarnessConfig

PRESETS: dict[str, HarnessConfig] = {
    "full": HarnessConfig(),
    "no_subagents": HarnessConfig(subagents=False),
    "no_truncation": HarnessConfig(truncate_tool_output=None),
    "no_caching": HarnessConfig(prompt_caching=False),
    "no_memory": HarnessConfig(memory_notes=False),
    "no_compaction": HarnessConfig(compaction_threshold=None),
    # Without safety, nothing is gated, verified or rolled back: the unsafe baseline.
    "no_safety": HarnessConfig(safety=False, verify_after_change=False, auto_rollback=False),
}


def preset(name: str, model: str | None = None) -> HarnessConfig:
    """A fresh copy of the named preset, with its model replaced when `model` is given."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; valid presets: {', '.join(PRESETS)}")
    overrides = {} if model is None else {"model": model}
    return dataclasses.replace(PRESETS[name], **overrides)
