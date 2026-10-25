"""Ablation presets: each one is the full harness with one feature turned off."""

from __future__ import annotations

import dataclasses

from babble.types import HarnessConfig

PRESETS: dict[str, HarnessConfig] = {
    "full": HarnessConfig(),
    "no_subagents": HarnessConfig(subagents=False),
    "no_truncation": HarnessConfig(truncate_tool_output=None),
    "no_caching": HarnessConfig(prompt_caching=False),
    "no_memory": HarnessConfig(memory_notes=False),
    "no_compaction": HarnessConfig(compaction_threshold=None),
    "server_compaction": HarnessConfig(
        server_context_management="compaction", compaction_threshold=None
    ),
    "server_clear_tool_uses": HarnessConfig(
        server_context_management="clear_tool_uses", compaction_threshold=None
    ),
    # Without safety, nothing is gated, verified or rolled back: the unsafe baseline.
    "no_safety": HarnessConfig(safety=False, verify_after_change=False, auto_rollback=False),
}


def harness_config_for_preset(name: str, model: str | None = None) -> HarnessConfig:
    """A fresh copy of the named preset, with its model replaced when `model` is given."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; valid presets: {', '.join(PRESETS)}")
    preset = PRESETS[name]
    if model is None:
        return dataclasses.replace(preset)
    return dataclasses.replace(preset, model=model)
