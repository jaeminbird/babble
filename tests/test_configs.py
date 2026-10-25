import pytest

from faultline.bench.configs import PRESETS, preset
from faultline.types import HarnessConfig


def test_every_preset_is_named_after_the_one_feature_it_turns_off():
    assert list(PRESETS) == [
        "full",
        "no_subagents",
        "no_truncation",
        "no_caching",
        "no_memory",
        "no_compaction",
        "no_safety",
    ]


def test_full_is_the_default_harness_config():
    assert PRESETS["full"] == HarnessConfig()


@pytest.mark.parametrize(
    ("name", "changed"),
    [
        ("no_subagents", {"subagents": False}),
        ("no_truncation", {"truncate_tool_output": None}),
        ("no_caching", {"prompt_caching": False}),
        ("no_memory", {"memory_notes": False}),
        ("no_compaction", {"compaction_threshold": None}),
        (
            "no_safety",
            {"safety": False, "verify_after_change": False, "auto_rollback": False},
        ),
    ],
)
def test_each_ablation_changes_exactly_its_own_fields(name, changed):
    expected = dict(vars(HarnessConfig()), **changed)
    assert vars(PRESETS[name]) == expected


def test_preset_returns_a_copy_so_callers_cannot_change_the_shared_preset():
    config = preset("full")
    config.max_turns = 1
    assert PRESETS["full"].max_turns == HarnessConfig().max_turns
    assert preset("full") is not PRESETS["full"]


def test_preset_overrides_the_model_only_when_given():
    assert preset("full").model == HarnessConfig().model
    assert preset("no_memory", model="claude-haiku-5-5").model == "claude-haiku-5-5"
    assert preset("no_memory", model="claude-haiku-5-5").memory_notes is False


def test_unknown_preset_lists_the_valid_names():
    with pytest.raises(ValueError) as excinfo:
        preset("nope")
    message = str(excinfo.value)
    assert "'nope'" in message
    for name in PRESETS:
        assert name in message
