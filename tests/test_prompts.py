import re

from babble.harness.prompts import SUBAGENT_PROMPT, SYSTEM_PROMPT


def test_prompt_is_short_and_plain():
    assert len(SYSTEM_PROMPT.split()) < 350


def test_prompt_names_the_tools_and_the_submit_rule():
    for name in ("list_devices", "show", "ping", "propose_change", "submit_diagnosis"):
        assert name in SYSTEM_PROMPT
    assert "exactly once" in SYSTEM_PROMPT


def test_prompt_has_no_dates_or_numbers_that_could_drift():
    assert not re.search(r"\b(19|20)\d{2}\b", SYSTEM_PROMPT)
    assert SYSTEM_PROMPT == SYSTEM_PROMPT.strip() + "\n"


def test_subagent_prompt_is_short_and_names_its_finish_tool():
    assert len(SUBAGENT_PROMPT.split()) < 200
    assert "submit_findings" in SUBAGENT_PROMPT
    assert "exactly once" in SUBAGENT_PROMPT
    assert "cannot change configuration" in SUBAGENT_PROMPT
