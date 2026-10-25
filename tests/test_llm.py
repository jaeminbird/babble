import pytest

from faultline.llm import (
    PRICING,
    FakeLLMClient,
    assert_append_only,
    cost,
    response,
    text,
    tool_use,
)
from faultline.types import Usage


def test_pricing_table_covers_the_three_models():
    assert PRICING == {
        "claude-opus-5-5": (4.00, 20.00),
        "claude-sonnet-5-5": (2.00, 10.00),
        "claude-haiku-5-5": (0.10, 0.50),
    }


def test_cost_plain_input_and_output():
    usage = Usage(input_tokens=1_000_000, output_tokens=500_000)
    assert cost(usage, "claude-opus-5-5") == pytest.approx(4.00 + 10.00)
    assert cost(usage, "claude-sonnet-5-5") == pytest.approx(2.00 + 5.00)
    assert cost(usage, "claude-haiku-5-5") == pytest.approx(0.10 + 0.25)


def test_cost_zero_usage_is_zero():
    assert cost(Usage(), "claude-sonnet-5-5") == 0.0


def test_cache_write_costs_1_25x_input():
    usage = Usage(cache_creation_input_tokens=1_000_000)
    assert cost(usage, "claude-opus-5-5") == pytest.approx(4.00 * 1.25)
    assert cost(usage, "claude-haiku-5-5") == pytest.approx(0.10 * 1.25)


def test_cache_read_is_0_05x_input_on_opus_and_0_1x_elsewhere():
    usage = Usage(cache_read_input_tokens=1_000_000)
    assert cost(usage, "claude-opus-5-5") == pytest.approx(4.00 * 0.05)
    assert cost(usage, "claude-sonnet-5-5") == pytest.approx(2.00 * 0.1)
    assert cost(usage, "claude-haiku-5-5") == pytest.approx(0.10 * 0.1)


def test_cost_combines_all_token_kinds():
    usage = Usage(
        input_tokens=200_000,
        output_tokens=100_000,
        cache_creation_input_tokens=400_000,
        cache_read_input_tokens=1_000_000,
    )
    expected = (
        200_000 * 2.00 + 400_000 * 2.00 * 1.25 + 1_000_000 * 2.00 * 0.1 + 100_000 * 10.00
    ) / 1_000_000
    assert cost(usage, "claude-sonnet-5-5") == pytest.approx(expected)


def test_cost_unknown_model_raises_key_error():
    with pytest.raises(KeyError):
        cost(Usage(input_tokens=1), "gpt-unknown")


# ------------------------------------------------------------------ FakeLLMClient


def _call(client: FakeLLMClient, messages=None, **overrides):
    kwargs = {
        "model": "claude-opus-5-5",
        "system": [{"type": "text", "text": "sys"}],
        "messages": messages if messages is not None else [{"role": "user", "content": "hi"}],
        "tools": [],
        "max_tokens": 100,
    }
    kwargs.update(overrides)
    return client.complete(**kwargs)


def test_fake_returns_responses_in_order():
    first = response(text("one"))
    second = response(text("two"))
    client = FakeLLMClient([first, second])
    assert _call(client) is first
    assert _call(client) is second


def test_fake_raises_when_exhausted():
    client = FakeLLMClient([response(text("only"))])
    _call(client)
    with pytest.raises(AssertionError, match="FakeLLMClient ran out of responses"):
        _call(client)


def test_fake_records_kwargs_deep_copied():
    client = FakeLLMClient([response(text("ok"))])
    messages = [{"role": "user", "content": "hi"}]
    _call(client, messages=messages, effort="low", cache=False)
    messages[0]["content"] = "mutated after the call"
    assert len(client.calls) == 1
    recorded = client.calls[0]
    assert recorded["messages"] == [{"role": "user", "content": "hi"}]
    assert recorded["effort"] == "low"
    assert recorded["cache"] is False
    assert recorded["model"] == "claude-opus-5-5"
    assert recorded["max_tokens"] == 100


def test_tool_use_helper_builds_block_and_generates_ids():
    block = tool_use("run_show", {"device": "r1"})
    assert block["type"] == "tool_use"
    assert block["name"] == "run_show"
    assert block["input"] == {"device": "r1"}
    assert block["id"]
    assert tool_use("x", {}, id="toolu_fixed")["id"] == "toolu_fixed"
    assert tool_use("x", {})["id"] != tool_use("x", {})["id"]


def test_text_helper_builds_block():
    assert text("hello") == {"type": "text", "text": "hello"}


def test_response_helper_defaults():
    tool_resp = response(text("let me check"), tool_use("run_show", {}))
    assert tool_resp.stop_reason == "tool_use"
    assert tool_resp.model == "claude-opus-5-5"
    assert tool_resp.usage == Usage()
    assert [c.name for c in tool_resp.tool_calls] == ["run_show"]

    done = response(text("all good"))
    assert done.stop_reason == "end_turn"
    assert done.text == "all good"


def test_response_helper_overrides():
    usage = Usage(input_tokens=5, output_tokens=7)
    resp = response(
        text("x"), stop_reason="max_tokens", usage=usage, model="claude-haiku-5-5"
    )
    assert resp.stop_reason == "max_tokens"
    assert resp.usage is usage
    assert resp.model == "claude-haiku-5-5"


# ------------------------------------------------------------------ assert_append_only

USER = {"role": "user", "content": "diagnose r1"}
ASSISTANT = {"role": "assistant", "content": [{"type": "text", "text": "checking"}]}
RESULT = {
    "role": "user",
    "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
}


def test_append_only_passes_for_growing_conversation():
    calls = [
        {"messages": [USER]},
        {"messages": [USER, ASSISTANT, RESULT]},
        {"messages": [USER, ASSISTANT, RESULT, ASSISTANT]},
    ]
    assert_append_only(calls)


def test_append_only_passes_with_no_calls():
    assert_append_only([])


def test_append_only_ignores_interleaved_conversations():
    other = {"role": "user", "content": "a subagent question"}
    calls = [
        {"messages": [USER]},
        {"messages": [other]},
        {"messages": [USER, ASSISTANT]},
        {"messages": [other, ASSISTANT]},
    ]
    assert_append_only(calls)


def test_append_only_fails_when_an_earlier_message_is_edited():
    edited = {"role": "assistant", "content": [{"type": "text", "text": "rewritten"}]}
    calls = [
        {"messages": [USER, ASSISTANT]},
        {"messages": [USER, edited, RESULT]},
    ]
    with pytest.raises(AssertionError, match="first differing message index is 1"):
        assert_append_only(calls)


def test_append_only_fails_when_history_is_truncated():
    calls = [
        {"messages": [USER, ASSISTANT, RESULT]},
        {"messages": [USER, ASSISTANT]},
    ]
    with pytest.raises(AssertionError, match="first differing message index is 2"):
        assert_append_only(calls)


def test_append_only_accepts_fake_client_recordings():
    client = FakeLLMClient([response(text("a")), response(text("b"))])
    _call(client, messages=[USER])
    _call(client, messages=[USER, ASSISTANT, RESULT])
    assert_append_only(client.calls)
