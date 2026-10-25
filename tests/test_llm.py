import copy
from types import SimpleNamespace

import pytest
from anthropic.types import TextBlock, ToolUseBlock

from babble.llm import (
    PRICING,
    AnthropicClient,
    FakeLLMClient,
    assert_append_only,
    cost,
    response,
    text,
    tool_use,
)
from babble.types import Usage


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
    assert tool_use("x", {}, tool_use_id="toolu_fixed")["id"] == "toolu_fixed"
    assert tool_use("x", {})["id"] != tool_use("x", {})["id"]


def test_text_helper_builds_block():
    assert text("hello") == {"type": "text", "text": "hello"}


def test_response_helper_defaults():
    tool_resp = response(text("let me check"), tool_use("run_show", {}))
    assert tool_resp.stop_reason == "tool_use"
    assert tool_resp.model == "claude-opus-5-5"
    assert tool_resp.usage == Usage()
    assert [tool_call.name for tool_call in tool_resp.tool_calls] == ["run_show"]

    done = response(text("all good"))
    assert done.stop_reason == "end_turn"
    assert done.text == "all good"


def test_response_helper_overrides():
    usage = Usage(input_tokens=5, output_tokens=7)
    llm_response = response(
        text("x"), stop_reason="max_tokens", usage=usage, model="claude-haiku-5-5"
    )
    assert llm_response.stop_reason == "max_tokens"
    assert llm_response.usage is usage
    assert llm_response.model == "claude-haiku-5-5"


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


# ------------------------------------------------------------------ AnthropicClient

FALLBACK_BETA = "server-side-fallback-2026-07-01"
SYSTEM = [{"type": "text", "text": "You are a network agent."}, {"type": "text", "text": "Rules."}]
TOOL = {"name": "show", "description": "Run a show command", "input_schema": {"type": "object"}}


class _FakeStream:
    def __init__(self, final):
        self._final = final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._final


class _Endpoint:
    """Stands in for client.messages or client.beta.messages. Records stream() kwargs."""

    def __init__(self, final):
        self._final = final
        self.kwargs = None

    def stream(self, **kwargs):
        self.kwargs = kwargs
        return _FakeStream(self._final)


class _StubAnthropic:
    def __init__(self, final):
        self.messages = _Endpoint(final)
        self.beta = SimpleNamespace(messages=_Endpoint(final))


def _final(
    *blocks,
    stop_reason="end_turn",
    model="claude-haiku-5-5",
    usage=None,
):
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        model=model,
        usage=usage
        or SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
        ),
    )


def _complete(stub, model="claude-haiku-5-5", fallbacks=True, **overrides):
    client = AnthropicClient(stub, fallbacks=fallbacks)
    kwargs = {
        "model": model,
        "system": SYSTEM,
        "messages": [USER],
        "tools": [TOOL],
        "max_tokens": 1000,
    }
    kwargs.update(overrides)
    return client.complete(**kwargs)


def test_request_uses_plain_stream_with_caching_thinking_and_effort():
    stub = _StubAnthropic(_final(TextBlock(type="text", text="ok", citations=None)))
    original_system = copy.deepcopy(SYSTEM)

    _complete(stub, effort="high")

    assert stub.beta.messages.kwargs is None
    assert stub.messages.kwargs == {
        "model": "claude-haiku-5-5",
        "max_tokens": 1000,
        "system": [
            {"type": "text", "text": "You are a network agent."},
            {"type": "text", "text": "Rules.", "cache_control": {"type": "ephemeral"}},
        ],
        "messages": [USER],
        "tools": [TOOL],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
        "cache_control": {"type": "ephemeral"},
    }
    assert SYSTEM == original_system


def test_cache_false_and_no_effort_omit_cache_and_output_config():
    stub = _StubAnthropic(_final(TextBlock(type="text", text="ok", citations=None)))

    _complete(stub, cache=False, tools=[])

    kwargs = stub.messages.kwargs
    assert "cache_control" not in kwargs
    assert "output_config" not in kwargs
    assert "tools" not in kwargs
    assert kwargs["system"] == SYSTEM
    assert kwargs["thinking"] == {"type": "adaptive"}


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5"])
def test_opus_and_sonnet_use_beta_stream_with_server_fallbacks(model):
    stub = _StubAnthropic(_final(TextBlock(type="text", text="ok", citations=None)))

    _complete(stub, model=model)

    assert stub.messages.kwargs is None
    kwargs = stub.beta.messages.kwargs
    assert kwargs["betas"] == [FALLBACK_BETA]
    assert kwargs["fallbacks"] == "default"
    assert kwargs["model"] == model


@pytest.mark.parametrize(
    ("model", "fallbacks"),
    [("claude-haiku-5-5", True), ("claude-opus-5-5", False)],
)
def test_plain_stream_when_model_has_no_fallback_or_fallbacks_disabled(model, fallbacks):
    stub = _StubAnthropic(_final(TextBlock(type="text", text="ok", citations=None)))

    _complete(stub, model=model, fallbacks=fallbacks)

    assert stub.beta.messages.kwargs is None
    assert "fallbacks" not in stub.messages.kwargs
    assert "betas" not in stub.messages.kwargs


def test_response_blocks_are_plain_dicts_and_metadata_is_mapped():
    final = _final(
        TextBlock(type="text", text="checking", citations=None),
        ToolUseBlock(type="tool_use", id="toolu_1", name="show", input={"device": "r1"}),
        stop_reason="tool_use",
        model="claude-sonnet-5-5",
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            cache_creation_input_tokens=300,
            cache_read_input_tokens=None,
        ),
    )
    stub = _StubAnthropic(final)

    llm_response = _complete(stub, model="claude-sonnet-5-5")

    assert llm_response.content == [
        {"type": "text", "text": "checking"},
        {"type": "tool_use", "id": "toolu_1", "name": "show", "input": {"device": "r1"}},
    ]
    assert llm_response.stop_reason == "tool_use"
    assert llm_response.model == "claude-sonnet-5-5"
    assert llm_response.text == "checking"
    assert [tool_call.id for tool_call in llm_response.tool_calls] == ["toolu_1"]
    assert llm_response.usage == Usage(
        input_tokens=100,
        output_tokens=20,
        cache_creation_input_tokens=300,
        cache_read_input_tokens=0,
    )
