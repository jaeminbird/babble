"""LLM client: the Anthropic SDK adapter, a scripted fake for tests, and pricing."""

from __future__ import annotations

import copy
import itertools
from typing import Any

import anthropic

from faultline.types import LLMResponse, Usage

# $ per 1M tokens: (input, output)
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

CACHE_WRITE_MULTIPLIER = 1.25
# Cache reads are 0.05x input on opus-5-5 and 0.1x on the other models.
CACHE_READ_MULTIPLIER = {"claude-opus-5-5": 0.05}
DEFAULT_CACHE_READ_MULTIPLIER = 0.1


def cost(usage: Usage, model: str) -> float:
    """Dollar cost of one call. Raises KeyError for a model not in PRICING."""
    input_price, output_price = PRICING[model]
    read_multiplier = CACHE_READ_MULTIPLIER.get(model, DEFAULT_CACHE_READ_MULTIPLIER)
    dollars = (
        usage.input_tokens * input_price
        + usage.cache_creation_input_tokens * input_price * CACHE_WRITE_MULTIPLIER
        + usage.cache_read_input_tokens * input_price * read_multiplier
        + usage.output_tokens * output_price
    )
    return dollars / 1_000_000


# --------------------------------------------------------------------- fake


class FakeLLMClient:
    """Returns scripted responses in order and records every call's kwargs."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        model: str,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
        effort: str | None = None,
        cache: bool = True,
    ) -> LLMResponse:
        self.calls.append(
            copy.deepcopy(
                {
                    "model": model,
                    "system": system,
                    "messages": messages,
                    "tools": tools,
                    "max_tokens": max_tokens,
                    "effort": effort,
                    "cache": cache,
                }
            )
        )
        if not self._responses:
            raise AssertionError("FakeLLMClient ran out of responses")
        return self._responses.pop(0)


_tool_ids = itertools.count(1)


def tool_use(name: str, input: dict[str, Any], id: str | None = None) -> dict[str, Any]:
    """A tool_use content block for a scripted response."""
    return {
        "type": "tool_use",
        "id": id or f"toolu_fake{next(_tool_ids):04d}",
        "name": name,
        "input": input,
    }


def text(s: str) -> dict[str, Any]:
    """A text content block for a scripted response."""
    return {"type": "text", "text": s}


def response(
    *blocks: dict[str, Any],
    stop_reason: str | None = None,
    usage: Usage | None = None,
    model: str = "claude-opus-5-5",
) -> LLMResponse:
    """Build an LLMResponse. stop_reason defaults to tool_use if any block is a tool_use."""
    if stop_reason is None:
        has_tool = any(b.get("type") == "tool_use" for b in blocks)
        stop_reason = "tool_use" if has_tool else "end_turn"
    return LLMResponse(
        content=list(blocks),
        stop_reason=stop_reason,
        usage=usage or Usage(),
        model=model,
    )


# ----------------------------------------------------------------- Anthropic

FALLBACK_MODELS = frozenset({"claude-opus-5-5", "claude-sonnet-5-5"})
FALLBACK_BETA = "server-side-fallback-2026-07-01"
EPHEMERAL = {"type": "ephemeral"}


class AnthropicClient:
    """LLMClient backed by the official anthropic SDK. Streams each request."""

    def __init__(self, client: anthropic.Anthropic | None = None, fallbacks: bool = True):
        self._client = client if client is not None else anthropic.Anthropic()
        self.fallbacks = fallbacks

    def complete(
        self,
        *,
        model: str,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
        effort: str | None = None,
        cache: bool = True,
    ) -> LLMResponse:
        kwargs = _request_kwargs(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            effort=effort,
            cache=cache,
        )
        if self.fallbacks and model in FALLBACK_MODELS:
            endpoint = self._client.beta.messages
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        else:
            endpoint = self._client.messages
        with endpoint.stream(**kwargs) as stream:
            final = stream.get_final_message()
        return _to_response(final)


def _request_kwargs(
    *,
    model: str,
    system: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_tokens: int,
    effort: str | None,
    cache: bool,
) -> dict[str, Any]:
    # Copy the system blocks so the caller's list is never mutated.
    system_blocks = copy.deepcopy(system)
    if cache and system_blocks:
        system_blocks[-1]["cache_control"] = dict(EPHEMERAL)
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_blocks,
        "messages": messages,
        "thinking": {"type": "adaptive"},
    }
    if tools:
        kwargs["tools"] = tools
    if effort:
        kwargs["output_config"] = {"effort": effort}
    if cache:
        # Top-level cache_control caches the conversation tail as well as the system prompt.
        kwargs["cache_control"] = dict(EPHEMERAL)
    return kwargs


def _to_response(final: Any) -> LLMResponse:
    usage = final.usage
    return LLMResponse(
        content=[block.model_dump(mode="json", exclude_none=True) for block in final.content],
        stop_reason=final.stop_reason,
        usage=Usage(
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
            cache_read_input_tokens=usage.cache_read_input_tokens or 0,
        ),
        model=final.model,
    )


# ------------------------------------------------------------------ invariants


def assert_append_only(calls: list[dict[str, Any]]) -> None:
    """Check that history is append-only across recorded calls.

    Calls are grouped into conversations by their first message. For each call, the most
    recent earlier call in the same conversation must have messages that are an exact prefix
    of this call's messages. Raises AssertionError naming the first differing index.
    """
    for i, call in enumerate(calls):
        messages = call["messages"]
        if not messages:
            continue
        for j in range(i - 1, -1, -1):
            earlier = calls[j]["messages"]
            if earlier and earlier[0] == messages[0]:
                for k, old in enumerate(earlier):
                    if k >= len(messages) or messages[k] != old:
                        raise AssertionError(
                            f"call {i} is not an append-only extension of call {j}: "
                            f"first differing message index is {k}"
                        )
                break
