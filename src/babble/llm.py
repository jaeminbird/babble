"""LLM client: the Anthropic SDK adapter, a scripted fake for tests, and pricing."""

from __future__ import annotations

import copy
import itertools
from typing import Any

import anthropic

from babble.types import LLMResponse, Usage

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
        server_context_management: str | None = None,
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
                    "server_context_management": server_context_management,
                }
            )
        )
        if not self._responses:
            raise AssertionError("FakeLLMClient ran out of responses")
        return self._responses.pop(0)


_fake_tool_use_numbers = itertools.count(1)


def tool_use(
    name: str, tool_input: dict[str, Any], tool_use_id: str | None = None
) -> dict[str, Any]:
    """A tool_use content block for a scripted response."""
    return {
        "type": "tool_use",
        "id": tool_use_id or f"toolu_fake{next(_fake_tool_use_numbers):04d}",
        "name": name,
        "input": tool_input,
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

    def __init__(self, client: anthropic.Anthropic | None = None):
        self._client = client if client is not None else anthropic.Anthropic()

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
        server_context_management: str | None = None,
    ) -> LLMResponse:
        request_kwargs = _request_kwargs(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            effort=effort,
            cache=cache,
        )
        uses_server_fallbacks = model in FALLBACK_MODELS
        if not uses_server_fallbacks and server_context_management is None:
            with self._client.messages.stream(**request_kwargs) as stream:
                return _to_response(stream.get_final_message())

        betas: list[str] = []
        if uses_server_fallbacks:
            betas.append(FALLBACK_BETA)
            request_kwargs["fallbacks"] = "default"
        if server_context_management is not None:
            beta_name, context_management = _server_context_management_request(
                server_context_management
            )
            betas.append(beta_name)
            request_kwargs["context_management"] = context_management
        request_kwargs["betas"] = betas
        with self._client.beta.messages.stream(**request_kwargs) as beta_stream:
            return _to_response(beta_stream.get_final_message())


def _server_context_management_request(strategy: str) -> tuple[str, dict[str, Any]]:
    """The beta header and the context_management body for one server-side strategy."""
    if strategy == "compaction":
        return "compact-2026-01-12", {"edits": [{"type": "compact_20260112"}]}
    if strategy == "clear_tool_uses":
        return "context-management-2025-06-27", {"edits": [{"type": "clear_tool_uses_20250919"}]}
    raise ValueError(f"unknown server_context_management {strategy!r}")


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


def _to_response(final_message: Any) -> LLMResponse:
    usage = final_message.usage
    return LLMResponse(
        content=[
            block.model_dump(mode="json", exclude_none=True) for block in final_message.content
        ],
        stop_reason=final_message.stop_reason,
        usage=Usage(
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
            cache_read_input_tokens=usage.cache_read_input_tokens or 0,
        ),
        model=final_message.model,
    )


# ------------------------------------------------------------------ invariants


def assert_append_only(calls: list[dict[str, Any]]) -> None:
    """Check that history is append-only across recorded calls.

    Calls are grouped into conversations by their first message. For each call, the most
    recent earlier call in the same conversation must have messages that are an exact prefix
    of this call's messages. Raises AssertionError naming the first differing index.
    """
    for call_index, call in enumerate(calls):
        messages = call["messages"]
        if not messages:
            continue
        for earlier_index in range(call_index - 1, -1, -1):
            earlier_messages = calls[earlier_index]["messages"]
            if earlier_messages and earlier_messages[0] == messages[0]:
                for message_index, earlier_message in enumerate(earlier_messages):
                    if message_index >= len(messages) or messages[message_index] != earlier_message:
                        raise AssertionError(
                            f"call {call_index} is not an append-only extension of call "
                            f"{earlier_index}: first differing message index is {message_index}"
                        )
                break
