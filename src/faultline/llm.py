"""LLM client: the Anthropic SDK adapter, a scripted fake for tests, and pricing."""

from __future__ import annotations

import copy
import itertools
from typing import Any

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
