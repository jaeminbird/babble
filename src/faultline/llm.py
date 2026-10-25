"""LLM client: the Anthropic SDK adapter, a scripted fake for tests, and pricing."""

from __future__ import annotations

from faultline.types import Usage

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
