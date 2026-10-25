import pytest

from faultline.llm import PRICING, cost
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
