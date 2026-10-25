import threading

import pytest

from babbler.harness.accounting import Accounting
from babbler.llm import cost
from babbler.types import Usage


def test_starts_empty():
    acc = Accounting()
    assert acc.usage == Usage()
    assert acc.cost_usd == 0.0
    assert acc.by_model == {}
    assert acc.calls == 0
    assert acc.unpriced == []


def test_add_sums_usage_and_counts_calls():
    acc = Accounting()
    acc.add(Usage(100, 10, 5, 50), "claude-haiku-5-5")
    acc.add(Usage(200, 20, 0, 0), "claude-haiku-5-5")
    assert acc.usage == Usage(300, 30, 5, 50)
    assert acc.calls == 2


def test_cost_matches_llm_pricing():
    acc = Accounting()
    u1 = Usage(1_000_000, 100_000, 0, 0)
    u2 = Usage(0, 0, 0, 2_000_000)
    acc.add(u1, "claude-opus-5-5")
    acc.add(u2, "claude-sonnet-5-5")
    expected = cost(u1, "claude-opus-5-5") + cost(u2, "claude-sonnet-5-5")
    assert acc.cost_usd == pytest.approx(expected)
    assert acc.cost_usd > 0


def test_by_model_splits_usage_per_model():
    acc = Accounting()
    acc.add(Usage(10, 1), "claude-opus-5-5")
    acc.add(Usage(20, 2), "claude-haiku-5-5")
    acc.add(Usage(5, 0), "claude-opus-5-5")
    assert acc.by_model["claude-opus-5-5"] == Usage(15, 1)
    assert acc.by_model["claude-haiku-5-5"] == Usage(20, 2)


def test_unknown_model_counts_tokens_but_adds_no_cost():
    acc = Accounting()
    acc.add(Usage(1000, 1000), "some-future-model")
    acc.add(Usage(1000, 1000), "some-future-model")
    assert acc.usage == Usage(2000, 2000)
    assert acc.calls == 2
    assert acc.cost_usd == 0.0
    assert acc.unpriced == ["some-future-model"]


def test_priced_and_unpriced_models_mix():
    acc = Accounting()
    acc.add(Usage(1_000_000, 0), "claude-haiku-5-5")
    acc.add(Usage(1_000_000, 0), "mystery")
    assert acc.cost_usd == pytest.approx(0.10)
    assert acc.unpriced == ["mystery"]


def test_concurrent_adds_lose_no_usage_or_calls():
    acc = Accounting()
    thread_count, adds_per_thread = 8, 500
    barrier = threading.Barrier(thread_count)

    def add_many(model: str) -> None:
        barrier.wait()
        for _ in range(adds_per_thread):
            acc.add(Usage(3, 2, 1, 4), model)

    threads = [
        threading.Thread(target=add_many, args=("claude-haiku-5-5" if i % 2 else "mystery",))
        for i in range(thread_count)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = thread_count * adds_per_thread
    assert acc.calls == total
    assert acc.usage == Usage(3 * total, 2 * total, total, 4 * total)
    assert acc.by_model["claude-haiku-5-5"].input_tokens == 3 * total // 2
    assert acc.by_model["mystery"].input_tokens == 3 * total // 2
    assert acc.cost_usd == pytest.approx(cost(Usage(3, 2, 1, 4), "claude-haiku-5-5") * total / 2)
    assert acc.unpriced == ["mystery"]
