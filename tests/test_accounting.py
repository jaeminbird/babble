import threading

import pytest

from babble.harness.accounting import Accounting
from babble.llm import cost
from babble.types import Usage


def test_starts_empty():
    accounting = Accounting()
    assert accounting.usage == Usage()
    assert accounting.cost_usd == 0.0
    assert accounting.unpriced == []


def test_add_sums_usage():
    accounting = Accounting()
    accounting.add(Usage(100, 10, 5, 50), "claude-haiku-5-5")
    accounting.add(Usage(200, 20, 0, 0), "claude-haiku-5-5")
    assert accounting.usage == Usage(300, 30, 5, 50)


def test_cost_matches_llm_pricing():
    accounting = Accounting()
    opus_usage = Usage(1_000_000, 100_000, 0, 0)
    sonnet_usage = Usage(0, 0, 0, 2_000_000)
    accounting.add(opus_usage, "claude-opus-5-5")
    accounting.add(sonnet_usage, "claude-sonnet-5-5")
    expected = cost(opus_usage, "claude-opus-5-5") + cost(sonnet_usage, "claude-sonnet-5-5")
    assert accounting.cost_usd == pytest.approx(expected)
    assert accounting.cost_usd > 0


def test_unknown_model_counts_tokens_but_adds_no_cost():
    accounting = Accounting()
    accounting.add(Usage(1000, 1000), "some-future-model")
    accounting.add(Usage(1000, 1000), "some-future-model")
    assert accounting.usage == Usage(2000, 2000)
    assert accounting.cost_usd == 0.0
    assert accounting.unpriced == ["some-future-model"]


def test_priced_and_unpriced_models_mix():
    accounting = Accounting()
    accounting.add(Usage(1_000_000, 0), "claude-haiku-5-5")
    accounting.add(Usage(1_000_000, 0), "mystery")
    assert accounting.cost_usd == pytest.approx(0.10)
    assert accounting.unpriced == ["mystery"]


def test_concurrent_adds_lose_no_usage():
    accounting = Accounting()
    thread_count, adds_per_thread = 8, 500
    barrier = threading.Barrier(thread_count)

    def add_many(model: str) -> None:
        barrier.wait()
        for _ in range(adds_per_thread):
            accounting.add(Usage(3, 2, 1, 4), model)

    threads = [
        threading.Thread(target=add_many, args=("claude-haiku-5-5" if i % 2 else "mystery",))
        for i in range(thread_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    total = thread_count * adds_per_thread
    assert accounting.usage == Usage(3 * total, 2 * total, total, 4 * total)
    assert accounting.cost_usd == pytest.approx(
        cost(Usage(3, 2, 1, 4), "claude-haiku-5-5") * total / 2
    )
    assert accounting.unpriced == ["mystery"]
