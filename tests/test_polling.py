from babble.polling import poll_until


def test_returns_true_on_the_first_try_without_sleeping():
    sleeps: list[float] = []
    assert poll_until(lambda: True, 10, 1, sleep=sleeps.append) is True
    assert sleeps == []


def test_retries_with_the_poll_interval_until_the_condition_holds():
    answers = iter([False, False, True])
    sleeps: list[float] = []
    assert poll_until(lambda: next(answers), 10, 0.5, sleep=sleeps.append) is True
    assert sleeps == [0.5, 0.5]


def test_zero_timeout_tries_once_and_returns_false():
    calls: list[int] = []

    def condition() -> bool:
        calls.append(1)
        return False

    assert poll_until(condition, 0, 1, sleep=lambda _seconds: None) is False
    assert calls == [1]


def test_gives_up_once_the_timeout_has_passed():
    calls: list[int] = []

    def condition() -> bool:
        calls.append(1)
        return False

    assert poll_until(condition, 0.05, 0.01) is False
    assert len(calls) > 1
