"""poll_until: the one retry loop shared by the lab and bench waits."""

from __future__ import annotations

import time
from collections.abc import Callable


def poll_until(
    condition: Callable[[], bool],
    timeout_seconds: float,
    poll_interval_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Call `condition` until it returns true and return True. Between calls, sleep
    `poll_interval_seconds`. Once `timeout_seconds` has passed without success, return False.
    The condition is always tried at least once."""
    started_at = time.monotonic()
    while True:
        if condition():
            return True
        if time.monotonic() - started_at >= timeout_seconds:
            return False
        sleep(poll_interval_seconds)
