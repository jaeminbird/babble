"""Token and dollar accounting across the calls of one run."""

from __future__ import annotations

import threading

from babble.llm import cost
from babble.types import Usage


class Accounting:
    def __init__(self) -> None:
        self.usage = Usage()
        self.cost_usd = 0.0
        # Models with no entry in the price table. Their tokens are counted but add $0.
        self.unpriced: list[str] = []
        # Subagents share one Accounting and add usage from parallel threads.
        self._lock = threading.Lock()

    def add(self, usage: Usage, model: str) -> None:
        with self._lock:
            self.usage = self.usage + usage
            try:
                self.cost_usd += cost(usage, model)
            except KeyError:
                if model not in self.unpriced:
                    self.unpriced.append(model)
