"""Token and dollar accounting across the calls of one run."""

from __future__ import annotations

from faultline.llm import cost
from faultline.types import Usage


class Accounting:
    def __init__(self) -> None:
        self.usage = Usage()
        self.cost_usd = 0.0
        self.by_model: dict[str, Usage] = {}
        self.calls = 0
        # Models with no entry in the price table. Their tokens are counted but add $0.
        self.unpriced: list[str] = []

    def add(self, usage: Usage, model: str) -> None:
        self.usage = self.usage + usage
        self.by_model[model] = self.by_model.get(model, Usage()) + usage
        self.calls += 1
        try:
            self.cost_usd += cost(usage, model)
        except KeyError:
            if model not in self.unpriced:
                self.unpriced.append(model)
