import threading
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class UsageTracker(BaseCallbackHandler):
    def __init__(self, price_input_per_1m: float | None = None, price_output_per_1m: float | None = None):
        self.price_input_per_1m = price_input_per_1m
        self.price_output_per_1m = price_output_per_1m
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self._lock = threading.Lock()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        input_tokens = output_tokens = 0
        found = False
        for generations in getattr(response, "generations", None) or []:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if usage:
                    found = True
                    input_tokens += int(usage.get("input_tokens", 0))
                    output_tokens += int(usage.get("output_tokens", 0))
        if not found:
            token_usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
            input_tokens = int(token_usage.get("prompt_tokens", 0))
            output_tokens = int(token_usage.get("completion_tokens", 0))

        with self._lock:
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

    def snapshot(self, elapsed: float = 0.0) -> dict:
        with self._lock:
            return make_usage(
                self.input_tokens, self.output_tokens, self.calls, elapsed,
                self.price_input_per_1m, self.price_output_per_1m,
            )


def make_usage(
    input_tokens: int,
    output_tokens: int,
    calls: int,
    elapsed: float,
    price_input_per_1m: float | None = None,
    price_output_per_1m: float | None = None,
) -> dict:
    cost = None
    if price_input_per_1m is not None and price_output_per_1m is not None:
        cost = round(input_tokens / 1e6 * price_input_per_1m + output_tokens / 1e6 * price_output_per_1m, 4)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "calls": calls,
        "elapsed": round(elapsed, 1),
        "cost": cost,
    }


def add_usage(previous: dict | None, current: dict, price_input_per_1m: float | None = None,
              price_output_per_1m: float | None = None) -> dict:
    previous = previous or {}
    return make_usage(
        int(previous.get("input_tokens", 0)) + current["input_tokens"],
        int(previous.get("output_tokens", 0)) + current["output_tokens"],
        int(previous.get("calls", 0)) + current["calls"],
        float(previous.get("elapsed", 0.0)) + current["elapsed"],
        price_input_per_1m,
        price_output_per_1m,
    )
