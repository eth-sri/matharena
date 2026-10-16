"""Configurable token prices, selected by the input length of one request."""

import math
from collections.abc import Mapping


class TokenPricing:
    def __init__(
        self, read_cost=0, write_cost=0, cache_read_cost=None, cache_write_cost=0, long_context_pricing=None,
    ):
        self.base = {
            "read_cost": read_cost or 0,
            "write_cost": write_cost or 0,
            "cache_read_cost": (read_cost or 0) if cache_read_cost is None else cache_read_cost,
            "cache_write_cost": cache_write_cost or 0,
        }
        self.threshold = None
        self.long_context = None
        if long_context_pricing is None:
            return
        if not isinstance(long_context_pricing, Mapping):
            raise ValueError("long_context_pricing must be a mapping")
        unknown = set(long_context_pricing) - {"input_token_threshold", *self.base}
        if unknown:
            raise ValueError(f"Unknown long_context_pricing fields: {sorted(unknown)}")
        threshold = long_context_pricing.get("input_token_threshold")
        if type(threshold) is not int or threshold <= 0:
            raise ValueError("long_context_pricing.input_token_threshold must be a positive integer")
        for name in ("read_cost", "write_cost"):
            if name not in long_context_pricing:
                raise ValueError(f"long_context_pricing.{name} is required")
        rates = self.base | {k: v for k, v in long_context_pricing.items() if k in self.base}
        for name, value in rates.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"long_context_pricing.{name} must be a finite nonnegative number")
        self.threshold = threshold
        self.long_context = rates

    @classmethod
    def from_config(cls, config):
        return cls(**{
            name: config.get(name) for name in (
                "read_cost", "write_cost", "cache_read_cost", "cache_write_cost", "long_context_pricing",
            )
        })

    def cost(self, input_tokens, output_tokens, cached_input_tokens=0, cached_write_tokens=0, *, base_only=False):
        """Input includes cache reads; cache writes are a separate category.

        All input categories select the tier, and all tokens in the request
        use that tier. Output tokens never select a tier.
        """
        input_tokens = max(input_tokens, 0)
        output_tokens = max(output_tokens, 0)
        cached_input_tokens = min(max(cached_input_tokens, 0), input_tokens)
        cached_write_tokens = max(cached_write_tokens, 0)
        rates = self.base
        if (
            not base_only
            and self.threshold is not None
            and input_tokens + cached_write_tokens > self.threshold
        ):
            rates = self.long_context
        return (
            (input_tokens - cached_input_tokens) * float(rates["read_cost"])
            + cached_input_tokens * float(rates["cache_read_cost"])
            + cached_write_tokens * float(rates["cache_write_cost"])
            + output_tokens * float(rates["write_cost"])
        ) / 1_000_000
