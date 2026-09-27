"""Think-if-unsure: when a decision gets a short thought before it (docs/think-if-unsure-plan.md).

Request extension (absent = the single pass, exactly as before):
    "thinking": {"mode": "off" | "auto" | "always", "max_tokens": 256, "threshold": 0.7, "unknown_threshold": null,
                 "return_thought": false}
`auto` thinks on a question only when its single-pass answer is unsure: the top listed option's share of the non-unknown mass is
below `threshold`, or (when set) the unknown mass is above `unknown_threshold`. The model thinks with Qwen3.5's thinking
template, greedily, for at most `max_tokens`; the decision is then read right after '</think>' as in the single pass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .contracts import UNKNOWN

MODES = ("off", "auto", "always")
MAX_THOUGHT_TOKENS = 4096


@dataclass(frozen=True)
class ThinkingPolicy:
    mode: str = "off"
    max_tokens: int = 256
    threshold: float = 0.7
    unknown_threshold: float | None = None
    return_thought: bool = False

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"thinking.mode must be one of {', '.join(MODES)}")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or not 1 <= self.max_tokens <= MAX_THOUGHT_TOKENS:
            raise ValueError(f"thinking.max_tokens must be an integer in 1..{MAX_THOUGHT_TOKENS}")
        for name in ("threshold", "unknown_threshold"):
            value = getattr(self, name)
            if value is None and name == "unknown_threshold":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"thinking.{name} must be a number in [0, 1]")
        if not isinstance(self.return_thought, bool):
            raise ValueError("thinking.return_thought must be true or false")

    @property
    def active(self):
        return self.mode != "off"

    def with_request(self, value):
        """This policy (the server's defaults) overridden by a request's `thinking` object (None = keep the defaults)."""
        if value is None:
            return self
        if not isinstance(value, dict):
            raise ValueError("thinking must be an object")
        unknown = set(value) - {"mode", "max_tokens", "threshold", "unknown_threshold", "return_thought"}
        if unknown:
            raise ValueError(f"thinking has unknown keys: {', '.join(sorted(unknown))}")
        return replace(self, **value)

    def should_think(self, result):
        if self.mode == "always":
            return True
        if self.mode == "off":
            return False
        confidence, unknown = unsureness(result)
        return confidence < self.threshold or (self.unknown_threshold is not None and unknown > self.unknown_threshold)


def unsureness(result):
    """(top listed option's share of the non-unknown mass, unknown mass) of a single-pass Result."""
    unknown = float(result.scores.get(UNKNOWN, 0.0))
    real = [p for key, p in result.scores.items() if key != UNKNOWN]
    mass = sum(real)
    return (max(real) / mass if mass > 0 else 0.0), unknown
