from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PRICING = {
    "openai": {
        "gpt-4o": {"input": 5.00, "output": 15.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "gpt-4": {"input": 30.00, "output": 60.00},
    },
    "anthropic": {
        "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
        "claude-opus-3-5-20241022": {"input": 15.00, "output": 75.00},
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        "claude-sonnet-3-5-20241022": {"input": 3.00, "output": 15.00},
        "claude-haiku-3-20240307": {"input": 0.25, "output": 1.25},
    },
    "gemini": {
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-1.5-flash-8b": {"input": 0.0375, "output": 0.15},
    },
    "claude_code": {"claude": {"input": 0.0, "output": 0.0}},
    "openclaw": {"openclaw": {"input": 0.0, "output": 0.0}},
    "ollama": {"local": {"input": 0.0, "output": 0.0}},
}


@dataclass
class CostRecord:
    provider: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    timestamp: float = field(default_factory=lambda: __import__("time").time())


class CostTracker:
    def __init__(self):
        self._records: list[CostRecord] = []
        self._session_total: float = 0.0
        self._provider_totals: dict[str, float] = {}

    def record(
        self,
        provider: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
    ) -> float:
        cost = self.estimate_cost(provider, model, tokens_in, tokens_out)
        record = CostRecord(
            provider=provider,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
        )
        self._records.append(record)
        self._session_total += cost
        self._provider_totals[provider] = self._provider_totals.get(provider, 0.0) + cost
        return cost

    def estimate_cost(
        self,
        provider: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
    ) -> float:
        if provider not in PRICING:
            return 0.0

        provider_pricing = PRICING[provider]

        normalized_model = model
        if model not in provider_pricing:
            for known_model in provider_pricing:
                if model.startswith(known_model.split("-")[0]):
                    normalized_model = known_model
                    break
            else:
                return 0.0

        if normalized_model not in provider_pricing:
            return 0.0

        pricing = provider_pricing[normalized_model]
        input_cost = (tokens_in / 1_000_000) * pricing["input"]
        output_cost = (tokens_out / 1_000_000) * pricing["output"]

        return round(input_cost + output_cost, 6)

    def get_session_total(self) -> float:
        return round(self._session_total, 6)

    def get_provider_breakdown(self) -> dict[str, float]:
        return {k: round(v, 6) for k, v in self._provider_totals.items()}

    def get_recent_costs(self, limit: int = 50) -> list[dict[str, Any]]:
        records = self._records[-limit:]
        return [
            {
                "provider": r.provider,
                "model": r.model,
                "tokens_in": r.tokens_in,
                "tokens_out": r.tokens_out,
                "cost_usd": r.cost_usd,
                "timestamp": r.timestamp,
            }
            for r in records
        ]

    def reset(self):
        self._records.clear()
        self._session_total = 0.0
        self._provider_totals.clear()


_global_cost_tracker = CostTracker()


def get_cost_tracker() -> CostTracker:
    return _global_cost_tracker
