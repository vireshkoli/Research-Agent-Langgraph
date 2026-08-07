"""The OpenAI client. Every LLM call in this project goes through this module.

Three things live here and nowhere else:

1. **Token and cost accounting.** Cost is computed from real usage at published list
   prices, so every number in the README and the eval report is money actually spent.
2. **The per-run budget guard.** `CostTracker.record` raises the moment cumulative
   spend crosses the cap, with a reserve held back so `finalize` can always afford to
   synthesise an answer.
3. **A process-level kill switch**, independent of any tracker, so a runaway
   development loop cannot burn real money even if a caller forgets to pass one.

Chat Completions is used deliberately rather than the Responses API: the gpt-5.4
model cards list Chat Completions and Batch only, and nothing here needs Responses.
"""

import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from research_agent.config import settings

# USD per 1M tokens (input, cached input, output) — OpenAI list prices.
# Cached input is uniformly 10% of input across this generation.
PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-5.6-sol": (5.00, 0.50, 30.00),
    "gpt-5.6-terra": (2.50, 0.25, 15.00),
    "gpt-5.6-luna": (1.00, 0.10, 6.00),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5-nano": (0.05, 0.005, 0.40),
}

# The Batch API is half price. Eval scoring has no latency requirement, so the
# judge runs through it.
BATCH_DISCOUNT = 0.5


class QueryBudgetExceeded(Exception):
    """Raised when one run's accumulated LLM spend crosses its configured cap."""


class ProcessBudgetExceeded(Exception):
    """Raised when total spend across this process crosses RA_MAX_PROCESS_COST_USD.

    This is the guard of last resort. It fires regardless of whether the caller
    passed a CostTracker, and it is not resettable from configuration.
    """


@dataclass
class LLMCall:
    purpose: str
    model: str
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cost_usd: float
    latency_ms: float


@dataclass
class CostTracker:
    """Accumulates LLM spend for one run; enforces the per-run budget.

    `reserve_usd` is withheld from the loop so that when the budget is exhausted
    there is still headroom for the final synthesis call. Without it, hitting the
    cap mid-loop would return no answer at all. `finalize` calls
    `release_reserve()` before its own call.
    """

    budget_usd: float
    reserve_usd: float = 0.0
    calls: list[LLMCall] = field(default_factory=list)
    _reserve_released: bool = False

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_cached_tokens(self) -> int:
        return sum(c.cached_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def cache_hit_rate(self) -> float:
        """Share of input tokens served from the prompt cache, at 10% of list price."""
        total_in = self.total_input_tokens
        return self.total_cached_tokens / total_in if total_in else 0.0

    @property
    def effective_budget_usd(self) -> float:
        return self.budget_usd if self._reserve_released else self.budget_usd - self.reserve_usd

    @property
    def remaining_usd(self) -> float:
        return self.effective_budget_usd - self.total_cost_usd

    def release_reserve(self) -> None:
        """Hand the withheld headroom to the caller. Only finalize should call this."""
        self._reserve_released = True

    def record(self, call: LLMCall) -> None:
        self.calls.append(call)
        if self.total_cost_usd > self.effective_budget_usd:
            raise QueryBudgetExceeded(
                f"run spend ${self.total_cost_usd:.4f} exceeds "
                f"${self.effective_budget_usd:.4f} (purpose={call.purpose})"
            )


def cost_usd(
    model: str,
    input_tokens: int,
    cached_tokens: int = 0,
    output_tokens: int = 0,
    batch: bool = False,
) -> float:
    """Cost of one call in USD.

    `input_tokens` as reported by OpenAI already includes `cached_tokens`, so the
    uncached portion is the difference. `output_tokens` already includes reasoning
    tokens — counting those separately would double-bill them.
    """
    if model not in PRICES:
        raise ValueError(f"no list price for model {model!r}; known: {sorted(PRICES)}")
    in_price, cached_price, out_price = PRICES[model]
    uncached = max(0, input_tokens - cached_tokens)
    total = (uncached * in_price + cached_tokens * cached_price + output_tokens * out_price) / 1e6
    return total * BATCH_DISCOUNT if batch else total


# --- process-level guard -------------------------------------------------------

_process_spend_usd = 0.0


def process_spend_usd() -> float:
    """Total spend across every call made by this process."""
    return _process_spend_usd


def _charge_process(amount: float) -> None:
    global _process_spend_usd
    _process_spend_usd += amount
    cap = settings().max_process_cost_usd
    if cap > 0 and _process_spend_usd > cap:
        raise ProcessBudgetExceeded(
            f"process spend ${_process_spend_usd:.4f} exceeds RA_MAX_PROCESS_COST_USD "
            f"${cap:.2f}. Raise the cap deliberately if this is expected."
        )


def _reset_process_spend() -> None:
    """Test-only hook. Production code must never call this."""
    global _process_spend_usd
    _process_spend_usd = 0.0


# --- client --------------------------------------------------------------------


def _load_env_file() -> None:
    """Populate os.environ from ./.env (KEY=VALUE lines) without overriding real env vars."""
    env_file = Path(".env")
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


@lru_cache
def _client() -> Any:
    from openai import OpenAI

    _load_env_file()
    return OpenAI()


def _usage_of(response: Any) -> tuple[int, int, int, int]:
    """Extract (input, cached, output, reasoning) tokens from a Chat Completions response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0, 0
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return (
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(prompt_details, "cached_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
        getattr(completion_details, "reasoning_tokens", 0) or 0,
    )


def _request_kwargs(
    model: str,
    max_tokens: int,
    temperature: float | None,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    """Assemble the parameters that vary by model family.

    Reasoning models reject `temperature`, so the two are mutually exclusive here.
    """
    kwargs: dict[str, Any] = {"model": model, "max_completion_tokens": max_tokens}
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    elif temperature is not None:
        kwargs["temperature"] = temperature
    return kwargs


def _record(
    response: Any,
    *,
    model: str,
    purpose: str,
    elapsed_ms: float,
    batch: bool,
    tracker: CostTracker | None,
) -> LLMCall:
    """Price one response, charge the process guard, and record it on the tracker.

    The process guard is charged first and unconditionally: it must fire even when
    no tracker was passed, which is exactly the case a runaway dev loop hits.
    """
    input_tokens, cached_tokens, output_tokens, reasoning_tokens = _usage_of(response)
    spend = cost_usd(model, input_tokens, cached_tokens, output_tokens, batch=batch)
    call = LLMCall(
        purpose=purpose,
        model=model,
        input_tokens=input_tokens,
        cached_tokens=cached_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cost_usd=spend,
        latency_ms=elapsed_ms,
    )
    _charge_process(spend)
    if tracker is not None:
        tracker.record(call)
    return call


# --- public API ----------------------------------------------------------------


def complete(
    system: str,
    user: str,
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float | None = 0.0,
    reasoning_effort: str | None = None,
    purpose: str = "generate",
    tracker: CostTracker | None = None,
) -> str:
    """One text completion. Returns the text; records tokens and cost on the tracker."""
    model = model or settings().agent_model
    started = time.perf_counter()
    response = _client().chat.completions.create(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        **_request_kwargs(model, max_tokens, temperature, reasoning_effort),
    )
    _record(
        response,
        model=model,
        purpose=purpose,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        batch=False,
        tracker=tracker,
    )
    return response.choices[0].message.content or ""


def call_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: str | None = None,
    max_tokens: int = 2048,
    reasoning_effort: str | None = None,
    purpose: str = "act",
    tracker: CostTracker | None = None,
) -> dict[str, Any]:
    """One tool-calling turn. Returns the assistant message as a plain wire-format dict.

    Native function calling is used rather than JSON-in-text: the tool call arrives
    already structured, which removes an entire class of parse failure from the loop.
    """
    model = model or settings().agent_model
    started = time.perf_counter()
    response = _client().chat.completions.create(
        messages=messages,
        tools=tools,
        **_request_kwargs(model, max_tokens, None, reasoning_effort),
    )
    _record(
        response,
        model=model,
        purpose=purpose,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        batch=False,
        tracker=tracker,
    )
    message = response.choices[0].message
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in (message.tool_calls or [])
        ],
    }


def parse[T: BaseModel](
    system: str,
    user: str,
    schema: type[T],
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float | None = 0.0,
    reasoning_effort: str | None = None,
    purpose: str = "parse",
    tracker: CostTracker | None = None,
    batch: bool = False,
) -> T | None:
    """One structured-output call. Returns a validated model, or None if it refused.

    Callers must handle None with a deterministic default rather than crashing —
    an unparseable judgement is data ("unparseable"), not an exception.
    """
    model = model or settings().agent_model
    started = time.perf_counter()
    response = _client().chat.completions.parse(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format=schema,
        **_request_kwargs(model, max_tokens, temperature, reasoning_effort),
    )
    _record(
        response,
        model=model,
        purpose=purpose,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        batch=batch,
        tracker=tracker,
    )
    return response.choices[0].message.parsed
