"""The OpenAI client. Every LLM call in this project goes through this module.

Built on the **Responses API**. The original plan specified Chat Completions, on the
research finding that the gpt-5.4 model cards list Chat Completions and Batch but not
Responses. A Phase-1 spike measured the opposite constraint:

    Chat Completions + tools + reasoning_effort  ->  400 on nano *and* mini
        "Function tools with reasoning_effort are not supported for gpt-5.4-nano
         in /v1/chat/completions. To use function tools, use /v1/responses or set
         reasoning_effort to 'none'."
    Responses + tools + reasoning                ->  works

Responses supports both modes at identical cost when reasoning is off, so it is
strictly more flexible, and it is what OpenAI recommends for new projects.

Four guards sit between this project and an unexpected bill, each catching what the
one below it cannot:

  1. `CostTracker`      one run     — raises the moment cumulative spend crosses the
                                      cap, holding back a reserve so `finalize` can
                                      always afford to answer.
  2. process guard      one process — fires even when a caller forgets a tracker,
                                      which is exactly the runaway-dev-loop case.
  3. project ledger     forever     — survives restarts; a few hundred short dev runs
                                      across a few hundred processes are invisible to
                                      the first two.
  4. cassettes          dev only    — replays a recorded response for $0 instead of
                                      paying for the same prompt a hundredth time.
"""

import json
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from research_agent import cassette
from research_agent.config import settings
from research_agent.spend import record as record_project_spend

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
    """Raised when spend across this process crosses RA_MAX_PROCESS_COST_USD.

    Fires regardless of whether the caller passed a CostTracker, which is exactly
    the case a runaway development loop hits.
    """


class CassetteMiss(Exception):
    """Strict replay mode was asked for a call that was never recorded."""


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
    replayed: bool = False


@dataclass
class ToolTurn:
    """One tool-calling turn.

    `items` are the model's own output items with `status` stripped, ready to be
    echoed straight back as conversation history — the Responses API rejects the
    field it emits there, and reasoning items must be returned alongside tool
    outputs for reasoning models to work correctly across turns.
    """

    items: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]  # {"call_id", "name", "args"}
    text: str


@dataclass
class CostTracker:
    """Accumulates LLM spend for one run; enforces the per-run budget.

    `reserve_usd` is withheld from the loop so that when the budget is exhausted
    there is still headroom for the final synthesis call. Without it, hitting the
    cap mid-loop would return no answer at all. `finalize` calls `release_reserve()`
    before its own call.
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
    """Extract (input, cached, output, reasoning) tokens from a Responses result."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0, 0
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return (
        getattr(usage, "input_tokens", 0) or 0,
        getattr(input_details, "cached_tokens", 0) or 0,
        getattr(usage, "output_tokens", 0) or 0,
        getattr(output_details, "reasoning_tokens", 0) or 0,
    )


def _echoable(item: Any) -> dict[str, Any]:
    """One output item, shaped so it can be sent straight back as history.

    The API emits `status` on its own output items but rejects it on input, so it
    has to come off before the next turn.
    """
    payload = item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else dict(item)
    payload.pop("status", None)
    return payload


def _request_kwargs(model: str, max_tokens: int, reasoning_effort: str | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"model": model, "max_output_tokens": max_tokens}
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}
    return kwargs


def _settle(
    *,
    model: str,
    purpose: str,
    usage: tuple[int, int, int, int],
    elapsed_ms: float,
    batch: bool,
    replayed: bool,
    tracker: CostTracker | None,
) -> LLMCall:
    """Price one call and charge it to every guard, innermost last.

    Order matters. The project ledger and process guard are charged before the
    tracker, so a run that blows its own budget still leaves both of them truthful
    about what it spent on the way.
    """
    input_tokens, cached_tokens, output_tokens, reasoning_tokens = usage
    spend = 0.0 if replayed else cost_usd(model, input_tokens, cached_tokens, output_tokens, batch)
    call = LLMCall(
        purpose=purpose,
        model=model,
        input_tokens=input_tokens,
        cached_tokens=cached_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cost_usd=spend,
        latency_ms=elapsed_ms,
        replayed=replayed,
    )
    if spend:
        record_project_spend(spend, purpose, settings().max_project_cost_usd)
        _charge_process(spend)
    if tracker is not None:
        tracker.record(call)
    return call


def _cached(cache_key: str) -> dict[str, Any] | None:
    hit = cassette.load(cache_key)
    if hit is None and cassette.misses_are_fatal():
        raise CassetteMiss(
            f"no recording for {cache_key} and RA_LLM_CACHE=replay. "
            "Re-record with RA_LLM_CACHE=auto, or set RA_LLM_CACHE=off."
        )
    return hit


def _usage_from_cache(payload: dict[str, Any]) -> tuple[int, int, int, int]:
    usage = payload.get("usage", {})
    return (
        int(usage.get("input_tokens", 0)),
        int(usage.get("cached_tokens", 0)),
        int(usage.get("output_tokens", 0)),
        int(usage.get("reasoning_tokens", 0)),
    )


# --- public API ----------------------------------------------------------------


def complete(
    system: str,
    user: str,
    model: str | None = None,
    max_tokens: int = 4096,
    reasoning_effort: str | None = None,
    purpose: str = "generate",
    tracker: CostTracker | None = None,
) -> str:
    """One text completion. Returns the text; records tokens and cost on the tracker."""
    model = model or settings().agent_model
    request = _request_kwargs(model, max_tokens, reasoning_effort)
    cache_key = cassette.key({"kind": "complete", "system": system, "user": user, **request})

    if (hit := _cached(cache_key)) is not None:
        _settle(
            model=model,
            purpose=purpose,
            usage=_usage_from_cache(hit),
            elapsed_ms=0.0,
            batch=False,
            replayed=True,
            tracker=tracker,
        )
        return str(hit["text"])

    started = time.perf_counter()
    response = _client().responses.create(
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **request,
    )
    usage = _usage_of(response)
    _settle(
        model=model,
        purpose=purpose,
        usage=usage,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        batch=False,
        replayed=False,
        tracker=tracker,
    )
    text = response.output_text or ""
    cassette.save(cache_key, {"text": text, "usage": _usage_dict(usage)})
    return text


def call_tools(
    history: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: str | None = None,
    max_tokens: int = 4096,
    reasoning_effort: str | None = None,
    purpose: str = "act",
    tracker: CostTracker | None = None,
) -> ToolTurn:
    """One tool-calling turn over the running conversation.

    Native function calling rather than JSON-in-text: the call arrives already
    structured, which removes an entire class of parse failure from the loop.
    """
    model = model or settings().agent_model
    request = _request_kwargs(model, max_tokens, reasoning_effort)
    cache_key = cassette.key({"kind": "call_tools", "history": history, "tools": tools, **request})

    if (hit := _cached(cache_key)) is not None:
        _settle(
            model=model,
            purpose=purpose,
            usage=_usage_from_cache(hit),
            elapsed_ms=0.0,
            batch=False,
            replayed=True,
            tracker=tracker,
        )
        return ToolTurn(items=hit["items"], tool_calls=hit["tool_calls"], text=hit["text"])

    started = time.perf_counter()
    response = _client().responses.create(input=history, tools=tools, **request)
    usage = _usage_of(response)
    _settle(
        model=model,
        purpose=purpose,
        usage=usage,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        batch=False,
        replayed=False,
        tracker=tracker,
    )

    turn = ToolTurn(
        items=[_echoable(item) for item in response.output],
        tool_calls=[
            {
                "call_id": item.call_id,
                "name": item.name,
                "args": _safe_json(item.arguments),
            }
            for item in response.output
            if getattr(item, "type", "") == "function_call"
        ],
        text=response.output_text or "",
    )
    cassette.save(
        cache_key,
        {
            "items": turn.items,
            "tool_calls": turn.tool_calls,
            "text": turn.text,
            "usage": _usage_dict(usage),
        },
    )
    return turn


def parse[T: BaseModel](
    system: str,
    user: str,
    schema: type[T],
    model: str | None = None,
    max_tokens: int = 4096,
    reasoning_effort: str | None = None,
    purpose: str = "parse",
    tracker: CostTracker | None = None,
    batch: bool = False,
) -> T | None:
    """One structured-output call. Returns a validated model, or None if it refused.

    Callers must handle None with a deterministic default rather than crashing — an
    unparseable judgement is data ("unparseable"), not an exception.
    """
    model = model or settings().agent_model
    request = _request_kwargs(model, max_tokens, reasoning_effort)
    cache_key = cassette.key(
        {
            "kind": "parse",
            "system": system,
            "user": user,
            "schema": schema.model_json_schema(),
            **request,
        }
    )

    if (hit := _cached(cache_key)) is not None:
        _settle(
            model=model,
            purpose=purpose,
            usage=_usage_from_cache(hit),
            elapsed_ms=0.0,
            batch=batch,
            replayed=True,
            tracker=tracker,
        )
        return schema.model_validate(hit["parsed"]) if hit.get("parsed") is not None else None

    started = time.perf_counter()
    response = _client().responses.parse(
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text_format=schema,
        **request,
    )
    usage = _usage_of(response)
    _settle(
        model=model,
        purpose=purpose,
        usage=usage,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        batch=batch,
        replayed=False,
        tracker=tracker,
    )
    parsed = response.output_parsed
    cassette.save(
        cache_key,
        {
            "parsed": parsed.model_dump() if parsed is not None else None,
            "usage": _usage_dict(usage),
        },
    )
    return parsed


def _usage_dict(usage: tuple[int, int, int, int]) -> dict[str, int]:
    input_tokens, cached_tokens, output_tokens, reasoning_tokens = usage
    return {
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


def _safe_json(raw: str) -> dict[str, Any]:
    """Tool arguments as a dict. Malformed JSON becomes data the agent can react to."""
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"__malformed__": raw}
    return value if isinstance(value, dict) else {"__malformed__": raw}
