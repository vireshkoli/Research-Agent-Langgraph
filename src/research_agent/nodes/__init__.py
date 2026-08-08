"""The six graph nodes, plus shared helpers for recording steps and LLM calls.

Every node follows the same contract:

- It returns a **partial** state update — its delta, never the whole list. Under
  `operator.add` a node that returns the full list silently duplicates everything.
- It catches `QueryBudgetExceeded` around its own LLM call and turns it into
  `early_exit_reason`, so a breach becomes a routing decision rather than a
  traceback. That is what makes "a breach returns a partial answer" true.
- It appends exactly one `Step` to the scratchpad, so the trace is a faithful
  record of the path taken.
"""

import time
from typing import Any

from research_agent.llm import CostTracker, LLMCall
from research_agent.state import Observation, ResearchState, Step, ToolCall


def make_step(
    state: ResearchState,
    node: str,
    *,
    tool_calls: list[ToolCall] | None = None,
    observations: list[Observation] | None = None,
    reflection: dict[str, Any] | None = None,
    note: str | None = None,
) -> Step:
    return Step(
        i=len(state.get("scratchpad", [])) + 1,
        node=node,
        plan_version=state.get("plan_version", 0),
        ts_offset_ms=(time.perf_counter() - state.get("t_start", time.perf_counter())) * 1000,
        tool_calls=tool_calls or [],
        observations=observations or [],
        reflection=reflection,
        note=note,
    )


def call_records(tracker: CostTracker, already_seen: int) -> list[dict[str, Any]]:
    """The tracker's new calls as plain dicts, ready for the additive state key."""
    return [_as_dict(call) for call in tracker.calls[already_seen:]]


def _as_dict(call: LLMCall) -> dict[str, Any]:
    return {
        "purpose": call.purpose,
        "model": call.model,
        "input_tokens": call.input_tokens,
        "cached_tokens": call.cached_tokens,
        "output_tokens": call.output_tokens,
        "reasoning_tokens": call.reasoning_tokens,
        "cost_usd": call.cost_usd,
        "latency_ms": call.latency_ms,
        "replayed": call.replayed,
    }


def spend_delta(tracker: CostTracker, already_seen: int) -> float:
    """Only the spend this node added; `spend_usd` is an additive state key."""
    return sum(call.cost_usd for call in tracker.calls[already_seen:])
