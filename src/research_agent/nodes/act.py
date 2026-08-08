"""act — decide the next tool call(s).

Builds the turn in the order the prompt cache wants (see prompts.py): a stable
system + question prefix, then the append-only conversation, then a freshly
rendered context block carrying the current source registry. Everything before that
last block stays byte-identical across turns and is served from cache at 10% of
list price.

Tools are offered only while they are under their per-run cap, so a model that has
already used its eight searches simply stops being told `web_search` exists. That is
gentler than letting it call and be refused, and it costs nothing.

Emitting no tool calls is a legitimate outcome — it means the model believes it can
answer — so it sets `act_requested_stop` rather than being treated as a failure.
"""

from typing import Any

from research_agent.budget import tool_allowed
from research_agent.config import Settings, settings
from research_agent.llm import CostTracker, QueryBudgetExceeded, call_tools
from research_agent.nodes import call_records, make_step, spend_delta
from research_agent.prompts import SYSTEM, render_context, responses_tools
from research_agent.state import ResearchState, ToolCall
from research_agent.tools.registry import REGISTRY, openai_schemas


def available_tools(state: ResearchState) -> list[dict[str, Any]]:
    """Only the tools still under their per-run cap."""
    names = [name for name in REGISTRY if tool_allowed(state, name)]
    return responses_tools(openai_schemas(names))


def build_history(state: ResearchState) -> list[dict[str, Any]]:
    """The full turn, ordered for cache reuse."""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": state["question"]},
        *state.get("messages", []),
        {"role": "user", "content": render_context(state)},
    ]


def act_node(
    state: ResearchState, tracker: CostTracker, cfg: Settings | None = None
) -> ResearchState:
    cfg = cfg or settings()
    seen = len(tracker.calls)
    tools = available_tools(state)

    if not tools:
        # Every tool is capped out. Answer with what we have.
        return {
            "act_requested_stop": True,
            "pending_calls": [],
            "step": 1,
            "scratchpad": [make_step(state, "act", note="all tools exhausted")],
        }

    try:
        turn = call_tools(
            build_history(state),
            tools,
            model=cfg.agent_model,
            reasoning_effort=cfg.reasoning_effort or None,
            purpose="act",
            tracker=tracker,
        )
    except QueryBudgetExceeded:
        return {
            "early_exit_reason": "budget_usd",
            "act_requested_stop": True,
            "pending_calls": [],
            "step": 1,
            "llm_calls": call_records(tracker, seen),
            "spend_usd": spend_delta(tracker, seen),
            "scratchpad": [make_step(state, "act", note="budget exceeded during act")],
        }

    pending: list[ToolCall] = [
        ToolCall(call_id=call["call_id"], name=call["name"], args=call["args"])
        for call in turn.tool_calls
    ]

    return {
        # The model's own output items, echoed back next turn as history. `status`
        # has already been stripped by llm.py — the API emits it and rejects it.
        "messages": turn.items,
        "pending_calls": pending,
        "act_requested_stop": not pending,
        "step": 1,
        "llm_calls": call_records(tracker, seen),
        "spend_usd": spend_delta(tracker, seen),
        "scratchpad": [make_step(state, "act", tool_calls=pending, note=turn.text[:200] or None)],
    }
