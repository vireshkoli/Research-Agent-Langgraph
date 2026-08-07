"""One pure function that decides when the run must stop.

LangGraph's conditional-edge functions are *pure routers*: they receive state and
return a node name, and they cannot write state. So the edge that decides to bail
cannot record *why* it bailed. The way out is a single deterministic function called
twice — once by the router to choose `finalize`, once inside `finalize` to write the
reason. Because it is a pure function of state, the two answers agree by
construction rather than by discipline.

Being pure also means the entire budget test suite runs against hand-built state
dicts with no API calls, which is most of what makes "budgets are a headline
feature" checkable rather than merely claimed.

The reserve is the subtle part. If the loop ran until spend hit 100% of the cap,
`finalize`'s own LLM call would raise and the run would return nothing at all — the
opposite of graceful degradation. So the loop stops at `max_run_cost_usd -
finalize_reserve_usd` and `finalize` releases the remainder for itself.
"""

from research_agent.config import Settings
from research_agent.state import EarlyExitReason, ResearchState, elapsed_seconds
from research_agent.tools.registry import REGISTRY


def budget_verdict(state: ResearchState, config: Settings) -> EarlyExitReason | None:
    """Why this run must stop now, or None to continue. Pure: no I/O, no LLM.

    Checked cheapest-first, and in the order a reader would want to see reported:
    a run that blew both its step count and its wall clock reports the step count,
    because that is the one the operator can act on.
    """
    # An earlier node already decided; never overwrite its reason with a later one.
    if reason := state.get("early_exit_reason"):
        return reason

    if state.get("step", 0) >= config.max_steps:
        return "max_steps"

    if elapsed_seconds(state) >= config.max_seconds:
        return "wall_clock"

    if state.get("spend_usd", 0.0) >= loop_budget_usd(config):
        return "budget_usd"

    if state.get("consecutive_tool_failures", 0) >= config.max_tool_failures:
        return "tool_failures"

    if state.get("replans", 0) > config.max_replans:
        return "max_replans"

    counts = state.get("tool_calls_by_type", {})
    for name, spec in REGISTRY.items():
        if counts.get(name, 0) >= spec.max_calls:
            return "tool_cap"

    return None


def loop_budget_usd(config: Settings) -> float:
    """What the loop may spend, holding back enough for finalize to answer."""
    return max(0.0, config.max_run_cost_usd - config.finalize_reserve_usd)


def recursion_limit(config: Settings) -> int:
    """LangGraph's own ceiling, set high enough that our budgets always trip first.

    It counts *super-steps*, not nodes: one act->observe->reflect round is three.
    Sizing it as a backstop rather than a control means a GraphRecursionError is a
    genuine "this graph is broken" signal instead of a routine occurrence.
    """
    return 4 * config.max_steps + 12


def remaining(state: ResearchState, config: Settings) -> dict[str, float]:
    """Headroom on every axis, for the UI meter and the trace."""
    return {
        "steps": max(0, config.max_steps - state.get("step", 0)),
        "seconds": max(0.0, config.max_seconds - elapsed_seconds(state)),
        "usd": max(0.0, loop_budget_usd(config) - state.get("spend_usd", 0.0)),
    }


def tool_allowed(state: ResearchState, name: str) -> bool:
    """Whether one more call to `name` is within its per-run cap."""
    spec = REGISTRY.get(name)
    if spec is None:
        return False
    return state.get("tool_calls_by_type", {}).get(name, 0) < spec.max_calls
