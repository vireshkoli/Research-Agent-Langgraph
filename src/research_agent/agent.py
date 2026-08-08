"""run() — the outermost layer, and the one that makes the crash guarantee true.

The graph handles budget breaches by routing to `finalize`. This module handles the
two things the graph cannot handle itself:

- **`GraphRecursionError`** means the routing has a bug that our own budgets failed
  to catch. The run is not abandoned: state is read back out of the checkpointer and
  synthesised deterministically. This is why the checkpointer is not optional even
  for a single-shot run — it is the crash-recovery buffer, not a chat feature.
- **Any other exception** gets the same treatment, with `internal_error` recorded.

So every exit from `run()` returns a `RunTrace` with an answer in it. There is no
path that raises at the caller, which is what the eval's adversarial tier depends
on: an unanswerable question still has to produce something scoreable.

The `CostTracker` is created here, once, and threaded through every node. Nodes are
bound to it with closures at graph-build time rather than being passed it through
state, because a tracker in state would be serialised into every checkpoint.
"""

import time
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from research_agent.budget import recursion_limit
from research_agent.config import Settings, settings
from research_agent.graph import ACT, COMPACT, FINALIZE, OBSERVE, PLAN, REFLECT, build_graph
from research_agent.llm import CostTracker
from research_agent.nodes.act import act_node
from research_agent.nodes.compact import compact_node
from research_agent.nodes.finalize import _synthesize_deterministic, resolve_citations
from research_agent.nodes.observe import observe_node
from research_agent.nodes.plan import plan_node
from research_agent.nodes.reflect import reflect_node
from research_agent.state import ResearchState, Variant, initial_state
from research_agent.trace import RunTrace, build_trace, write_trace


def build_agent(tracker: CostTracker, cfg: Settings, checkpointer: Any = None) -> Any:
    """Bind the nodes to this run's tracker and compile the graph."""
    return build_graph(
        {
            PLAN: lambda state: plan_node(state, tracker, cfg),
            ACT: lambda state: act_node(state, tracker, cfg),
            OBSERVE: lambda state: observe_node(state, cfg),
            COMPACT: lambda state: compact_node(state, tracker, cfg),
            REFLECT: lambda state: reflect_node(state, tracker, cfg),
            FINALIZE: lambda state: _finalize(state, tracker, cfg),
        },
        cfg=cfg,
        checkpointer=checkpointer or InMemorySaver(),
    )


def _finalize(state: ResearchState, tracker: CostTracker, cfg: Settings) -> ResearchState:
    from research_agent.nodes.finalize import finalize_node

    return finalize_node(state, tracker, cfg)


def run(
    question: str,
    variant: Variant = "full",
    cfg: Settings | None = None,
    trace_path: Path | None = None,
) -> RunTrace:
    """Answer one question. Always returns a trace; never raises at the caller."""
    cfg = cfg or settings()
    tracker = CostTracker(budget_usd=cfg.max_run_cost_usd, reserve_usd=cfg.finalize_reserve_usd)
    state = initial_state(question, variant)
    thread = {
        "configurable": {"thread_id": state["run_id"]},
        "recursion_limit": recursion_limit(cfg),
    }
    graph = build_agent(tracker, cfg)

    started = time.perf_counter()
    error: str | None = None

    try:
        final: ResearchState = graph.invoke(state, thread)
    except GraphRecursionError:
        # Our budgets should always trip first, so this means the graph is broken.
        # Recover what state exists rather than losing the whole run.
        final = _recover(graph, thread, state, "recursion_limit")
        error = "GraphRecursionError: the graph exceeded its recursion limit"
    except Exception as exc:  # noqa: BLE001 — the whole point is that nothing escapes
        final = _recover(graph, thread, state, "internal_error")
        error = f"{type(exc).__name__}: {exc}"

    elapsed_ms = (time.perf_counter() - started) * 1000
    trace = build_trace(final, cfg, elapsed_ms, error=error)
    if trace_path:
        write_trace(trace, trace_path)
    return trace


def _recover(
    graph: Any, thread: dict[str, Any], fallback: ResearchState, reason: str
) -> ResearchState:
    """Read state back from the checkpointer and synthesise without an LLM."""
    state = fallback
    try:
        snapshot = graph.get_state(thread)
        if snapshot and snapshot.values:
            state = dict(snapshot.values)  # type: ignore[assignment]
    except Exception:  # noqa: BLE001 — recovery must not itself fail
        pass

    state["early_exit_reason"] = reason  # type: ignore[typeddict-item]
    if not state.get("answer"):
        answer = _synthesize_deterministic(state, reason)
        state["answer"] = answer
        state["citations"] = resolve_citations(state, answer)
        state["used_deterministic_finalize"] = True
    return state
