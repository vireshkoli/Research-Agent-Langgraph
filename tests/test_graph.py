"""The compiled graph, driven with stub nodes. Still zero API calls.

`build_graph` takes its node implementations as an argument precisely so this is
possible: the topology exercised here is the same object the real agent runs, so a
green suite means the wiring is right, not that a parallel mock of it is.
"""

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from research_agent.budget import recursion_limit
from research_agent.config import Settings
from research_agent.graph import (
    ACT,
    COMPACT,
    FINALIZE,
    OBSERVE,
    PLAN,
    REFLECT,
    build_graph,
    mermaid,
)
from research_agent.state import ResearchState, initial_state


class Recorder:
    """Stub nodes that record the path taken and apply scripted state updates."""

    def __init__(self, **updates: Any) -> None:
        self.visits: list[str] = []
        self.updates = updates

    def nodes(self) -> dict[str, Any]:
        return {name: self._node(name) for name in (PLAN, ACT, OBSERVE, COMPACT, REFLECT, FINALIZE)}

    def _node(self, name: str) -> Any:
        def run(state: ResearchState) -> dict[str, Any]:
            self.visits.append(name)
            update = self.updates.get(name)
            result = update(state, len(self.visits)) if callable(update) else dict(update or {})
            return {**result, "scratchpad": [_step(name, len(self.visits))]}

        return run


def _step(node: str, index: int) -> dict[str, Any]:
    return {
        "i": index,
        "node": node,
        "plan_version": 0,
        "ts_offset_ms": 0.0,
        "tool_calls": [],
        "observations": [],
        "reflection": None,
        "note": None,
    }


def config(**overrides: object) -> Settings:
    base: dict[str, object] = {"max_steps": 8, "max_seconds": 120.0, "max_run_cost_usd": 0.05}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def run(
    recorder: Recorder,
    state: ResearchState | None = None,
    limit: int = 44,
    cfg: Settings | None = None,
) -> ResearchState:
    graph = build_graph(recorder.nodes(), cfg=cfg)
    return graph.invoke(state or initial_state("q"), {"recursion_limit": limit})


# --- structure ------------------------------------------------------------------


def test_build_graph_rejects_a_missing_node() -> None:
    partial = Recorder().nodes()
    del partial[REFLECT]
    with pytest.raises(ValueError, match="missing node"):
        build_graph(partial)


def test_the_graph_compiles_and_accepts_a_checkpointer() -> None:
    # The checkpointer is not optional even for single-shot runs: it is the buffer
    # agent.run() reads state back from after a GraphRecursionError.
    graph = build_graph(Recorder().nodes(), checkpointer=InMemorySaver())
    result = graph.invoke(
        initial_state("q"), {"configurable": {"thread_id": "t1"}, "recursion_limit": 44}
    )
    assert result["answer"] == ""


# --- paths ----------------------------------------------------------------------


def test_shortest_path_is_plan_act_observe_reflect_finalize() -> None:
    recorder = Recorder(
        **{ACT: {"step": 1}, REFLECT: {"reflect_decision": "finalize"}, FINALIZE: {"answer": "a"}}
    )
    result = run(recorder)
    assert recorder.visits == [PLAN, ACT, OBSERVE, REFLECT, FINALIZE]
    assert result["answer"] == "a"


def test_act_always_reaches_observe_even_with_no_tool_calls() -> None:
    # The edge is static by design: one cheap super-step buys a diagram whose
    # arrows are the whole story.
    recorder = Recorder(
        **{
            ACT: {"step": 1, "act_requested_stop": True},
            REFLECT: {"reflect_decision": "finalize"},
        }
    )
    run(recorder)
    assert recorder.visits[:3] == [PLAN, ACT, OBSERVE]


def test_reflect_can_send_the_loop_back_to_act() -> None:
    def reflect(state: ResearchState, _n: int) -> dict[str, Any]:
        # Two more rounds, then stop.
        keep_going = state.get("step", 0) < 3
        return {"reflect_decision": "continue" if keep_going else "finalize"}

    recorder = Recorder(**{ACT: {"step": 1}, REFLECT: reflect})
    run(recorder)
    assert recorder.visits.count(ACT) == 3
    assert recorder.visits.count(OBSERVE) == 3
    assert recorder.visits[-1] == FINALIZE


def test_reflect_can_send_the_loop_back_to_plan() -> None:
    def reflect(state: ResearchState, _n: int) -> dict[str, Any]:
        first_time = state.get("replans", 0) == 0
        return {"reflect_decision": "replan" if first_time else "finalize", "replans": 1}

    recorder = Recorder(**{ACT: {"step": 1}, REFLECT: reflect})
    run(recorder)
    assert recorder.visits.count(PLAN) == 2
    assert recorder.visits[-1] == FINALIZE


def test_compaction_is_entered_and_leads_to_reflect() -> None:
    recorder = Recorder(
        **{
            ACT: {"step": 1, "llm_calls": [{"purpose": "act", "input_tokens": 50_000}]},
            REFLECT: {"reflect_decision": "finalize"},
        }
    )
    run(recorder)
    assert recorder.visits == [PLAN, ACT, OBSERVE, COMPACT, REFLECT, FINALIZE]


# --- breach paths: the headline feature -----------------------------------------


def test_a_step_breach_skips_reflect_entirely() -> None:
    # Over budget, the run must stop spending on reflection and go straight to
    # producing whatever answer it can.
    recorder = Recorder(**{ACT: {"step": 99}})
    run(recorder)
    assert recorder.visits == [PLAN, ACT, OBSERVE, FINALIZE]
    assert REFLECT not in recorder.visits


def test_a_spend_breach_skips_reflect_entirely() -> None:
    recorder = Recorder(**{ACT: {"step": 1, "spend_usd": 9.0}})
    run(recorder)
    assert recorder.visits == [PLAN, ACT, OBSERVE, FINALIZE]


def test_tool_failures_trip_the_circuit_breaker_mid_loop() -> None:
    recorder = Recorder(**{ACT: {"step": 1}, OBSERVE: {"consecutive_tool_failures": 3}})
    run(recorder)
    assert recorder.visits == [PLAN, ACT, OBSERVE, FINALIZE]


def test_every_run_ends_at_finalize() -> None:
    # The guarantee behind "a breach returns a partial answer, never a crash":
    # there is no path to END that skips synthesis.
    for updates in (
        {ACT: {"step": 99}},
        {ACT: {"step": 1, "spend_usd": 9.0}},
        {ACT: {"step": 1}, REFLECT: {"reflect_decision": "finalize"}},
        {ACT: {"step": 1, "act_requested_stop": True}, REFLECT: {"reflect_decision": "finalize"}},
    ):
        recorder = Recorder(**updates)
        run(recorder)
        assert recorder.visits[-1] == FINALIZE, updates


def test_a_per_run_config_override_is_honoured_by_the_routers() -> None:
    # Regression: LangGraph invokes a conditional edge with state alone, so routers
    # registered as bare functions reached for the global settings() and ignored
    # any override. `--max-steps 1` ran two steps before this was bound at build
    # time. Every budget-breach eval case would have run on default budgets.
    def reflect(_state: ResearchState, _n: int) -> dict[str, Any]:
        return {"reflect_decision": "continue"}  # never voluntarily stops

    recorder = Recorder(**{ACT: {"step": 1}, REFLECT: reflect})
    result = run(recorder, cfg=config(max_steps=1))

    assert result["step"] == 1, "the run must stop after exactly one step"
    assert recorder.visits.count(ACT) == 1
    assert recorder.visits == [PLAN, ACT, OBSERVE, FINALIZE]


def test_budgets_trip_before_langgraph_recursion_limit() -> None:
    # If GraphRecursionError ever fires in practice the graph is broken, so it must
    # be unreachable while our own budgets are doing their job.
    def reflect(_state: ResearchState, _n: int) -> dict[str, Any]:
        return {"reflect_decision": "continue"}  # never voluntarily stops

    recorder = Recorder(**{ACT: {"step": 1}, REFLECT: reflect})
    cfg = config(max_steps=8)
    result = run(recorder, limit=recursion_limit(cfg))

    assert recorder.visits[-1] == FINALIZE
    assert result["step"] >= 8


# --- variants -------------------------------------------------------------------


def test_baseline_variant_is_a_single_pass() -> None:
    recorder = Recorder(**{ACT: {"step": 1}})
    run(recorder, initial_state("q", variant="baseline"))
    assert recorder.visits == [PLAN, ACT, OBSERVE, FINALIZE]


def test_no_reflect_variant_never_visits_reflect_after_a_stop() -> None:
    recorder = Recorder(**{ACT: {"step": 1, "act_requested_stop": True}})
    run(recorder, initial_state("q", variant="no_reflect"))
    assert REFLECT not in recorder.visits


# --- reducers -------------------------------------------------------------------


def test_additive_reducers_receive_deltas_not_totals() -> None:
    # Under operator.add a node that returns the whole list instead of its new
    # items silently duplicates everything, and nothing else in the system will
    # tell you. This is the guard for that class of bug.
    recorder = Recorder(**{ACT: {"step": 1}, REFLECT: {"reflect_decision": "finalize"}})
    result = run(recorder)

    assert result["step"] == 1, "act returned a delta of 1, so the total must be 1"
    assert len(result["scratchpad"]) == len(recorder.visits)
    assert [s["node"] for s in result["scratchpad"]] == recorder.visits


def test_reducerless_keys_are_replaced_not_merged() -> None:
    # `plan` must be replaced wholesale on a replan; concatenating would carry a
    # dead plan's sub-questions into the new one.
    recorder = Recorder(
        **{
            PLAN: {"plan": ["a", "b"]},
            ACT: {"step": 1},
            REFLECT: {"reflect_decision": "finalize", "plan": ["c"]},
        }
    )
    result = run(recorder)
    assert result["plan"] == ["c"]


def test_merge_counts_sums_rather_than_overwrites() -> None:
    recorder = Recorder(
        **{
            ACT: {"step": 1, "tool_calls_by_type": {"web_search": 1}},
            OBSERVE: {"tool_calls_by_type": {"web_search": 2, "calculator": 1}},
            REFLECT: {"reflect_decision": "finalize"},
        }
    )
    result = run(recorder)
    assert result["tool_calls_by_type"] == {"web_search": 3, "calculator": 1}


# --- diagram --------------------------------------------------------------------


def test_mermaid_covers_every_node_and_both_routers() -> None:
    # The README diagram is generated from the same constants as the graph, so it
    # cannot drift out of sync with the wiring.
    diagram = mermaid()
    for node in (PLAN, ACT, OBSERVE, COMPACT, REFLECT, FINALIZE):
        assert node in diagram
    assert "route_after_observe" in diagram
    assert "route_after_reflect" in diagram
    assert diagram.startswith("flowchart TD")
