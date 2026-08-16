"""Budget guards and graph routing. Zero API calls — this is the point of the design.

`budget_verdict` is a pure function of state, so every breach path can be checked
against a hand-built dict. That is what makes "budgets are a headline feature"
something a reader can verify rather than take on trust.
"""

import time

import pytest

from research_agent.budget import (
    budget_verdict,
    loop_budget_usd,
    recursion_limit,
    remaining,
    tool_allowed,
)
from research_agent.config import Settings
from research_agent.graph import route_after_observe, route_after_reflect
from research_agent.state import ResearchState, initial_state


def config(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "max_steps": 8,
        "max_seconds": 120.0,
        "max_run_cost_usd": 0.05,
        "finalize_reserve_usd": 0.008,
        "max_replans": 1,
        "max_tool_failures": 3,
        "compact_threshold_tokens": 12_000,
        "max_compactions": 3,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def state(**overrides: object) -> ResearchState:
    base = initial_state("does it matter?")
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


# --- budget_verdict: every breach path ------------------------------------------


def test_a_fresh_run_may_continue() -> None:
    assert budget_verdict(state(), config()) is None


def test_step_cap() -> None:
    assert budget_verdict(state(step=8), config(max_steps=8)) == "max_steps"
    assert budget_verdict(state(step=7), config(max_steps=8)) is None


def test_wall_clock_cap() -> None:
    # t_start is a perf_counter reading, so backdating it simulates a long run
    # without actually waiting.
    old = state(t_start=time.perf_counter() - 500)
    assert budget_verdict(old, config(max_seconds=120)) == "wall_clock"


def test_usd_cap_trips_at_the_loop_budget_not_the_full_budget() -> None:
    # The reserve is withheld so finalize can still afford to answer. A loop that
    # spent right up to max_run_cost_usd would return nothing at all.
    cfg = config(max_run_cost_usd=0.05, finalize_reserve_usd=0.008)
    assert loop_budget_usd(cfg) == pytest.approx(0.042)
    assert budget_verdict(state(spend_usd=0.041), cfg) is None
    assert budget_verdict(state(spend_usd=0.042), cfg) == "budget_usd"


def test_reserve_larger_than_budget_does_not_go_negative() -> None:
    assert loop_budget_usd(config(max_run_cost_usd=0.001, finalize_reserve_usd=0.01)) == 0.0


def test_consecutive_tool_failures_trip_the_circuit_breaker() -> None:
    cfg = config(max_tool_failures=3)
    assert budget_verdict(state(consecutive_tool_failures=2), cfg) is None
    assert budget_verdict(state(consecutive_tool_failures=3), cfg) == "tool_failures"


def test_replan_cap() -> None:
    cfg = config(max_replans=1)
    assert budget_verdict(state(replans=1), cfg) is None
    assert budget_verdict(state(replans=2), cfg) == "max_replans"


def test_per_tool_cap() -> None:
    # web_search is capped at 8 calls per run by its ToolSpec.
    assert budget_verdict(state(tool_calls_by_type={"web_search": 7}), config()) is None
    assert budget_verdict(state(tool_calls_by_type={"web_search": 8}), config()) == "tool_cap"


def test_an_existing_reason_is_never_overwritten() -> None:
    # Whichever node decided first owns the explanation; a later check must not
    # relabel it, or the trace would report the wrong cause.
    breached = state(early_exit_reason="tool_failures", step=99)
    assert budget_verdict(breached, config(max_steps=8)) == "tool_failures"


def test_verdict_is_deterministic_for_the_same_state() -> None:
    # The router and finalize call this independently and must agree; they only do
    # so because it is a pure function.
    fixed = state(step=8)
    assert budget_verdict(fixed, config()) == budget_verdict(fixed, config())


def test_recursion_limit_leaves_room_for_our_own_budgets() -> None:
    # LangGraph's limit counts super-steps and is a backstop, not a control: it
    # must be unreachable so GraphRecursionError means "the graph is broken".
    cfg = config(max_steps=8)
    assert recursion_limit(cfg) == 44
    assert recursion_limit(cfg) > 3 * cfg.max_steps


def test_remaining_reports_headroom_on_every_axis() -> None:
    left = remaining(state(step=3, spend_usd=0.02), config())
    assert left["steps"] == 5
    assert left["usd"] == pytest.approx(0.022)
    assert 0 < left["seconds"] <= 120


def test_remaining_never_goes_negative() -> None:
    left = remaining(state(step=99, spend_usd=9.0), config())
    assert left["steps"] == 0 and left["usd"] == 0.0


def test_tool_allowed_respects_the_registry_cap() -> None:
    assert tool_allowed(state(tool_calls_by_type={"calculator": 9}), "calculator")
    assert not tool_allowed(state(tool_calls_by_type={"calculator": 10}), "calculator")
    assert not tool_allowed(state(), "no_such_tool")


# --- route_after_observe --------------------------------------------------------


def test_observe_routes_to_finalize_on_any_breach() -> None:
    assert route_after_observe(state(step=99), config()) == "finalize"
    assert route_after_observe(state(spend_usd=9.0), config()) == "finalize"
    assert route_after_observe(state(consecutive_tool_failures=5), config()) == "finalize"


def test_observe_routes_to_reflect_normally() -> None:
    assert route_after_observe(state(step=1), config()) == "reflect"


def test_observe_routes_to_compact_when_the_prompt_grew() -> None:
    grew = state(step=1, llm_calls=[{"purpose": "act", "input_tokens": 20_000}])
    assert route_after_observe(grew, config(compact_threshold_tokens=12_000)) == "compact"


def test_compaction_stops_after_max_compactions() -> None:
    # Rolling summaries drift; past the cap the run stops degrading further and
    # just proceeds.
    grew = state(step=1, compactions=3, llm_calls=[{"purpose": "act", "input_tokens": 20_000}])
    assert route_after_observe(grew, config(max_compactions=3)) == "reflect"


def test_compaction_triggers_on_act_tokens_not_any_call() -> None:
    # A huge judge or plan prompt must not trigger compaction of the agent loop.
    other = state(step=1, llm_calls=[{"purpose": "plan", "input_tokens": 90_000}])
    assert route_after_observe(other, config()) == "reflect"


def test_baseline_variant_answers_after_one_pass() -> None:
    assert route_after_observe(state(variant="baseline", step=1), config()) == "finalize"


def test_no_overrule_variant_skips_the_coverage_check() -> None:
    running = state(variant="no_overrule", step=1)
    assert route_after_observe(running, config()) == "reflect"
    stopped = state(variant="no_overrule", step=1, act_requested_stop=True)
    assert route_after_observe(stopped, config()) == "finalize"


# --- route_after_reflect --------------------------------------------------------


def test_reflect_routes_on_its_own_decision() -> None:
    assert route_after_reflect(state(reflect_decision="continue"), config()) == "act"
    assert route_after_reflect(state(reflect_decision="replan"), config()) == "plan"
    assert route_after_reflect(state(reflect_decision="finalize"), config()) == "finalize"


def test_reflect_budget_is_rechecked_because_reflect_itself_costs_money() -> None:
    # A run can cross its cap between the two routers: reflect is an LLM call.
    spent = state(reflect_decision="continue", spend_usd=9.0)
    assert route_after_reflect(spent, config()) == "finalize"


def test_a_replan_past_the_cap_finalizes_rather_than_looping() -> None:
    # Without this, "replan" would be an unbounded loop generator.
    looping = state(reflect_decision="replan", replans=2)
    assert route_after_reflect(looping, config(max_replans=1)) == "finalize"


def test_missing_decision_finalizes_rather_than_spinning() -> None:
    # reflect returning unparseable output must not leave the graph without a route.
    assert route_after_reflect(state(reflect_decision=None), config()) == "finalize"
