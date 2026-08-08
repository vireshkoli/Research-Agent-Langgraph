"""The five real nodes, with the LLM faked. Still zero API calls.

Faking follows the house style from project #1: monkeypatch the imported name in
the module under test, no mock library. Each node imports `parse`, `complete` or
`call_tools` directly, so replacing that attribute is enough.

The tests that matter most are the failure ones — an unparseable plan, a refused
reflection, a synthesis that raises — because those are the paths that decide
whether a bad run degrades or dies.
"""

import pytest

from research_agent.config import Settings
from research_agent.llm import CostTracker, QueryBudgetExceeded, ToolTurn
from research_agent.nodes import act as act_module
from research_agent.nodes import finalize as finalize_module
from research_agent.nodes import plan as plan_module
from research_agent.nodes import reflect as reflect_module
from research_agent.nodes.act import act_node, build_history
from research_agent.nodes.finalize import (
    _synthesize_deterministic,
    finalize_node,
    resolve_citations,
    unresolved_citations,
)
from research_agent.nodes.observe import observe_node
from research_agent.nodes.plan import PlanOutput, plan_node
from research_agent.nodes.reflect import ReflectOutput, reflect_node
from research_agent.state import ResearchState, Source, ToolCall, initial_state
from research_agent.tools.base import Source as ToolSource
from research_agent.tools.base import ToolResult


def cfg(**overrides: object) -> Settings:
    base: dict[str, object] = {"max_steps": 8, "keep_last_steps": 3}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def tracker() -> CostTracker:
    return CostTracker(budget_usd=1.0, reserve_usd=0.01)


def state(**overrides: object) -> ResearchState:
    base = initial_state("Which is larger, A or B?")
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def source(sid: str, url: str) -> Source:
    return Source(
        sid=sid, url=url, title=f"Title {sid}", snippet="", tool="web_search", first_seen_step=0
    )


# --- plan ------------------------------------------------------------------------


def test_plan_produces_subquestions_and_seeds_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        plan_module,
        "parse",
        lambda *a, **k: PlanOutput(reasoning="r", subquestions=["How big is A?", "How big is B?"]),
    )
    update = plan_node(state(), tracker(), cfg())

    assert update["plan"] == ["How big is A?", "How big is B?"]
    assert update["covered"] == {"How big is A?": False, "How big is B?": False}
    assert update["plan_version"] == 1


def test_plan_falls_back_to_the_question_when_parsing_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A refused structured output is a worse plan, not a dead run.
    monkeypatch.setattr(plan_module, "parse", lambda *a, **k: None)
    update = plan_node(state(), tracker(), cfg())
    assert update["plan"] == ["Which is larger, A or B?"]


def test_plan_deduplicates_and_caps_subquestions(monkeypatch: pytest.MonkeyPatch) -> None:
    # A model returning a dozen sub-questions would spend the whole step budget on
    # searches before answering any of them.
    monkeypatch.setattr(
        plan_module,
        "parse",
        lambda *a, **k: PlanOutput(
            reasoning="r", subquestions=["a", "A", " a ", "b", "c", "d", "e", "f", ""]
        ),
    )
    update = plan_node(state(), tracker(), cfg())
    assert update["plan"] == ["a", "b", "c", "d", "e"]


def test_plan_survives_a_budget_breach_with_a_usable_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broke(*a: object, **k: object) -> None:
        raise QueryBudgetExceeded("no money")

    monkeypatch.setattr(plan_module, "parse", broke)
    update = plan_node(state(), tracker(), cfg())

    assert update["early_exit_reason"] == "budget_usd"
    assert update["plan"] == ["Which is larger, A or B?"], "finalize still needs a plan"


def test_a_replan_replaces_rather_than_extends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        plan_module, "parse", lambda *a, **k: PlanOutput(reasoning="r", subquestions=["new"])
    )
    update = plan_node(state(plan=["old1", "old2"], plan_version=1), tracker(), cfg())

    assert update["plan"] == ["new"], "a merge would carry a dead plan's sub-questions"
    assert update["covered"] == {"new": False}
    assert update["replans"] == 1


# --- act -------------------------------------------------------------------------


def test_act_orders_the_turn_so_the_prefix_stays_cacheable() -> None:
    # Stable system + question first, conversation next, freshly rendered context
    # last. Anything that moved the context earlier would bust the cache on every
    # turn, because the source list grows.
    history = build_history(state(sources=[source("S1", "https://a")]))

    assert history[0]["role"] == "system"
    assert history[1]["content"] == "Which is larger, A or B?"
    assert history[-1]["role"] == "user"
    assert "S1" in history[-1]["content"]


def test_act_emits_pending_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        act_module,
        "call_tools",
        lambda *a, **k: ToolTurn(
            items=[{"type": "function_call", "name": "web_search"}],
            tool_calls=[{"call_id": "c1", "name": "web_search", "args": {"query": "A"}}],
            text="",
        ),
    )
    update = act_node(state(), tracker(), cfg())

    assert update["pending_calls"] == [
        ToolCall(call_id="c1", name="web_search", args={"query": "A"})
    ]
    assert update["act_requested_stop"] is False
    assert update["step"] == 1


def test_act_with_no_tool_calls_requests_a_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    # Not a failure: the model believes it can answer.
    monkeypatch.setattr(
        act_module, "call_tools", lambda *a, **k: ToolTurn(items=[], tool_calls=[], text="done")
    )
    update = act_node(state(), tracker(), cfg())
    assert update["act_requested_stop"] is True
    assert update["pending_calls"] == []


def test_act_stops_offering_tools_that_hit_their_cap() -> None:
    # web_search is capped at 8 per run. Withdrawing it is gentler than letting the
    # model call it and be refused.
    exhausted = state(tool_calls_by_type={"web_search": 8})
    names = {tool["name"] for tool in act_module.available_tools(exhausted)}
    assert "web_search" not in names
    assert "calculator" in names


def test_act_with_every_tool_exhausted_stops_without_calling_the_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*a: object, **k: object) -> None:
        raise AssertionError("should not have called the model")

    monkeypatch.setattr(act_module, "call_tools", explode)
    capped = state(
        tool_calls_by_type={
            "web_search": 8,
            "fetch_page": 8,
            "calculator": 10,
            "code_execution": 5,
            "file_ops": 10,
        }
    )
    update = act_node(capped, tracker(), cfg())
    assert update["act_requested_stop"] is True


def test_act_turns_a_budget_breach_into_a_route_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broke(*a: object, **k: object) -> None:
        raise QueryBudgetExceeded("no money")

    monkeypatch.setattr(act_module, "call_tools", broke)
    update = act_node(state(), tracker(), cfg())

    assert update["early_exit_reason"] == "budget_usd"
    assert update["act_requested_stop"] is True


# --- observe ---------------------------------------------------------------------


def test_observe_mints_source_ids_and_counts_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "research_agent.nodes.observe.dispatch",
        lambda name, args: ToolResult(
            ok=True,
            content="found it",
            sources=[ToolSource(url="https://a", title="A", snippet="s", tool="web_search")],
            meta={"credits": 1},
        ),
    )
    pending = [ToolCall(call_id="c1", name="web_search", args={"query": "A"})]
    update = observe_node(state(pending_calls=pending), cfg())

    assert [s["sid"] for s in update["sources"]] == ["S1"]
    assert update["tool_calls_by_type"] == {"web_search": 1}
    assert update["search_credits"] == 1
    assert update["pending_calls"] == [], "the handoff key must drain"
    assert update["messages"][0]["type"] == "function_call_output"
    assert "[S1]" in update["messages"][0]["output"]


def test_observe_reuses_a_source_id_for_a_url_already_seen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Re-finding a page must not issue a second id, or the registry fills with
    # duplicates and citation counts stop meaning anything.
    monkeypatch.setattr(
        "research_agent.nodes.observe.dispatch",
        lambda name, args: ToolResult(
            ok=True,
            content="again",
            sources=[ToolSource(url="https://a", title="A", snippet="", tool="web_search")],
        ),
    )
    existing = state(
        sources=[source("S1", "https://a")],
        pending_calls=[ToolCall(call_id="c", name="web_search", args={"query": "A"})],
    )
    update = observe_node(existing, cfg())

    assert update["sources"] == [], "no new source should be minted"
    assert update["scratchpad"][0]["observations"][0]["source_ids"] == ["S1"]


def test_observe_counts_consecutive_failures_and_resets_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "research_agent.nodes.observe.dispatch",
        lambda name, args: ToolResult.failure("search is down"),
    )
    failing = state(
        consecutive_tool_failures=1,
        pending_calls=[ToolCall(call_id="c", name="web_search", args={"query": "A"})],
    )
    assert observe_node(failing, cfg())["consecutive_tool_failures"] == 2

    monkeypatch.setattr(
        "research_agent.nodes.observe.dispatch",
        lambda name, args: ToolResult(ok=True, content="fine"),
    )
    recovering = state(
        consecutive_tool_failures=2,
        pending_calls=[ToolCall(call_id="c", name="web_search", args={"query": "A"})],
    )
    assert recovering and observe_node(recovering, cfg())["consecutive_tool_failures"] == 0


def test_observe_reports_malformed_arguments_back_to_the_model() -> None:
    # The model emitted invalid JSON; telling it so beats crashing, and it usually
    # recovers on the next turn.
    broken = state(
        pending_calls=[ToolCall(call_id="c", name="web_search", args={"__malformed__": "{oops"})]
    )
    update = observe_node(broken, cfg())
    observation = update["scratchpad"][0]["observations"][0]
    assert not observation["ok"]
    assert "not valid JSON" in observation["error"]


def test_observe_is_a_noop_when_there_is_nothing_to_run() -> None:
    update = observe_node(state(), cfg())
    assert update["pending_calls"] == []
    assert update["scratchpad"][0]["note"] == "no tool calls"


# --- reflect ---------------------------------------------------------------------


def test_reflect_is_skipped_on_the_first_step() -> None:
    # Paying for a coverage check that can only say "keep going" is waste.
    update = reflect_node(state(step=1), tracker(), cfg())
    assert update["reflect_decision"] == "continue"
    assert "skipped" in update["scratchpad"][0]["note"]


def test_reflect_records_an_overrule_when_it_reverses_a_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This is the number that justifies the node existing. If it stays zero across
    # the eval, the node gets cut and the README says so.
    monkeypatch.setattr(
        reflect_module,
        "parse",
        lambda *a, **k: ReflectOutput(
            reasoning="B is unanswered",
            covered=["How big is A?"],
            open_gaps=["How big is B?"],
            decision="continue",
        ),
    )
    proposed_stop = state(step=2, act_requested_stop=True, plan=["How big is A?", "How big is B?"])
    update = reflect_node(proposed_stop, tracker(), cfg())

    assert update["reflect_decision"] == "continue"
    assert update["reflect_overrules"] == 1
    assert update["act_requested_stop"] is False, "the stop is consumed"


def test_reflect_does_not_record_an_overrule_when_it_agrees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reflect_module,
        "parse",
        lambda *a, **k: ReflectOutput(
            reasoning="all done", covered=["q"], open_gaps=[], decision="finalize"
        ),
    )
    update = reflect_node(state(step=2, act_requested_stop=True, plan=["q"]), tracker(), cfg())
    assert update["reflect_overrules"] == 0


def test_reflect_ignores_coverage_claims_for_questions_not_in_the_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reflect_module,
        "parse",
        lambda *a, **k: ReflectOutput(
            reasoning="r",
            covered=["something invented"],
            open_gaps=["also invented"],
            decision="continue",
        ),
    )
    update = reflect_node(state(step=2, plan=["real question"]), tracker(), cfg())

    assert update["covered"] == {"real question": False}
    assert update["open_gaps"] == ["real question"]


def test_unparseable_reflection_still_yields_a_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reflect_module, "parse", lambda *a, **k: None)
    update = reflect_node(state(step=2), tracker(), cfg())
    assert update["reflect_decision"] in ("continue", "finalize")


# --- finalize --------------------------------------------------------------------


def test_finalize_resolves_only_real_citations() -> None:
    known = state(sources=[source("S1", "https://a"), source("S2", "https://b")])
    answer = "A is bigger [S1]. Also relevant [S2]. And this [S9] does not exist."

    assert resolve_citations(known, answer) == ["S1", "S2"]
    assert unresolved_citations(known, answer) == ["S9"]


def test_finalize_falls_back_to_a_deterministic_answer_when_the_llm_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The guarantee behind "a breach returns a partial result": this path uses no
    # LLM, so it cannot fail for lack of money.
    def broke(*a: object, **k: object) -> None:
        raise QueryBudgetExceeded("no money")

    monkeypatch.setattr(finalize_module, "complete", broke)
    exhausted = state(step=99, plan=["How big is A?"], sources=[source("S1", "https://a")])
    update = finalize_node(exhausted, tracker(), cfg())

    assert update["used_deterministic_finalize"] is True
    assert update["answer"], "an answer is always produced"
    assert "How big is A?" in update["answer"]
    assert "https://a" in update["answer"]
    assert update["early_exit_reason"] == "max_steps"


def test_finalize_explains_an_early_exit_in_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(finalize_module, "complete", lambda *a, **k: "A is bigger.")
    update = finalize_node(state(step=99), tracker(), cfg())
    assert "stopped early" in update["answer"]
    assert "step budget" in update["answer"]


def test_finalize_treats_an_empty_synthesis_as_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An empty string is not an answer; falling back beats returning nothing.
    monkeypatch.setattr(finalize_module, "complete", lambda *a, **k: "   ")
    update = finalize_node(state(plan=["q"]), tracker(), cfg())
    assert update["used_deterministic_finalize"] is True
    assert update["answer"]


def test_deterministic_synthesis_needs_no_llm_and_no_state_beyond_the_run() -> None:
    built = _synthesize_deterministic(
        state(
            plan=["How big is A?"],
            covered={"How big is A?": True},
            sources=[source("S1", "https://a")],
        ),
        "max_steps",
    )
    assert "How big is A?" in built
    assert "[answered]" in built
    assert "https://a" in built


def test_finalize_releases_the_reserve_before_synthesising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The loop stopped short of the cap precisely so this call could happen.
    seen: dict[str, float] = {}
    budget = CostTracker(budget_usd=0.05, reserve_usd=0.008)

    def record(*a: object, **k: object) -> str:
        seen["effective"] = budget.effective_budget_usd
        return "answer"

    monkeypatch.setattr(finalize_module, "complete", record)
    finalize_node(state(), budget, cfg())
    assert seen["effective"] == pytest.approx(0.05)
