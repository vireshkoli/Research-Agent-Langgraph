"""Compaction, and the crash-recovery paths in agent.run(). No API calls.

The headline test is `test_compaction_cannot_lose_a_citation_even_with_an_adversarial_summary`.
It does not check that the summarisation prompt asks nicely; it disables the
summariser's cooperation entirely — returning a summary that mentions no source ids
at all — and shows the citations survive anyway. They survive because the registry
lives in `state["sources"]` and compaction rewrites `state["scratchpad"]`. Those are
different things, so no summariser mistake can reach the citations.

That test fails the day someone moves sources into message text, which is exactly
when you would want to be told.
"""

from typing import Any

import pytest
from langgraph.errors import GraphRecursionError

from research_agent import agent as agent_module
from research_agent.config import Settings
from research_agent.llm import CostTracker, QueryBudgetExceeded
from research_agent.nodes import compact as compact_module
from research_agent.nodes.compact import compact_node
from research_agent.prompts import render_context
from research_agent.state import Observation, ResearchState, Source, Step, initial_state


def cfg(**overrides: object) -> Settings:
    base: dict[str, object] = {"keep_last_steps": 3, "max_compactions": 3}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def tracker() -> CostTracker:
    return CostTracker(budget_usd=1.0)


def observation(tool: str, content: str, sids: list[str]) -> Observation:
    return Observation(
        call_id="c",
        tool=tool,
        args={"query": "q"},
        ok=True,
        content=content,
        error=None,
        latency_ms=1.0,
        raw_chars=len(content),
        truncated=False,
        source_ids=sids,
    )


def step(index: int, content: str, sids: list[str]) -> Step:
    return Step(
        i=index,
        node="observe",
        plan_version=1,
        ts_offset_ms=0.0,
        tool_calls=[],
        observations=[observation("web_search", content, sids)],
        reflection=None,
        note=None,
    )


def source(sid: str) -> Source:
    return Source(
        sid=sid,
        url=f"https://example.com/{sid}",
        title=f"Doc {sid}",
        snippet="s",
        tool="web_search",
        first_seen_step=0,
    )


def loaded_state(n_steps: int = 8) -> ResearchState:
    """A run with plenty of history and one source per step."""
    state = initial_state("Which is larger, A or B?")
    state["plan"] = ["How big is A?", "How big is B?"]
    state["covered"] = {"How big is A?": True, "How big is B?": False}
    state["sources"] = [source(f"S{i}") for i in range(1, n_steps + 1)]
    state["scratchpad"] = [
        step(i, f"Finding number {i}. " + ("filler text " * 60), [f"S{i}"])
        for i in range(1, n_steps + 1)
    ]
    return state


def apply(state: ResearchState, update: ResearchState) -> ResearchState:
    """Merge a node's partial update the way the graph's reducers would."""
    merged = dict(state)
    for key, value in update.items():
        if key in ("scratchpad", "sources", "llm_calls"):
            merged[key] = list(merged.get(key, [])) + list(value)  # type: ignore[arg-type]
        elif key in ("compactions", "spend_usd"):
            merged[key] = merged.get(key, 0) + value  # type: ignore[operator]
        else:
            merged[key] = value
    return merged  # type: ignore[return-value]


# --- the guarantee ---------------------------------------------------------------


def test_compaction_cannot_lose_a_citation_even_with_an_adversarial_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The summariser is made maximally unhelpful: it returns a summary naming no
    # sources at all. Every citation must still be visible to the model afterwards.
    monkeypatch.setattr(
        compact_module, "complete", lambda *a, **k: "Some things happened. Nothing to cite."
    )
    before = loaded_state()
    expected = {s["sid"] for s in before["sources"]}

    update = compact_node(before, tracker(), cfg())
    after = apply(before, update)

    assert "sources" not in update, "compaction must not touch the source registry at all"
    assert {s["sid"] for s in after["sources"]} == expected

    rendered = render_context(after)
    for sid in expected:
        assert f"[{sid}]" in rendered, f"{sid} vanished from the model's view"


def test_citations_reach_the_model_from_the_registry_not_from_the_scratchpad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The structural property the test above relies on, asserted directly: wipe the
    # scratchpad entirely and every citation is still rendered into the prompt,
    # because it is read from state["sources"].
    state = loaded_state()
    state["scratchpad"] = []
    state["summary"] = ""

    rendered = render_context(state)

    for sid in (s["sid"] for s in state["sources"]):
        assert f"[{sid}]" in rendered


# --- ordinary behaviour ----------------------------------------------------------


def test_compaction_shrinks_the_rendered_context(monkeypatch: pytest.MonkeyPatch) -> None:
    # Catches the genuinely degenerate case where a small model's "summary" comes
    # back longer than what it summarised.
    monkeypatch.setattr(compact_module, "complete", lambda *a, **k: "- A is 405B [S1]\n- B unknown")
    before = loaded_state()
    after = apply(before, compact_node(before, tracker(), cfg()))

    from research_agent.prompts import render_digest

    folded = len(render_digest(before)) - len(render_digest(after))
    assert after["summary"], "a summary was produced"
    assert after["compacted_upto"] == 5, "8 steps minus the 3 kept verbatim"
    assert folded >= 0


def test_the_last_n_steps_stay_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    # The model needs recent detail to choose its next call, so the tail is never
    # folded into the summary.
    monkeypatch.setattr(compact_module, "complete", lambda *a, **k: "summary")
    before = loaded_state(8)
    after = apply(before, compact_node(before, tracker(), cfg(keep_last_steps=3)))

    kept = after["scratchpad"][after["compacted_upto"] :]
    assert any("Finding number 6" in s["observations"][0]["content"] for s in kept)
    assert any("Finding number 8" in s["observations"][0]["content"] for s in kept)


def test_no_step_is_ever_summarised_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    # A rolling summary already drifts; folding the same evidence in twice would
    # compound that and inflate whatever it says.
    folded: list[str] = []

    def capture(system: str, user: str, **k: object) -> str:
        folded.append(user)
        return "summary"

    monkeypatch.setattr(compact_module, "complete", capture)
    state = loaded_state(8)

    state = apply(state, compact_node(state, tracker(), cfg(keep_last_steps=3)))
    first_window = folded[-1]
    state["scratchpad"] = state["scratchpad"] + [step(9, "Finding number 9.", ["S9"])]
    compact_node(state, tracker(), cfg(keep_last_steps=3))
    second_window = folded[-1]

    assert "Finding number 1." in first_window
    assert "Finding number 1." not in second_window, "step 1 was folded a second time"
    assert len(folded) == 2


def test_compaction_is_a_noop_when_everything_is_already_folded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def counted(*a: object, **k: object) -> str:
        calls.append(1)
        return "summary"

    monkeypatch.setattr(compact_module, "complete", counted)
    # 3 steps with keep_last_steps=3 leaves nothing outside the verbatim tail.
    update = compact_node(loaded_state(3), tracker(), cfg(keep_last_steps=3))

    assert calls == [], "no summariser call when the window is empty"
    assert "summary" not in update


def test_an_empty_summary_does_not_blank_the_existing_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compact_module, "complete", lambda *a, **k: "   ")
    before = loaded_state()
    before["summary"] = "previously established facts"
    update = compact_node(before, tracker(), cfg())

    assert "summary" not in update
    assert apply(before, update)["summary"] == "previously established facts"


def test_a_budget_breach_during_compaction_degrades_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broke(*a: object, **k: object) -> None:
        raise QueryBudgetExceeded("no money")

    monkeypatch.setattr(compact_module, "complete", broke)
    update = compact_node(loaded_state(), tracker(), cfg())
    assert update["early_exit_reason"] == "budget_usd"


# --- agent.run(): the outermost guarantees ---------------------------------------


class ExplodingGraph:
    """A compiled graph that fails the way LangGraph would."""

    def __init__(self, error: Exception, recovered: ResearchState | None = None) -> None:
        self.error = error
        self.recovered = recovered

    def invoke(self, state: ResearchState, thread: dict[str, Any]) -> ResearchState:
        raise self.error

    def get_state(self, thread: dict[str, Any]) -> Any:
        if self.recovered is None:
            raise RuntimeError("checkpointer unavailable")
        return type("Snapshot", (), {"values": self.recovered})()


def test_a_recursion_error_recovers_state_and_still_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Our budgets should always trip first, so this means the graph is broken.
    # The run is still not abandoned: state comes back out of the checkpointer.
    partial = loaded_state(3)
    monkeypatch.setattr(
        agent_module,
        "build_agent",
        lambda *a, **k: ExplodingGraph(GraphRecursionError("loop"), partial),
    )
    trace = agent_module.run("Which is larger, A or B?")

    assert trace.outcome.status == "error"
    assert trace.outcome.early_exit_reason == "recursion_limit"
    assert trace.outcome.used_deterministic_finalize is True
    assert trace.answer, "an answer is produced from recovered state"
    assert "S1" in trace.answer, "recovered sources appear in the answer"


def test_any_exception_is_contained_and_still_produces_a_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_module,
        "build_agent",
        lambda *a, **k: ExplodingGraph(RuntimeError("something unexpected"), loaded_state(2)),
    )
    trace = agent_module.run("Which is larger, A or B?")

    assert trace.outcome.status == "error"
    assert trace.outcome.early_exit_reason == "internal_error"
    assert "RuntimeError: something unexpected" in (trace.outcome.error or "")
    assert trace.answer


def test_recovery_survives_an_unreadable_checkpointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Recovery must not itself be a source of failure.
    monkeypatch.setattr(
        agent_module, "build_agent", lambda *a, **k: ExplodingGraph(RuntimeError("boom"), None)
    )
    trace = agent_module.run("Which is larger, A or B?")

    assert trace.answer
    assert trace.outcome.early_exit_reason == "internal_error"


def test_run_never_raises_whatever_the_graph_does(monkeypatch: pytest.MonkeyPatch) -> None:
    # The property the adversarial eval tier depends on: an unanswerable question
    # still has to produce something scoreable.
    for error in (
        GraphRecursionError("loop"),
        RuntimeError("boom"),
        ValueError("bad state"),
        KeyError("missing"),
    ):
        monkeypatch.setattr(
            agent_module, "build_agent", lambda *a, e=error, **k: ExplodingGraph(e, loaded_state(1))
        )
        trace = agent_module.run("q")
        assert trace.answer, f"{type(error).__name__} produced no answer"
