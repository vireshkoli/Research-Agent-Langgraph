"""Public-demo spend controls, and the UI generator that drives them.

The thing being protected here is a real API key behind a public URL. The tests
that matter are the ones asserting money *is not* spent: that the cap refuses, that
a borrowed key is never billed to the owner, and that the borrowed key is put back
afterwards so it cannot leak into the next visitor's request.
"""

import os
from collections.abc import Iterator
from typing import Any

import pytest

from research_agent import demo_guard
from research_agent.config import settings
from research_agent.trace import Outcome, RunTrace
from research_agent.ui import render as ui


@pytest.fixture(autouse=True)
def isolated_demo_db(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("RA_DEMO_DB", str(tmp_path / "demo.sqlite3"))
    monkeypatch.setenv("RA_DAILY_CAP_USD", "0.25")
    monkeypatch.setenv("RA_MAX_QUESTION_CHARS", "500")
    settings.cache_clear()
    yield
    settings.cache_clear()


def fake_trace(cost: float = 0.002) -> RunTrace:
    return RunTrace(
        run_id="r1",
        ts="now",
        question="q",
        variant="full",
        config={},
        plan={},
        steps=[],
        sources=[
            {
                "sid": "S1",
                "url": "https://a",
                "title": "A",
                "snippet": "",
                "tool": "web_search",
                "first_seen_step": 0,
            }
        ],
        answer="The answer [S1].",
        citations=["S1"],
        coverage={},
        outcome=Outcome(status="completed"),
        usage={
            "totals": {
                "cost_usd": cost,
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_hit_rate": 0.0,
            },
            "steps": 1,
            "search_credits": 1,
        },
        timings_ms={"total": 1000.0},
    )


# --- the cap ---------------------------------------------------------------------


def test_a_fresh_day_allows_a_question() -> None:
    assert demo_guard.check("Who won the 2024 Nobel Prize in Physics?", None) is None


def test_spend_accumulates_and_then_refuses() -> None:
    demo_guard.record(0.20)
    assert demo_guard.check("still fine?", None) is None

    demo_guard.record(0.06)  # now over $0.25
    refusal = demo_guard.check("too late?", None)

    assert refusal is not None
    assert "budget for today" in refusal
    assert "your own OpenAI API key" in refusal, "the refusal must offer a way forward"


def test_a_visitor_with_their_own_key_bypasses_the_cap() -> None:
    # The cap protects the owner's key. Someone paying with theirs is not the
    # thing being protected against.
    demo_guard.record(99.0)
    assert demo_guard.check("anything", None) is not None
    assert demo_guard.check("anything", "sk-visitor-key") is None


def test_an_empty_or_oversized_question_is_refused_before_any_spend() -> None:
    assert "Ask a question" in (demo_guard.check("   ", None) or "")
    long_question = "x" * 5000
    refusal = demo_guard.check(long_question, None)
    assert refusal and "characters" in refusal


def test_the_cap_survives_a_restart() -> None:
    # The counter is in SQLite rather than memory precisely because a free host
    # restarts containers, and an in-memory cap would reset every time.
    demo_guard.record(0.30)
    assert demo_guard.spent_today() == (pytest.approx(0.30), 1)
    assert demo_guard.check("q", None) is not None


def test_a_zero_cap_means_uncapped_for_local_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RA_DAILY_CAP_USD", "0")
    settings.cache_clear()
    demo_guard.record(99.0)
    assert demo_guard.check("q", None) is None
    assert "uncapped" in demo_guard.status_line()


# --- borrowed keys ---------------------------------------------------------------


def test_a_borrowed_key_is_used_then_put_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-owner")

    with demo_guard.borrowed_key("sk-visitor"):
        assert os.environ["OPENAI_API_KEY"] == "sk-visitor"

    assert os.environ["OPENAI_API_KEY"] == "sk-owner", "the owner's key must be restored"


def test_a_borrowed_key_is_restored_even_when_the_run_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without the finally, one failed request would leave a visitor's key installed
    # for everyone who came after.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-owner")

    with pytest.raises(RuntimeError), demo_guard.borrowed_key("sk-visitor"):
        raise RuntimeError("the run blew up")

    assert os.environ["OPENAI_API_KEY"] == "sk-owner"


def test_borrowing_nothing_leaves_the_environment_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-owner")
    with demo_guard.borrowed_key(None):
        assert os.environ["OPENAI_API_KEY"] == "sk-owner"
    assert os.environ["OPENAI_API_KEY"] == "sk-owner"


# --- the UI generator ------------------------------------------------------------


def test_the_ui_streams_progress_then_a_final_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_stream(question: str, **kwargs: Any) -> Iterator[tuple[str, Any]]:
        yield "progress", {"node": "plan", "update": {"plan": ["How big is A?"]}}
        yield (
            "progress",
            {
                "node": "act",
                "update": {"pending_calls": [{"call_id": "c", "name": "web_search", "args": {}}]},
            },
        )
        yield "done", fake_trace()

    monkeypatch.setattr(ui, "stream", fake_stream)
    frames = list(ui.answer("Who won?", "", "full"))

    assert len(frames) >= 3, "at least one frame per progress event plus the final one"
    assert frames[-1][0] == "The answer [S1]."
    assert "Planning" in frames[-1][1]
    assert "S1" in frames[-1][2]
    assert all(frame[0] == "" for frame in frames[:-1]), "the answer appears only at the end"


def test_the_ui_records_spend_only_when_the_owner_paid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_stream(question: str, **kwargs: Any) -> Iterator[tuple[str, Any]]:
        yield "done", fake_trace(cost=0.01)

    monkeypatch.setattr(ui, "stream", fake_stream)

    list(ui.answer("q", "", "full"))
    assert demo_guard.spent_today()[0] == pytest.approx(0.01)

    list(ui.answer("q", "sk-visitor", "full"))
    assert demo_guard.spent_today()[0] == pytest.approx(0.01), "a visitor's key is not billed here"


def test_the_ui_refuses_without_starting_a_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*a: object, **k: object) -> None:
        raise AssertionError("the agent should not have run")

    monkeypatch.setattr(ui, "stream", explode)
    demo_guard.record(99.0)

    frames = list(ui.answer("q", "", "full"))
    assert len(frames) == 1
    assert "budget for today" in frames[0][0]


def test_the_view_redraws_while_a_node_is_still_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The reason the generator drives a worker thread rather than iterating
    # stream() directly: a node that takes 20s used to freeze the display for the
    # whole of it, which reads as a hang. Here one slow node must still produce
    # several frames.
    import time as _time

    def slow_stream(question: str, **kwargs: Any) -> Iterator[tuple[str, Any]]:
        _time.sleep(0.5)
        yield "done", fake_trace()

    monkeypatch.setattr(ui, "stream", slow_stream)
    frames = list(ui.answer("q", "", "full"))

    assert len(frames) > 3, "a slow node should still tick the UI"
    assert frames[-1][0] == "The answer [S1]."


def test_a_failure_on_the_worker_thread_is_re_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    # An exception inside the thread must not be swallowed into a run that appears
    # to hang forever.
    def broken_stream(question: str, **kwargs: Any) -> Iterator[tuple[str, Any]]:
        raise RuntimeError("worker exploded")
        yield  # pragma: no cover

    monkeypatch.setattr(ui, "stream", broken_stream)
    with pytest.raises(RuntimeError, match="worker exploded"):
        list(ui.answer("q", "", "full"))


def test_the_pipeline_preview_shows_what_is_coming_next() -> None:
    # A viewer who can see the shape of the run reads a pause as progress rather
    # than as a hang.
    waiting = ui._progress_markdown([], None, done=[], elapsed=1.2, frame=0)
    assert "Planning" in waiting
    assert "1.2s elapsed" in waiting

    mid = ui._progress_markdown(["- done"], None, done=["plan", "act"], elapsed=5.0, frame=3)
    assert "Running tools" in mid, "observe follows act"


def test_the_preview_does_not_advertise_stages_already_completed() -> None:
    # The graph loops, so a naive "everything after this" list would keep offering
    # stages the run has already been through twice.
    text = ui._progress_markdown(
        [], None, done=["plan", "act", "observe", "reflect"], elapsed=9.0, frame=1
    )
    assert "Choosing tools" in text, "reflect routes back to act"
    assert "Planning" not in text


def test_no_spinner_once_the_run_has_finished() -> None:
    finished = ui._progress_markdown([], None, done=["finalize"], elapsed=12.0, frame=0)
    assert "elapsed" not in finished
    assert not any(char in finished for char in ui.SPINNER)


def test_an_early_exit_is_explained_in_the_trace_pane(monkeypatch: pytest.MonkeyPatch) -> None:
    trace = fake_trace()
    trace.outcome = Outcome(status="early_exit", early_exit_reason="max_steps")

    def fake_stream(question: str, **kwargs: Any) -> Iterator[tuple[str, Any]]:
        yield "done", trace

    monkeypatch.setattr(ui, "stream", fake_stream)
    final = list(ui.answer("q", "", "full"))[-1]

    assert "Stopped early" in final[1]
    assert "max_steps" in final[1]
