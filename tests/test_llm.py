"""Cost accounting and budget guards. No API key required — nothing here calls OpenAI."""

from types import SimpleNamespace

import pytest

from research_agent import llm, spend
from research_agent.llm import (
    CostTracker,
    LLMCall,
    ProcessBudgetExceeded,
    QueryBudgetExceeded,
    cost_usd,
)

# gpt-5.4-nano list prices: $0.20 / $0.02 / $1.25 per 1M tokens.
NANO = "gpt-5.4-nano"


def make_call(cost: float, purpose: str = "act") -> LLMCall:
    return LLMCall(
        purpose=purpose,
        model=NANO,
        input_tokens=1000,
        cached_tokens=0,
        output_tokens=100,
        reasoning_tokens=0,
        cost_usd=cost,
        latency_ms=12.5,
    )


def fake_response(prompt: int, cached: int, completion: int, reasoning: int = 0) -> SimpleNamespace:
    """A Responses-API-shaped usage block."""
    return SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=prompt,
            output_tokens=completion,
            input_tokens_details=SimpleNamespace(cached_tokens=cached),
            output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        )
    )


def settle(usage: tuple[int, int, int, int], **overrides: object) -> llm.LLMCall:
    kwargs: dict[str, object] = {
        "model": NANO,
        "purpose": "act",
        "usage": usage,
        "elapsed_ms": 1.0,
        "batch": False,
        "replayed": False,
        "tracker": None,
    }
    kwargs.update(overrides)
    return llm._settle(**kwargs)  # type: ignore[arg-type]


# --- pricing -------------------------------------------------------------------


def test_cost_matches_list_price() -> None:
    # (1000 * 0.20 + 1000 * 1.25) / 1e6
    assert cost_usd(NANO, input_tokens=1000, output_tokens=1000) == pytest.approx(0.00145)


def test_cached_tokens_are_billed_at_the_cached_rate_and_not_twice() -> None:
    # OpenAI reports cached tokens *inside* prompt_tokens, so 1000 input with 800
    # cached is 200 at $0.20 plus 800 at $0.02 — not 1000 at $0.20 plus a surcharge.
    assert cost_usd(NANO, input_tokens=1000, cached_tokens=800) == pytest.approx(0.000056)


def test_caching_is_cheaper_than_not_caching() -> None:
    uncached = cost_usd(NANO, input_tokens=10_000)
    cached = cost_usd(NANO, input_tokens=10_000, cached_tokens=9_000)
    assert cached < uncached / 5


def test_batch_is_half_price() -> None:
    full = cost_usd(NANO, input_tokens=1000, output_tokens=1000)
    assert cost_usd(NANO, input_tokens=1000, output_tokens=1000, batch=True) == pytest.approx(
        full / 2
    )


def test_cached_tokens_exceeding_input_do_not_produce_negative_cost() -> None:
    assert cost_usd(NANO, input_tokens=100, cached_tokens=500) >= 0


def test_unknown_model_names_itself_in_the_error() -> None:
    with pytest.raises(ValueError, match="no list price"):
        cost_usd("gpt-nonexistent", input_tokens=100)


# --- per-run budget ------------------------------------------------------------


def test_tracker_raises_the_moment_spend_crosses_budget() -> None:
    tracker = CostTracker(budget_usd=0.01)
    tracker.record(make_call(0.006))
    assert tracker.total_cost_usd == pytest.approx(0.006)
    with pytest.raises(QueryBudgetExceeded, match="exceeds"):
        tracker.record(make_call(0.005))


def test_the_breaching_call_is_still_recorded() -> None:
    # The trace has to show what tipped the run over, so the call is appended
    # before the guard fires.
    tracker = CostTracker(budget_usd=0.01)
    with pytest.raises(QueryBudgetExceeded):
        tracker.record(make_call(0.02, purpose="finalize"))
    assert len(tracker.calls) == 1
    assert tracker.calls[0].purpose == "finalize"


def test_reserve_is_withheld_from_the_loop() -> None:
    # $0.01 budget with $0.004 reserved leaves the loop $0.006 to spend.
    tracker = CostTracker(budget_usd=0.01, reserve_usd=0.004)
    assert tracker.effective_budget_usd == pytest.approx(0.006)
    with pytest.raises(QueryBudgetExceeded):
        tracker.record(make_call(0.007))


def test_releasing_the_reserve_lets_finalize_spend_it() -> None:
    # This is the guarantee that a budget-exhausted run still returns an answer
    # rather than nothing at all.
    tracker = CostTracker(budget_usd=0.01, reserve_usd=0.004)
    tracker.record(make_call(0.0055))
    tracker.release_reserve()
    assert tracker.remaining_usd == pytest.approx(0.0045)
    tracker.record(make_call(0.004, purpose="finalize"))
    assert tracker.total_cost_usd == pytest.approx(0.0095)


def test_cache_hit_rate_is_reported_over_all_calls() -> None:
    tracker = CostTracker(budget_usd=1.0)
    for cached in (0, 900):
        tracker.record(
            LLMCall(
                purpose="act",
                model=NANO,
                input_tokens=1000,
                cached_tokens=cached,
                output_tokens=0,
                reasoning_tokens=0,
                cost_usd=0.0,
                latency_ms=1.0,
            )
        )
    assert tracker.cache_hit_rate == pytest.approx(0.45)


def test_cache_hit_rate_of_an_empty_tracker_is_zero_not_a_crash() -> None:
    assert CostTracker(budget_usd=1.0).cache_hit_rate == 0.0


# --- usage extraction ----------------------------------------------------------


def test_usage_extraction_reads_all_four_counters() -> None:
    assert llm._usage_of(fake_response(1000, 800, 500, 300)) == (1000, 800, 500, 300)


def test_missing_usage_details_degrade_to_zero_rather_than_raising() -> None:
    # Some responses omit the *_details blocks entirely; cost accounting must not
    # be the thing that breaks a run.
    bare = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            input_tokens_details=None,
            output_tokens_details=None,
        )
    )
    assert llm._usage_of(bare) == (10, 0, 5, 0)
    assert llm._usage_of(SimpleNamespace()) == (0, 0, 0, 0)


def test_reasoning_tokens_are_reported_but_not_billed_separately() -> None:
    # OpenAI bills reasoning tokens as output tokens and already counts them in
    # output_tokens. Adding them again would silently inflate every cost number.
    call = settle((1000, 0, 500, 400))
    assert call.reasoning_tokens == 400
    assert call.cost_usd == pytest.approx(cost_usd(NANO, input_tokens=1000, output_tokens=500))


def test_replayed_calls_cost_nothing_and_say_so() -> None:
    # A cassette hit really is free, so recording a notional price would make the
    # ledger and the README's total spend figure wrong.
    call = settle((10_000, 0, 5_000, 0), replayed=True)
    assert call.replayed is True
    assert call.cost_usd == 0.0
    assert call.input_tokens == 10_000  # tokens are still reported


def test_echoable_strips_the_status_field() -> None:
    # The Responses API emits `status` on its own output items and then rejects it
    # on input, so echoing a turn back verbatim is a 400.
    item = SimpleNamespace(
        model_dump=lambda exclude_none: {
            "type": "function_call",
            "name": "web_search",
            "status": "completed",
        }
    )
    assert llm._echoable(item) == {"type": "function_call", "name": "web_search"}


def test_malformed_tool_arguments_become_data_not_an_exception() -> None:
    assert llm._safe_json('{"query": "ok"}') == {"query": "ok"}
    assert "__malformed__" in llm._safe_json("{not json")
    assert "__malformed__" in llm._safe_json("[1, 2]")  # valid JSON, wrong shape


# --- process-level kill switch -------------------------------------------------


def stub_settings(**overrides: object) -> SimpleNamespace:
    base = {"max_process_cost_usd": 0.0, "max_project_cost_usd": 0.0}
    base.update(overrides)
    return SimpleNamespace(**base)


def test_process_guard_fires_without_any_tracker(monkeypatch: pytest.MonkeyPatch) -> None:
    # The runaway-dev-loop case: nobody passed a CostTracker, so the per-run guard
    # cannot help. This is the backstop that makes that survivable.
    monkeypatch.setattr(llm, "settings", lambda: stub_settings(max_process_cost_usd=0.002))

    settle((1000, 0, 0, 0))
    with pytest.raises(ProcessBudgetExceeded, match="RA_MAX_PROCESS_COST_USD"):
        for _ in range(20):
            settle((10_000, 0, 0, 0))


def test_process_guard_can_be_disabled_with_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "settings", lambda: stub_settings())
    for _ in range(50):
        settle((100_000, 0, 100_000, 0))
    assert llm.process_spend_usd() > 1.0


def test_replayed_calls_do_not_count_against_any_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    # The whole point of cassettes on a tight budget: a hundred replayed dev runs
    # must not consume the project ceiling.
    monkeypatch.setattr(llm, "settings", lambda: stub_settings(max_process_cost_usd=0.001))
    for _ in range(200):
        settle((100_000, 0, 100_000, 0), replayed=True)
    assert llm.process_spend_usd() == 0.0


# --- project ledger: the only guard that survives a restart ---------------------


def test_project_ledger_accumulates_across_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "settings", lambda: stub_settings())
    settle((1_000_000, 0, 0, 0), purpose="act")
    settle((1_000_000, 0, 0, 0), purpose="judge")

    ledger = spend.read()
    assert ledger.calls == 2
    assert ledger.total_usd == pytest.approx(0.40)  # 2 x 1M input at $0.20/M
    assert set(ledger.by_purpose) == {"act", "judge"}


def test_project_cap_stops_further_spend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "settings", lambda: stub_settings(max_project_cost_usd=0.10))
    with pytest.raises(spend.ProjectBudgetExceeded, match="ceiling"):
        for _ in range(10):
            settle((1_000_000, 0, 0, 0))


def test_the_breaching_call_is_still_recorded_in_the_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The ledger has to stay truthful about money actually spent, including on the
    # call that crossed the line — otherwise it under-reports exactly when it matters.
    monkeypatch.setattr(llm, "settings", lambda: stub_settings(max_project_cost_usd=0.10))
    with pytest.raises(spend.ProjectBudgetExceeded):
        settle((1_000_000, 0, 0, 0))
    assert spend.read().total_usd == pytest.approx(0.20)


def test_a_corrupt_ledger_does_not_block_work(monkeypatch: pytest.MonkeyPatch) -> None:
    spend.ledger_path().write_text("{ this is not json")
    ledger = spend.read()
    assert ledger.total_usd == 0.0
    assert ledger.updated == "unreadable"


def test_summary_is_readable_when_empty_and_when_populated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "no calls recorded" in spend.summary()
    monkeypatch.setattr(llm, "settings", lambda: stub_settings())
    settle((1_000_000, 0, 0, 0), purpose="act")
    assert "$0.2000" in spend.summary()
    assert "act" in spend.summary()
