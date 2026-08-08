"""Matchers, metrics and agreement statistics. No API calls.

These are the definitions the README has to stand behind, so the edge cases get
tests rather than prose: the ones where a harness would otherwise quietly return a
number that flatters the agent.
"""

import pytest
from evals.agreement import (
    Agreement,
    bootstrap_kappa_ci,
    cohens_kappa,
    compare,
    deflation,
    gwets_ac1,
    verbosity_correlation,
)
from evals.matchers import as_text, call_matches, matches, normalize
from evals.metrics import (
    anchors_satisfied,
    citation_metrics,
    independence_gap,
    pass_at_1,
    pass_hat_k,
    percentile,
    step_efficiency,
    tool_metrics,
    wilson_interval,
)
from evals.schema import ArgMatcher, EvalCase, ExpectedTool


def case(**overrides: object) -> EvalCase:
    base: dict[str, object] = {
        "id": "t1",
        "tier": "easy",
        "question": "q",
        "reference_answer": "r",
        "must_include": ["anchor"],
    }
    base.update(overrides)
    return EvalCase.model_validate(base)


def call(name: str, **args: object) -> dict[str, object]:
    return {"name": name, "args": args}


# --- normalisation ----------------------------------------------------------------


def test_normalisation_is_narrow_on_purpose() -> None:
    assert normalize("  Hello   World  ") == "hello world"
    assert normalize('"quoted"') == "quoted"
    assert normalize("5") == 5.0, "a model may emit a number as a string"
    assert normalize(5) == 5.0
    assert normalize(["b", "a"]) == normalize(["a", "b"]), "list order is not meaningful"


def test_booleans_do_not_become_numbers() -> None:
    # bool is a subclass of int in Python; letting True normalise to 1.0 would make
    # `strict=true` match `strict=1`.
    assert normalize(True) is True


def test_as_text_flattens_structures_for_substring_matching() -> None:
    assert "llama" in as_text(["Llama 3", "Mistral"])
    assert "query" in as_text({"query": "Llama"})


# --- matchers ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("actual", "spec", "expected"),
    [
        ("Llama 3.1 405B", ArgMatcher(matcher="contains_any", value=["llama 3", "llama"]), True),
        ("Mistral Large", ArgMatcher(matcher="contains_any", value=["llama"]), False),
        ("llama 3 parameters", ArgMatcher(matcher="contains_all", value=["llama", "param"]), True),
        ("llama 3", ArgMatcher(matcher="contains_all", value=["llama", "mistral"]), False),
        ("405 - 123", ArgMatcher(matcher="regex", value=r"405\s*-\s*123"), True),
        (405, ArgMatcher(matcher="numeric_close", value=405.0), True),
        (404, ArgMatcher(matcher="numeric_close", value=405.0, tolerance=0.01), True),
        (300, ArgMatcher(matcher="numeric_close", value=405.0, tolerance=0.01), False),
        (3, ArgMatcher(matcher="lte", value=5), True),
        (7, ArgMatcher(matcher="lte", value=5), False),
        ("anything at all", ArgMatcher(matcher="any"), True),
        ("exact match", ArgMatcher(matcher="exact", value="Exact Match"), True),
    ],
)
def test_matchers(actual: object, spec: ArgMatcher, expected: bool) -> None:
    assert matches(actual, spec) is expected


def test_numeric_matchers_refuse_non_numbers_instead_of_crashing() -> None:
    assert matches("not a number", ArgMatcher(matcher="numeric_close", value=5)) is False


def test_extra_arguments_are_tolerated_unless_the_case_says_otherwise() -> None:
    # A model passing an optional parameter we did not think to specify has not done
    # anything wrong; superset semantics, from the agentevals taxonomy.
    actual = call("web_search", query="llama 3 parameters", max_results=5)
    expected = {"query": ArgMatcher(matcher="contains_any", value=["llama"])}

    assert call_matches(actual, "web_search", expected) is True
    assert call_matches(actual, "web_search", expected, strict_args=True) is False


def test_a_wrong_tool_name_never_matches() -> None:
    assert call_matches(call("calculator", expression="1+1"), "web_search", {}) is False


# --- tool metrics -------------------------------------------------------------------


def test_perfect_tool_use() -> None:
    spec = case(
        expected_tools=[
            ExpectedTool(
                name="web_search",
                args={"query": ArgMatcher(matcher="contains_any", value=["llama"])},
            ),
            ExpectedTool(name="calculator", args={}),
        ]
    )
    metrics = tool_metrics(
        spec, [call("web_search", query="llama 3"), call("calculator", expression="1+1")]
    )

    assert metrics.precision == 1.0 and metrics.recall == 1.0 and metrics.f1 == 1.0


def test_bipartite_matching_does_not_let_one_call_satisfy_two_expectations() -> None:
    # Greedy left-to-right would match the single call against both expectations and
    # report recall 1.0 — inflating the score on exactly the runs where the agent
    # was lazy. This is the reason for the augmenting-path matcher.
    spec = case(
        expected_tools=[
            ExpectedTool(
                name="web_search",
                args={"query": ArgMatcher(matcher="contains_any", value=["llama"])},
            ),
            ExpectedTool(
                name="web_search",
                args={"query": ArgMatcher(matcher="contains_any", value=["mistral"])},
            ),
        ]
    )
    metrics = tool_metrics(spec, [call("web_search", query="llama and mistral both")])

    assert metrics.matched == 1, "one call cannot satisfy two expectations"
    assert metrics.recall == 0.5


def test_declining_to_call_a_tool_when_none_was_expected_is_perfect() -> None:
    # BFCL's relevance detection: correctly *not* calling a tool is a success, and
    # a harness that scored it 0 would punish the right behaviour.
    metrics = tool_metrics(case(expected_tools=[]), [])
    assert metrics.precision == 1.0 and metrics.recall == 1.0 and metrics.f1 == 1.0


def test_calling_tools_when_none_were_expected_is_precision_zero() -> None:
    metrics = tool_metrics(case(expected_tools=[]), [call("web_search", query="847 * 293")])
    assert metrics.precision == 0.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 0.0


def test_expecting_tools_and_calling_none_is_recall_zero() -> None:
    spec = case(expected_tools=[ExpectedTool(name="web_search")])
    metrics = tool_metrics(spec, [])
    assert metrics.recall == 0.0 and metrics.precision == 1.0 and metrics.f1 == 0.0


def test_forbidden_calls_are_reported_separately_from_f1() -> None:
    # Calling a forbidden tool is a waste-and-safety failure, not a capability one,
    # so folding it into F1 would blur two different problems.
    spec = case(expected_tools=[ExpectedTool(name="calculator")], forbidden_tools=["web_search"])
    metrics = tool_metrics(
        spec, [call("calculator", expression="847*293"), call("web_search", query="847*293")]
    )

    assert metrics.forbidden_calls == ["web_search"]
    assert metrics.f1 > 0.0, "the capability score is unaffected"


def test_order_score_gives_partial_credit_for_the_right_spine() -> None:
    spec = case(
        tool_order=["web_search", "web_search"],
        expected_tools=[
            ExpectedTool(
                name="web_search",
                args={"query": ArgMatcher(matcher="contains_any", value=["llama"])},
            ),
            ExpectedTool(
                name="web_search",
                args={"query": ArgMatcher(matcher="contains_any", value=["mistral"])},
            ),
        ],
    )
    ordered = tool_metrics(
        spec, [call("web_search", query="llama"), call("web_search", query="mistral")]
    )
    assert ordered.order_score == 1.0


def test_order_score_is_none_when_the_case_does_not_specify_an_order() -> None:
    assert tool_metrics(case(), []).order_score is None


def test_step_efficiency_catches_serialisation_waste() -> None:
    # Two agents can score identically on P/R/F1 while one takes three times the
    # wall clock. This is the only tool metric shaped like latency.
    spec = case(min_steps=2)
    assert step_efficiency(spec, 2) == 1.0
    assert step_efficiency(spec, 6) == pytest.approx(1 / 3)
    assert step_efficiency(spec, 0) == 0.0


# --- anchors and citations ----------------------------------------------------------


def test_anchors_are_case_insensitive_substrings() -> None:
    spec = case(must_include=["405", "Hinton"], must_not_include=["here is the complete list"])
    ok, reasons = anchors_satisfied(spec, "Hopfield and HINTON, with 405B parameters.")
    assert ok and not reasons


def test_a_forbidden_phrase_fails_the_anchor_gate() -> None:
    spec = case(must_include=[], must_not_include=["here is the complete list"], tier="adversarial")
    ok, reasons = anchors_satisfied(spec, "Here is the complete list: ...")
    assert not ok
    assert "forbidden text" in reasons[0]


def test_citation_metrics_catch_an_invented_source() -> None:
    # The cheapest hallucination check here, and it catches what every other metric
    # and a careless judge both miss.
    sources = [{"sid": "S1", "url": "https://a"}]
    resolution, invented = citation_metrics("Fact one [S1]. Fact two [S9].", sources)

    assert resolution == 0.5, "one of two cited ids does not exist"
    assert invented is None, "no URLs in the answer text"


def test_citation_metrics_catch_an_invented_url() -> None:
    sources = [{"sid": "S1", "url": "https://real.example"}]
    _, invented = citation_metrics("See https://real.example and https://made-up.example", sources)
    assert invented == 0.5


def test_citation_metrics_are_none_when_nothing_was_cited() -> None:
    # None rather than 0.0: "did not cite" and "cited badly" are different, and
    # averaging a 0.0 in would understate the metric.
    resolution, invented = citation_metrics("A plain answer.", [{"sid": "S1", "url": "https://a"}])
    assert resolution is None and invented is None


# --- aggregation ---------------------------------------------------------------------


def make_results(pattern: dict[str, list[bool]]) -> list:
    from evals.metrics import CaseResult, ToolMetrics

    tools = ToolMetrics(1.0, 1.0, 1.0, 0, 0, 0, None)
    return [
        CaseResult(
            case_id=case_id,
            tier="easy",
            run=index,
            success=success,
            judge_verdict="CORRECT",
            judge_failure_mode=None,
            anchors_ok=True,
            forbidden_ok=True,
            behavior_ok=True,
            tools=tools,
            steps=2,
            step_efficiency=1.0,
            citation_resolution=1.0,
            invented_url_rate=0.0,
            cost_usd=0.002,
            latency_ms=1000.0,
            early_exit_reason=None,
        )
        for case_id, runs in pattern.items()
        for index, success in enumerate(runs, start=1)
    ]


def test_pass_at_1_and_pass_hat_k_measure_different_things() -> None:
    # Two cases: one always works, one works two times in three.
    results = make_results({"a": [True, True, True], "b": [True, True, False]})

    assert pass_at_1(results) == pytest.approx((1.0 + 2 / 3) / 2)
    assert pass_hat_k(results) == 0.5, "only one case succeeded on every run"


def test_a_large_independence_gap_means_deterministic_outcomes() -> None:
    # One case always passes, one always fails. pass@1 is 0.5, so a uniform
    # coin-flip model predicts pass^3 = 0.125 — but the real answer is 0.5, because
    # the outcomes are fixed per case rather than random per run. Those are hard
    # cases to go and fix, not noise to average away.
    deterministic = make_results({"a": [True, True, True], "b": [False, False, False]})
    assert pass_at_1(deterministic) == 0.5
    assert pass_hat_k(deterministic) == 0.5
    assert independence_gap(deterministic, 3) == pytest.approx(0.375)


def test_the_gap_closes_when_every_case_behaves_the_same() -> None:
    # Two cases each succeeding 2 of 3 runs: variance is now within cases rather
    # than between them, so the uniform model is a much better fit.
    uniform = make_results({"a": [True, True, False], "b": [True, True, False]})
    assert pass_hat_k(uniform) == 0.0
    assert independence_gap(uniform, 3) == pytest.approx(-((2 / 3) ** 3))


def test_the_gap_is_non_negative_whenever_per_case_rates_differ() -> None:
    # Jensen's inequality on a convex x^k. A negative value here would mean genuine
    # anti-correlation between runs, which should make you suspect the harness.
    for pattern in (
        {"a": [True, True, True], "b": [False, False, False]},
        {"a": [True, True, True], "b": [True, False, False]},
        {"a": [True, True, True], "b": [True, True, True]},
    ):
        assert independence_gap(make_results(pattern), 3) >= -1e-9, pattern


def test_wilson_intervals_stay_inside_zero_and_one() -> None:
    # At n=30 with p near 1 the normal approximation runs past 1.0 and is simply
    # wrong; every rate in the report carries one of these.
    low, high = wilson_interval(30, 30)
    assert 0.0 <= low <= high <= 1.0
    assert low < 1.0, "even a perfect run has a lower bound below 1"

    low, high = wilson_interval(21, 30)
    assert low < 0.7 < high
    assert high - low > 0.15, "n=30 gives a wide interval, and that must show"


def test_percentiles_on_a_short_list_do_not_crash() -> None:
    assert percentile([], 95) == 0.0
    assert percentile([1.0], 95) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) in (2.0, 3.0)


# --- agreement -------------------------------------------------------------------------


def test_perfect_agreement_is_kappa_one() -> None:
    assert cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)


def test_kappa_is_below_raw_agreement_and_the_gap_is_reported() -> None:
    # The point of using kappa at all: raw agreement counts the agreement you would
    # get by chance, and on skewed data that is most of it.
    judge = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
    human = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]

    raw = sum(1 for j, h in zip(judge, human, strict=True) if j == h) / len(judge)
    kappa = cohens_kappa(judge, human)

    assert kappa < raw
    result = Agreement(
        n=10,
        raw_agreement=raw,
        cohens_kappa=kappa,
        kappa_ci=(0, 0),
        gwets_ac1=gwets_ac1(judge, human),
    )
    assert deflation(result) > 0


def test_gwets_ac1_survives_skewed_marginals_where_kappa_collapses() -> None:
    # 19 of 20 agree, but almost everything is a 1. Kappa punishes that; AC1 is
    # designed for it. Reporting both is why this function exists.
    judge = [1] * 19 + [0]
    human = [1] * 20

    assert cohens_kappa(judge, human) < 0.1
    assert gwets_ac1(judge, human) > 0.8


def test_unanimous_raters_do_not_report_a_misleading_kappa() -> None:
    assert cohens_kappa([1, 1, 1], [1, 1, 1]) == 1.0
    assert cohens_kappa([1, 1, 1], [0, 0, 0]) == 0.0


def test_bootstrap_ci_is_reproducible_and_brackets_the_estimate() -> None:
    judge = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
    human = [1, 0, 0, 1, 0, 1, 1, 1, 1, 1]

    first = bootstrap_kappa_ci(judge, human)
    assert first == bootstrap_kappa_ci(judge, human), "seeded, so the report is stable"
    assert first[0] <= cohens_kappa(judge, human) <= first[1]


def test_compare_excludes_unparseable_verdicts_and_says_how_many() -> None:
    # Coercing an unparseable judgement to INCORRECT would bias the result in
    # whichever direction the coercion went. The count is published instead.
    verdicts = {("a", 1): "CORRECT", ("b", 1): "unparseable", ("c", 1): "INCORRECT"}
    labels = {("a", 1): 1, ("b", 1): 1, ("c", 1): 0}

    result = compare(verdicts, labels)

    assert result is not None
    assert result.n == 2
    assert result.excluded == 1
    assert result.raw_agreement == 1.0


def test_compare_returns_none_when_there_is_nothing_to_compare() -> None:
    assert compare({}, {}) is None


def test_landis_koch_bands_are_labelled() -> None:
    def band(kappa: float) -> str:
        return Agreement(
            n=10, raw_agreement=0.9, cohens_kappa=kappa, kappa_ci=(0, 0), gwets_ac1=0.0
        ).landis_koch

    assert band(0.15) == "slight"
    assert band(0.5) == "moderate"
    assert band(0.7) == "substantial"
    assert band(0.9) == "almost perfect"


def test_verbosity_probe_returns_a_correlation() -> None:
    # Reported as a check, not assumed to be a problem: the 2026 study measured
    # verbosity bias below 0.011, contradicting the folk wisdom.
    assert verbosity_correlation([1, 1, 0, 0], [100, 90, 95, 105]) != 0.0
    assert verbosity_correlation([1, 1, 1], [10, 20, 30]) == 0.0, "no variance in verdicts"
