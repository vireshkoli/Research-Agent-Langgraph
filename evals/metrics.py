"""Trace-level metrics. Every definition here is one the README has to defend.

Two choices worth stating up front.

**Maximum bipartite matching, not greedy left-to-right.** Greedy over-counts
whenever one actual call could satisfy two expectations, which inflates recall on
exactly the cases where the agent was lazy. At these sizes a thirty-line augmenting
-path matcher is exact and pulls in no dependency.

**Undefined edge cases are where harnesses lie**, so all three are written down:
zero expected and zero actual is a perfect score (correct relevance detection); zero
expected with calls made is precision 0; expectations with no calls made is recall 0.

`pass^k` comes from tau2-bench and is the interesting number: `pass@1` says how often
a single attempt works, `pass^k` says how often *every* attempt works. Reporting both
alongside `(pass@1)^k` shows whether failures are independent or correlated, and
almost nobody publishes that comparison.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Any

from evals.matchers import call_matches
from evals.schema import EvalCase

CITATION = re.compile(r"\[(S\d+)\]")
URL = re.compile(r"https?://[^\s\)\]<>\"']+")


@dataclass
class ToolMetrics:
    precision: float
    recall: float
    f1: float
    matched: int
    n_actual: int
    n_expected: int
    order_score: float | None
    forbidden_calls: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    """Everything scored for one run of one case."""

    case_id: str
    tier: str
    run: int
    success: bool
    judge_verdict: str | None  # CORRECT | INCORRECT | unparseable | None (not judged)
    judge_failure_mode: str | None
    anchors_ok: bool
    forbidden_ok: bool
    behavior_ok: bool
    tools: ToolMetrics
    steps: int
    step_efficiency: float
    citation_resolution: float | None
    invented_url_rate: float | None
    cost_usd: float
    latency_ms: float
    early_exit_reason: str | None
    reasons: list[str] = field(default_factory=list)


# --- tool-call accuracy -----------------------------------------------------------


def _bipartite_matching(edges: dict[int, set[int]], n_expected: int) -> dict[int, int]:
    """Maximum matching expected -> actual, by augmenting paths (Kuhn's algorithm)."""
    pair_for_actual: dict[int, int] = {}

    def augment(expected: int, seen: set[int]) -> bool:
        for actual in edges.get(expected, ()):
            if actual in seen:
                continue
            seen.add(actual)
            if actual not in pair_for_actual or augment(pair_for_actual[actual], seen):
                pair_for_actual[actual] = expected
                return True
        return False

    for expected in range(n_expected):
        augment(expected, set())
    return pair_for_actual


def tool_metrics(case: EvalCase, actual_calls: list[dict[str, Any]]) -> ToolMetrics:
    expected = [tool for tool in case.expected_tools if tool.required]
    n_expected, n_actual = len(expected), len(actual_calls)

    edges = {
        i: {
            j
            for j, call in enumerate(actual_calls)
            if call_matches(call, spec.name, spec.args, case.strict_args)
        }
        for i, spec in enumerate(expected)
    }
    pairs = _bipartite_matching(edges, n_expected)
    matched = len(pairs)

    if n_expected == 0 and n_actual == 0:
        precision = recall = f1 = 1.0  # correctly declined to use any tool
    elif n_expected == 0:
        precision, recall, f1 = 0.0, 1.0, 0.0
    elif n_actual == 0:
        precision, recall, f1 = 1.0, 0.0, 0.0
    else:
        precision = matched / n_actual
        recall = matched / n_expected
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return ToolMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        matched=matched,
        n_actual=n_actual,
        n_expected=n_expected,
        order_score=_order_score(case, actual_calls, pairs),
        # Reported separately, never folded into F1: calling a forbidden tool is a
        # waste-and-safety failure, not a capability one.
        forbidden_calls=[
            call["name"] for call in actual_calls if call.get("name") in case.forbidden_tools
        ],
    )


def _order_score(
    case: EvalCase, actual_calls: list[dict[str, Any]], pairs: dict[int, int]
) -> float | None:
    """Longest correctly-ordered subsequence of the expected order, over its length.

    LCS rather than Kendall tau: LCS gives partial credit for getting the spine
    right and inserting an extra hop, which is the common near-miss. Kendall
    penalises every inversion equally and cannot tell those apart.
    """
    if not case.tool_order:
        return None

    sequence = [pairs[actual] for actual in sorted(pairs) if actual in pairs]
    target = list(range(len(case.tool_order)))
    if not target:
        return None
    return _lcs_length(sequence, target) / len(target)


def _lcs_length(left: list[int], right: list[int]) -> int:
    table = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i, a in enumerate(left, start=1):
        for j, b in enumerate(right, start=1):
            table[i][j] = (
                table[i - 1][j - 1] + 1 if a == b else max(table[i - 1][j], table[i][j - 1])
            )
    return table[-1][-1]


def step_efficiency(case: EvalCase, steps: int) -> float:
    """min_steps / actual, clipped.

    The only tool metric shaped like latency. An agent making one call per round
    over six rounds and one making three calls over two rounds score identically on
    precision, recall and F1 while costing three times the wall clock.
    """
    if steps <= 0:
        return 0.0
    return min(1.0, case.min_steps / steps)


# --- citation validity ------------------------------------------------------------


def citation_metrics(
    answer: str, sources: list[dict[str, Any]]
) -> tuple[float | None, float | None]:
    """(resolution, invented_url_rate).

    The cheapest hallucination check in the project, and it catches what every other
    metric and a careless judge both miss: a confident answer citing a source that
    does not exist.
    """
    known_ids = {source["sid"] for source in sources}
    known_urls = {source["url"] for source in sources}

    cited = CITATION.findall(answer)
    resolution = sum(1 for sid in cited if sid in known_ids) / len(cited) if cited else None

    urls = URL.findall(answer)
    invented = (
        sum(1 for url in urls if url.rstrip(".,);") not in known_urls) / len(urls) if urls else None
    )
    return resolution, invented


# --- success ----------------------------------------------------------------------


def anchors_satisfied(case: EvalCase, answer: str) -> tuple[bool, list[str]]:
    """Deterministic gate. Case-insensitive substring, no cleverness."""
    haystack = answer.lower()
    reasons: list[str] = []

    for needle in case.must_include:
        if needle.lower() not in haystack:
            reasons.append(f"missing required text {needle!r}")
    for needle in case.must_not_include:
        if needle.lower() in haystack:
            reasons.append(f"contains forbidden text {needle!r}")
    return not reasons, reasons


def behavior_satisfied(case: EvalCase, trace: Any) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    expected = case.expected_behavior

    answer = (trace.answer or "").strip()
    if expected.should_answer and not answer:
        reasons.append("produced no answer")

    if expected.min_answer_chars and len(answer) < expected.min_answer_chars:
        reasons.append(
            f"answer is {len(answer)} characters, expected at least "
            f"{expected.min_answer_chars} — consistent with parroting the prompt "
            "rather than reasoning about it"
        )

    if expected.must_cite and len(trace.citations) < expected.min_citations:
        reasons.append(
            f"cited {len(trace.citations)} sources, expected at least {expected.min_citations}"
        )

    reason = trace.outcome.early_exit_reason
    if expected.allowed_early_exit_reasons and reason not in expected.allowed_early_exit_reasons:
        reasons.append(f"early exit reason {reason!r} not in the allowed set")
    elif expected.should_early_exit and reason is None:
        reasons.append("expected an early exit, ran to completion")

    return not reasons, reasons


def is_success(
    case: EvalCase, trace: Any, judge_verdict: str | None
) -> tuple[bool, list[str], bool, bool, bool]:
    """All-or-nothing, tau2-bench style.

    A run succeeds only if the judge said CORRECT *and* every deterministic gate
    passed. Partial credit on a research answer would let a fabricated citation
    average away against a correct number, which is precisely the failure this eval
    exists to surface.
    """
    anchors_ok, anchor_reasons = anchors_satisfied(case, trace.answer or "")
    behavior_ok, behavior_reasons = behavior_satisfied(case, trace)

    called = {call["name"] for step in trace.steps for call in (step.get("tool_calls") or [])}
    forbidden_hit = sorted(called & set(case.forbidden_tools))
    forbidden_ok = not forbidden_hit

    reasons = anchor_reasons + behavior_reasons
    if forbidden_hit:
        reasons.append(f"called forbidden tool(s): {', '.join(forbidden_hit)}")
    if judge_verdict == "INCORRECT":
        reasons.append("judge said INCORRECT")

    judge_ok = judge_verdict in ("CORRECT", None)
    return (
        anchors_ok and behavior_ok and forbidden_ok and judge_ok,
        reasons,
        anchors_ok,
        forbidden_ok,
        behavior_ok,
    )


# --- aggregation ------------------------------------------------------------------


def pass_at_1(results: list[CaseResult]) -> float:
    """Mean over cases of the per-case success rate across its runs."""
    by_case = _group(results)
    if not by_case:
        return 0.0
    return sum(sum(r.success for r in runs) / len(runs) for runs in by_case.values()) / len(by_case)


def pass_hat_k(results: list[CaseResult]) -> float:
    """Fraction of cases where *every* run succeeded.

    From exactly k runs this is a one-bit measurement per case — an all-k-succeed
    rate, not a precise reliability estimate — and REPORT.md says so.
    """
    by_case = _group(results)
    if not by_case:
        return 0.0
    return sum(all(r.success for r in runs) for runs in by_case.values()) / len(by_case)


def independence_gap(results: list[CaseResult], k: int) -> float:
    """pass^k minus (pass@1)^k — how *bimodal* the outcomes are.

    `(pass@1)^k` is what pass^k would be if every case behaved identically, each run
    succeeding independently with probability pass@1. Real agents are not like that:
    they are reliably right on some cases and reliably wrong on others.

    Because x^k is convex, Jensen's inequality makes this quantity **>= 0 whenever
    per-case success rates differ at all**. So:

      large positive  outcomes are deterministic per case — a fixed set of cases
                      fails every time. Those are hard cases to go and fix, not
                      noise to average away.
      near zero       behaviour is close to uniform coin-flipping at pass@1, which
                      means run-to-run variance is the dominant failure mode.
      negative        would require genuine anti-correlation between runs; if this
                      ever appears, suspect the harness before believing it.

    An earlier version of this docstring had the sign backwards and called a
    negative gap the sign of correlated failure. It is the positive gap that carries
    the information, and the distinction is the reason for running k times at all.
    """
    return pass_hat_k(results) - pass_at_1(results) ** k


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% CI for a proportion.

    Wilson rather than normal-approximation: at n=30 with p near 0 or 1 the normal
    interval runs outside [0,1] and is simply wrong. Every rate in the report carries
    one of these, because a rate without an interval at this sample size is the thing
    an interviewer asks about first.
    """
    if total == 0:
        return 0.0, 0.0
    phat = successes / total
    denominator = 1 + z**2 / total
    centre = (phat + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(pct / 100 * (len(ordered) - 1)))))
    return ordered[index]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _group(results: list[CaseResult]) -> dict[str, list[CaseResult]]:
    grouped: dict[str, list[CaseResult]] = {}
    for result in results:
        grouped.setdefault(result.case_id, []).append(result)
    return grouped
