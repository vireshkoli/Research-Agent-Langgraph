"""Regenerate evals/REPORT.md from the committed results. Pure function, no API calls.

    python -m evals.report

CI runs this and then `git diff --exit-code`. An empty diff proves mechanically that
every number in the report — and therefore every number quoted in the README —
derives from a committed artifact rather than from something typed by hand once and
never checked again.

Every rate carries a 95% Wilson interval. At n=30 those intervals are wide, and that
is the point: a rate without one at this sample size is the first thing an
interviewer asks about.
"""

import json
import sys
from pathlib import Path
from typing import Any

from evals.agreement import compare, deflation
from evals.metrics import (
    CaseResult,
    ToolMetrics,
    independence_gap,
    mean,
    pass_at_1,
    pass_hat_k,
    percentile,
    wilson_interval,
)
from evals.schema import load_human_labels

RESULTS_DIR = Path(__file__).parent / "results"
REPORT = Path(__file__).parent / "REPORT.md"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _ci(successes: float, total: int) -> str:
    if not total:
        return "—"
    low, high = wilson_interval(int(round(successes)), total)
    return f"[{low * 100:.0f}–{high * 100:.0f}%]"


def _usd(value: float) -> str:
    return f"${value:.5f}"


def load_variant(name: str) -> dict[str, Any] | None:
    path = RESULTS_DIR / f"{name}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def to_results(payload: dict[str, Any]) -> list[CaseResult]:
    results = []
    for item in payload["results"]:
        data = dict(item)
        data["tools"] = ToolMetrics(**item["tools"])
        results.append(CaseResult(**data))
    return results


def headline(results: list[CaseResult], k: int) -> list[str]:
    n_cases = len({r.case_id for r in results})
    successes = sum(r.success for r in results)

    p1 = pass_at_1(results)
    pk = pass_hat_k(results)
    all_k = sum(all(r.success for r in group) for group in _by_case(results).values())

    return [
        "| Metric | Value | 95% CI | What it means |",
        "|---|---|---|---|",
        f"| **pass@1** | **{_pct(p1)}** | {_ci(successes / k, n_cases)} | "
        "a single attempt succeeds |",
        f"| **pass^{k}** | **{_pct(pk)}** | {_ci(all_k, n_cases)} | "
        f"*all {k}* attempts succeed — the reliability number |",
        f"| (pass@1)^{k} | {_pct(p1**k)} | — | what pass^{k} would be if every case "
        "behaved identically |",
        f"| independence gap | {independence_gap(results, k):+.3f} | — | "
        "positive means outcomes are deterministic per case, not random per run |",
    ]


def tier_table(results: list[CaseResult]) -> list[str]:
    lines = [
        "| Tier | Cases | pass@1 | 95% CI | Mean steps | Mean cost |",
        "|---|---|---|---|---|---|",
    ]
    for tier in ("easy", "multi_hop", "adversarial"):
        subset = [r for r in results if r.tier == tier]
        if not subset:
            continue
        n_cases = len({r.case_id for r in subset})
        rate = pass_at_1(subset)
        lines.append(
            f"| {tier} | {n_cases} | {_pct(rate)} | "
            f"{_ci(rate * n_cases, n_cases)} | "
            f"{mean([float(r.steps) for r in subset]):.1f} | "
            f"{_usd(mean([r.cost_usd for r in subset]))} |"
        )
    return lines


def tool_table(results: list[CaseResult]) -> list[str]:
    ordered = [r for r in results if r.tools.order_score is not None]
    forbidden = sum(1 for r in results if r.tools.forbidden_calls)
    resolutions = [r.citation_resolution for r in results if r.citation_resolution is not None]
    invented = [r.invented_url_rate for r in results if r.invented_url_rate is not None]

    order = _pct(mean([r.tools.order_score or 0 for r in ordered])) if ordered else "—"

    return [
        "| Metric | Value | What it uniquely catches |",
        "|---|---|---|",
        f"| Tool precision | {_pct(mean([r.tools.precision for r in results]))} | "
        "redundant or off-plan calls |",
        f"| Tool recall | {_pct(mean([r.tools.recall for r in results]))} | "
        "skipped hops, or answering from memory without looking anything up |",
        f"| Tool F1 | {_pct(mean([r.tools.f1 for r in results]))} | "
        "nothing on its own — it hides P/R asymmetry, which is why both are above |",
        f"| Order score (LCS) | {order} | doing hop 2 before hop 1, which implies guessing |",
        f"| Step efficiency | {_pct(mean([r.step_efficiency for r in results]))} | "
        "serialisation waste: same F1, 3x the latency |",
        f"| Forbidden-tool rate | {_pct(forbidden / len(results)) if results else '—'} | "
        "relevance-detection failures, e.g. searching the web for arithmetic |",
        f"| Citation resolution | {_pct(mean(resolutions)) if resolutions else '—'} | "
        "**fabricated sources** — invisible to every metric above and to a careless judge |",
        f"| Invented-URL rate | {_pct(mean(invented)) if invented else '—'} | "
        "URLs in the answer that were never retrieved |",
    ]


def cost_table(results: list[CaseResult], payload: dict[str, Any]) -> list[str]:
    costs = [r.cost_usd for r in results]
    latencies = [r.latency_ms / 1000 for r in results]
    steps = [float(r.steps) for r in results]

    return [
        "| | Mean | p50 | p95 |",
        "|---|---|---|---|",
        f"| Cost per run | {_usd(mean(costs))} | {_usd(percentile(costs, 50))} | "
        f"{_usd(percentile(costs, 95))} |",
        f"| Latency | {mean(latencies):.1f}s | {percentile(latencies, 50):.1f}s | "
        f"{percentile(latencies, 95):.1f}s |",
        f"| Steps | {mean(steps):.1f} | {percentile(steps, 50):.0f} | "
        f"{percentile(steps, 95):.0f} |",
        "",
        f"Total for this run: agent ${payload['agent_cost_usd']:.4f} + "
        f"judge ${payload['judge_cost_usd']:.4f} = "
        f"**${payload['agent_cost_usd'] + payload['judge_cost_usd']:.4f}** "
        f"over {len(results)} runs in {payload['elapsed_s']}s.",
    ]


def failure_table(results: list[CaseResult]) -> list[str]:
    modes: dict[str, int] = {}
    for result in results:
        if not result.success and result.judge_failure_mode:
            modes[result.judge_failure_mode] = modes.get(result.judge_failure_mode, 0) + 1

    failures = sum(1 for r in results if not r.success)
    if not modes:
        return [f"{failures} failed runs; no judge failure modes recorded."]

    lines = [f"Of {failures} failed runs:", "", "| Failure mode | Count |", "|---|---|"]
    lines += [
        f"| {mode} | {count} |" for mode, count in sorted(modes.items(), key=lambda kv: -kv[1])
    ]
    return lines


def budget_table(results: list[CaseResult]) -> list[str]:
    reasons: dict[str, int] = {}
    for result in results:
        if result.early_exit_reason:
            reasons[result.early_exit_reason] = reasons.get(result.early_exit_reason, 0) + 1

    exited = sum(reasons.values())
    lines = [
        f"{exited} of {len(results)} runs stopped early. Every one still returned an "
        "answer — the graph has no path to END that skips synthesis, and `finalize` "
        "falls back to an LLM-free assembly from state when the budget is genuinely gone.",
    ]
    if reasons:
        lines += ["", "| Reason | Runs |", "|---|---|"]
        lines += [f"| {reason} | {count} |" for reason, count in sorted(reasons.items())]
    return lines


def agreement_section(results: list[CaseResult]) -> list[str]:
    labels = load_human_labels()
    if not labels:
        return [
            "_No human labels committed yet, so judge-vs-human agreement is not "
            "reported. Until it is, treat every judge-derived number above as "
            "unvalidated._"
        ]

    verdicts = {(r.case_id, r.run): r.judge_verdict for r in results if r.judge_verdict}
    result = compare(verdicts, labels)
    if result is None:
        return ["_No overlapping labels and verdicts to compare._"]

    low, high = result.kappa_ci
    return [
        f"Hand-labelled **{result.n}** judgements before looking at any judge output.",
        "",
        "| Statistic | Value |",
        "|---|---|",
        f"| Raw agreement | {_pct(result.raw_agreement)} |",
        f"| **Cohen's κ** | **{result.cohens_kappa:.3f}** ({result.landis_koch}) |",
        f"| κ 95% CI (bootstrap, seeded) | [{low:.3f}, {high:.3f}] |",
        f"| Gwet's AC1 | {result.gwets_ac1:.3f} |",
        f"| Deflation, raw − κ | {deflation(result):.1f} pts |",
        f"| Coverage | {_pct(result.coverage)} ({result.excluded} excluded) |",
        "",
        "| | Human CORRECT | Human INCORRECT |",
        "|---|---|---|",
        f"| **Judge CORRECT** | {result.confusion['both_correct']} | "
        f"{result.confusion['judge_correct_human_incorrect']} |",
        f"| **Judge INCORRECT** | {result.confusion['judge_incorrect_human_correct']} | "
        f"{result.confusion['both_incorrect']} |",
        "",
        f"Raw agreement overstates κ by {deflation(result):.0f} points here. That gap is "
        "why raw agreement alone is not reported: it counts the agreement you would get "
        "by chance, which on skewed data is most of it. Gwet's AC1 is shown alongside "
        "because κ is unstable when one label dominates the marginals.",
    ]


def _matched(
    main: list[CaseResult], other: list[CaseResult]
) -> tuple[list[CaseResult], list[CaseResult]]:
    """Restrict both variants to the (case, run) pairs present in each.

    A variant that is missing runs — because a key rotation killed part of it, say —
    cannot be compared against a complete one on headline rates: whichever tiers it
    is missing decide the difference. Comparing only where both have data keeps the
    contrast about the ablation rather than about which cases survived.
    """
    keys = {(r.case_id, r.run) for r in main} & {(r.case_id, r.run) for r in other}
    return (
        [r for r in main if (r.case_id, r.run) in keys],
        [r for r in other if (r.case_id, r.run) in keys],
    )


def _variant_rows(label: str, main: list[CaseResult], other: list[CaseResult]) -> list[str]:
    left, right = _matched(main, other)
    if not left:
        return [f"_No runs shared with `{label}` to compare against._", ""]

    tiers = ", ".join(sorted({r.tier for r in left}))
    return [
        f"Compared on the **{len(left)} runs both variants completed** ({tiers}).",
        "",
        "| Variant | pass@1 | Mean steps | Mean cost |",
        "|---|---|---|---|",
        f"| {label} | {_pct(pass_at_1(right))} | "
        f"{mean([float(r.steps) for r in right]):.1f} | "
        f"{_usd(mean([r.cost_usd for r in right]))} |",
        f"| **full agent** | **{_pct(pass_at_1(left))}** | "
        f"{mean([float(r.steps) for r in left]):.1f} | "
        f"{_usd(mean([r.cost_usd for r in left]))} |",
        "",
    ]


def comparison_section(main: list[CaseResult], k: int) -> list[str]:
    lines: list[str] = []

    if baseline := load_variant("baseline"):
        lines += ["### Against a single round of search", ""]
        lines += _variant_rows(
            "baseline (one round of parallel search, then answer)",
            main,
            to_results(baseline),
        )

    if no_overrule := load_variant("no_overrule"):
        lines += ["### Does letting reflect overrule a stop earn its cost?", ""]
        if excluded := no_overrule.get("excluded_runs"):
            cases = ", ".join(no_overrule.get("excluded_cases", []))
            lines += [
                f"> {excluded} runs of this variant were discarded: "
                f"{no_overrule['excluded_reason']} Affected cases: {cases}. The "
                "comparison below is therefore restricted to the runs that survived, "
                "and does not cover the adversarial tier.",
                "",
            ]
        lines += _variant_rows(
            "no_overrule (act's decision to stop is final)", main, to_results(no_overrule)
        )

    return lines or ["_Ablations not run yet._"]


def _by_case(results: list[CaseResult]) -> dict[str, list[CaseResult]]:
    grouped: dict[str, list[CaseResult]] = {}
    for result in results:
        grouped.setdefault(result.case_id, []).append(result)
    return grouped


def main() -> int:
    payload = load_variant("full")
    if payload is None:
        print("no evals/results/full.json — run `python -m evals.run` first", file=sys.stderr)
        return 1

    results = to_results(payload)
    k = payload["runs_per_case"]
    n_cases = len({r.case_id for r in results})

    lines = [
        "# Evaluation report",
        "",
        f"{n_cases} cases x {k} independent runs = {len(results)} trajectories, "
        f"agent `{payload['agent_model']}`, judge "
        f"`{payload['judge_model'] or 'none'}` x{payload['judge_samples']}, "
        f"cassettes `{payload['cache']}`.",
        "",
        "Every per-run trace is committed under `evals/results/traces/`, so each number "
        "below can be traced to the run that produced it. Regenerate with "
        "`make report`; CI asserts the result is byte-identical to what is committed.",
        "",
        "## Headline",
        "",
        *headline(results, k),
        "",
        "## By tier",
        "",
        *tier_table(results),
        "",
        "## Tool-call accuracy",
        "",
        *tool_table(results),
        "",
        "## Cost and latency",
        "",
        *cost_table(results, payload),
        "",
        "## Budget behaviour",
        "",
        *budget_table(results),
        "",
        "## Where it fails",
        "",
        *failure_table(results),
        "",
        "## Is the judge any good?",
        "",
        *agreement_section(results),
        "",
        "## Against a baseline",
        "",
        *comparison_section(results, k),
        "",
        "## Reading these numbers honestly",
        "",
        f"- **n={n_cases} is small.** Every confidence interval above is wide, and any "
        "difference smaller than about 15 points is not distinguishable from noise.",
        f"- **pass^{k} from {k} runs is a one-bit measurement per case** — an "
        f"all-{k}-succeed rate, not a precise reliability estimate.",
        "- **Self-preference is not controlled.** Judge and agent are both OpenAI "
        "models; they are different generations, not different families. A judge from "
        "another provider would be needed to rule it out.",
        "- **Re-running produces different numbers.** The agent is non-deterministic "
        "and the web changes underneath it. The committed traces are the evidence, not "
        "a promise of reproducibility.",
        "",
        "_Generated from `evals/results/` by `python -m evals.report`._",
    ]

    REPORT.write_text("\n".join(lines) + "\n")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
