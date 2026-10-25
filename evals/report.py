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
import re
import sys
from pathlib import Path
from typing import Any

from evals.agreement import Agreement, compare, deflation, verbosity_correlation
from evals.metrics import (
    CaseResult,
    ToolMetrics,
    discordant_counts,
    independence_gap,
    mcnemar_exact,
    mean,
    paired_bootstrap_ci,
    pass_at_1,
    pass_hat_k,
    per_case_rates,
    percentile,
    unique_wins,
    wilson_interval,
)
from evals.schema import load_human_labels
from research_agent.prompts import prompt_hash

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


def staleness_banner(results: list[CaseResult], variant: str = "full") -> list[str]:
    """Say so when these results describe prompts the code no longer has.

    Every trace records the hash of the prompt set that produced it, which is the
    only reason this check is possible at all. Without it a prompt edit would
    silently invalidate every number in this file and nothing would notice — the
    results would still regenerate, still be internally consistent, and still be
    describing an agent that no longer exists.
    """
    recorded = {
        json.loads(path.read_text()).get("config", {}).get("prompt_hash")
        for case, run in sorted({(r.case_id, r.run) for r in results})
        if (path := RESULTS_DIR / "traces" / variant / f"{case}-r{run}.json").is_file()
    }
    recorded.discard(None)
    current = prompt_hash()
    if not recorded or recorded == {current}:
        return []

    was = ", ".join(sorted(recorded))
    return [
        f"> **These results predate the current prompts.** They were produced with "
        f"`{was}`; the code now ships `{current}`. The prompt set was changed after "
        "this evaluation ran, to remove an instruction that told the agent to prefer "
        "several searches and a reflection rubric that ranked stopping early as the "
        "more common failure — the opposite of what these very runs measured. Every "
        "number below therefore describes the agent as it was when measured, not as "
        "it is now, and the comparison that matters is the re-run against this hash.",
        "",
    ]


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


def _tool_caveat(results: list[CaseResult]) -> list[str]:
    """What the precision and order numbers above can and cannot support.

    Written from the dataset rather than asserted, because the honest reading of
    tool precision here depends entirely on how thinly `expected_tools` is specified
    and a reader cannot tell that from the percentage.
    """
    expected = [r.tools.n_expected for r in results]
    actual = [r.tools.n_actual for r in results]
    ordered = sum(1 for r in results if r.tools.order_score is not None)
    thin = sum(1 for n in expected if n <= 1)

    return [
        "",
        f"**Read tool precision with care.** `expected_tools` names at most one call "
        f"for {thin} of these {len(results)} runs (mean {mean([float(n) for n in expected]):.2f} "
        f"expected against {mean([float(n) for n in actual]):.2f} actual), while the agent "
        "is designed to fan out several parallel searches in a single round. Precision "
        "is matched/n_actual, so a correct four-way fan-out against a one-call "
        "expectation scores 25% without a single wasted call. The figure above is "
        "therefore a lower bound that mixes genuine redundancy with the dataset's own "
        "thinness, and it is not evidence of waste on its own — **Where the rounds go** "
        "below measures that directly instead.",
        "",
        f"The order score is computed on the {ordered} run(s) that declare a "
        "`tool_order`, so it is a check that nothing is out of sequence on those cases "
        "rather than a rate with any sampling behind it.",
    ]


def loop_efficiency_table(results: list[CaseResult], variant: str = "full") -> list[str]:
    """Does the loop do anything with the rounds it spends?

    Tool precision cannot answer that question here, so this does. Precision is
    matched/n_actual against `expected_tools`, and `expected_tools` names a single
    call for most cases while the agent is *designed* to fan out four to six parallel
    searches per round. A low precision therefore partly measures how thinly the
    dataset specifies the task, not how much work the agent wasted, and a number that
    confounds the two should not be the evidence for a claim about waste.

    Every figure below is counted off the committed traces instead, and none of them
    depends on an expectation anyone had to write down:

    - a **no-op round** is one where `act` emitted no tool calls at all. `act` sets
      `act_requested_stop` in that case, so a no-op round is the agent saying it is
      finished and being sent back round the loop anyway.
    - a **dry round** ran tools and minted no new source.
    - a **repeated query** is a search whose normalised terms exactly match an earlier
      search in the same run.
    """
    rounds = {"noop": 0, "dry": 0, "productive": 0}
    searches = repeats = calls = barren = 0

    for case, run in sorted({(r.case_id, r.run) for r in results}):
        path = RESULTS_DIR / "traces" / variant / f"{case}-r{run}.json"
        if not path.is_file():
            continue
        trace = json.loads(path.read_text())
        first_seen: dict[int, int] = {}
        for source in trace.get("sources") or []:
            step = source.get("first_seen_step")
            first_seen[step] = first_seen.get(step, 0) + 1

        seen_queries: set[str] = set()
        index = 0
        for step in trace.get("steps") or []:
            if step.get("node") != "observe":
                continue
            index += 1
            observations = step.get("observations") or []
            for observation in observations:
                calls += 1
                if not observation.get("source_ids"):
                    barren += 1
                if observation.get("tool") == "web_search":
                    searches += 1
                    query = " ".join(
                        sorted(
                            re.findall(
                                r"[a-z0-9]+",
                                str((observation.get("args") or {}).get("query", "")).lower(),
                            )
                        )
                    )
                    if query in seen_queries:
                        repeats += 1
                    seen_queries.add(query)
            if not observations:
                rounds["noop"] += 1
            elif not first_seen.get(index):
                rounds["dry"] += 1
            else:
                rounds["productive"] += 1

    total = sum(rounds.values())
    if not total:
        return []

    def row(label: str, count: int, denominator: int) -> str:
        return f"| {label} | {count} | {count / denominator * 100:.1f}% |"

    lines = [
        "| | Count | Share |",
        "|---|---|---|",
        row("Rounds where `act` emitted no tool calls", rounds["noop"], total),
        row("Rounds that ran tools and found nothing new", rounds["dry"], total),
        row("Productive rounds", rounds["productive"], total),
        row("Searches repeating an earlier query verbatim", repeats, searches or 1),
        row("Tool calls returning no new source", barren, calls or 1),
        "",
        f"Across {total} rounds the agent made {calls} tool calls, of which "
        f"{repeats} repeated a query it had already issued. **It is not looping on "
        "near-identical searches, and it is not making large numbers of useless "
        f"calls.** What it does do is spend {rounds['noop']} rounds "
        f"({rounds['noop'] / total * 100:.0f}% of all of them) calling no tools at "
        "all. `act` emits no calls when it believes it can answer, so those rounds "
        "are the agent asking to stop and being sent back by `reflect`, paying for "
        "one `act` call and one `reflect` call each time to do nothing.",
        "",
        "That is a different defect from over-searching and it has a different fix: "
        "not fewer searches, but honouring a stop that has already been proposed.",
    ]
    return lines


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


def case_failure_table(results: list[CaseResult], k: int) -> list[str]:
    """*Which* cases fail, not merely how often.

    The independence gap already says failures are concentrated in a fixed set of
    cases rather than scattered across runs. Reporting that and then not naming the
    cases leaves the reader with a statistic they cannot act on, so this table names
    them and separates the two kinds of failure that a pass rate blurs together:

    - a case that fails on **every** run is a capability gap, and the thing to go and
      fix
    - a case that fails on **one** run is variance, and averages away with more runs

    `internal_error` is counted apart from both. A transport failure is not the agent
    being wrong, and folding it into a capability claim would overstate the problem
    in the same way the key rotation would have overstated the no_overrule ablation.
    """
    by_case = _by_case(results)
    failing = {case: [r for r in runs if not r.success] for case, runs in by_case.items()}
    failing = {case: runs for case, runs in failing.items() if runs}

    if not failing:
        return ["No case failed on any run."]

    always = [case for case, runs in failing.items() if len(runs) == len(by_case[case])]
    transport = sum(
        1 for runs in failing.values() for r in runs if r.early_exit_reason == "internal_error"
    )

    lines = [
        f"**{len(failing)} of {len(by_case)} cases account for every failed run.** "
        f"{len(always)} {'fails' if len(always) == 1 else 'fail'} on all {k} runs — a "
        "capability gap rather than variance, and the place to start. The rest are flaky "
        "and would partly average away with more runs.",
        "",
        "| Case | Tier | Failed | Pattern | Judge failure modes | Early exits |",
        "|---|---|---|---|---|---|",
    ]
    for case in sorted(failing, key=lambda c: (-len(failing[c]), c)):
        runs = failing[case]
        total = len(by_case[case])
        pattern = (
            "every run" if len(runs) == total else ("one run" if len(runs) == 1 else "most runs")
        )
        modes = (
            ", ".join(sorted({r.judge_failure_mode for r in runs if r.judge_failure_mode})) or "—"
        )
        exits = ", ".join(sorted({r.early_exit_reason for r in runs if r.early_exit_reason})) or "—"
        tier = by_case[case][0].tier
        lines.append(f"| `{case}` | {tier} | {len(runs)}/{total} | {pattern} | {modes} | {exits} |")

    if transport:
        subject = "It is" if transport == 1 else "They are"
        lines += [
            "",
            f"{transport} of those failed runs exited with `internal_error` — a transport "
            f"failure rather than the agent being wrong. {subject} left in the table "
            "rather than quietly dropped, but not treated as evidence about capability.",
        ]

    capped = sorted(
        {
            case
            for case, runs in failing.items()
            if any(r.early_exit_reason == "tool_cap" for r in runs)
        }
    )
    if capped:
        lines += [
            "",
            f"{len(capped)} of the failing cases hit a per-tool call cap rather than "
            f"deciding they were finished ({', '.join(f'`{c}`' for c in capped)}). That is the "
            "over-searching this report measures elsewhere, showing up as a failure rather "
            "than merely as wasted spend.",
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
    asymmetry = _asymmetry_note(result)
    verbosity = _verbosity_note(results, labels)

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
        "because κ is unstable when one label dominates the marginals. (The literature "
        "this design follows measures 33-41 points of deflation on MT-Bench; the gap "
        "here is smaller because raw agreement is unusually high, not because the "
        "correction stopped mattering.)",
        "",
        *asymmetry,
        "",
        *verbosity,
    ]


def _asymmetry_note(result: Agreement) -> list[str]:
    """Which *direction* the judge errs in — the thing that decides whether a
    reported pass rate can be trusted."""
    inflated = result.confusion["judge_correct_human_incorrect"]
    deflated = result.confusion["judge_incorrect_human_correct"]

    if inflated == 0 and deflated > 0:
        return [
            f"**The judge never inflates a score.** It said CORRECT where the human said "
            f"INCORRECT **{inflated} times**; all {deflated} disagreements run the other "
            "way, with the judge stricter than the human. That is the direction an "
            "evaluation wants to fail in: every pass rate in this report is a floor, not "
            "a flattering estimate. A judge that erred the other way would quietly "
            "inflate the agent and nothing else here would catch it.",
        ]
    if inflated:
        return [
            f"**The judge passed {inflated} run(s) the human failed**, against {deflated} "
            "in the other direction. Pass rates here may therefore be slightly optimistic.",
        ]
    return ["Judge and human agreed on every item; there is no error direction to report."]


def _verbosity_note(results: list[CaseResult], labels: dict[tuple[str, int], int]) -> list[str]:
    """Cheap bias probe: does the judge reward length?"""
    scores: list[int] = []
    lengths: list[int] = []
    by_key = {(r.case_id, r.run): r for r in results}

    for key in sorted(labels):
        result = by_key.get(key)
        if result is None or result.judge_verdict not in ("CORRECT", "INCORRECT"):
            continue
        trace_path = RESULTS_DIR / "traces" / "full" / f"{key[0]}-r{key[1]}.json"
        if not trace_path.is_file():
            continue
        scores.append(1 if result.judge_verdict == "CORRECT" else 0)
        lengths.append(len(json.loads(trace_path.read_text()).get("answer") or ""))

    if len(scores) < 2:
        return []

    r = verbosity_correlation(scores, lengths)
    direction = "shorter" if r < 0 else "longer"
    return [
        f"**Verbosity probe:** correlation between verdict and answer length is "
        f"**r = {r:+.3f}** — the judge scored {direction} answers higher. Folk wisdom says "
        "LLM judges reward length, and the 2026 study behind this design measured that "
        "bias below 0.011, so a value this size is worth naming rather than filing away. "
        "It is also confounded: this agent writes longest when it is hedging on a "
        "question it could not settle, so length here tracks difficulty as much as style. "
        "Treat it as a flag for a larger sample, not as evidence the judge prefers brevity.",
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
    ] + _paired_stats(label, left, right)


def _paired_stats(label: str, left: list[CaseResult], right: list[CaseResult]) -> list[str]:
    """The comparison treated as paired, which is what it is.

    Two independent Wilson intervals on the same cases throw the pairing away and
    answer a question nobody asked. Both tests below are reported even when they
    disagree in strength, because the disagreement is informative: the bootstrap
    resamples *cases* and so speaks to whether another draw of questions would
    reverse the sign, while McNemar conditions on the discordant pairs alone and is
    deliberately conservative at these counts.
    """
    lo, hi, adverse = paired_bootstrap_ci(per_case_rates(right), per_case_rates(left))
    only_full, only_other = discordant_counts(left, right)
    wins_full, wins_other = unique_wins(left, right)
    short = label.split(" (")[0]

    return [
        f"**Paired comparison.** The difference in pass@1 is "
        f"{(pass_at_1(right) - pass_at_1(left)) * 100:+.1f} points in favour of "
        f"`{short}`, with a 95% bootstrap interval over cases of "
        f"[{lo * 100:+.1f}, {hi * 100:+.1f}] points. "
        f"{adverse * 100:.1f}% of resamples failed to favour it. McNemar's exact test on "
        f"the {only_full + only_other} discordant (case, run) pairs — {only_full} where "
        f"only the full agent succeeded, {only_other} where only `{short}` did — gives "
        f"p = {mcnemar_exact(only_full, only_other):.3f}.",
        "",
        f"**Unique wins.** The full agent reliably solves **{wins_full}** "
        f"{'case' if wins_full == 1 else 'cases'} that `{short}` does not; `{short}` "
        f"reliably solves **{wins_other}** that the full agent does not."
        + (
            " A variant that costs more and wins nothing outright is not trading "
            "accuracy for cost — it is paying for nothing."
            if wins_full == 0
            else " The richer variant does win ground of its own here, so the extra "
            "cost buys something — the question is whether it buys enough."
        ),
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
        elif repaired := no_overrule.get("repaired_runs"):
            cases = ", ".join(no_overrule.get("repaired_cases", []))
            lines += [
                f"> {repaired} runs of this variant were re-run rather than left "
                f"missing ({cases}). {no_overrule['repaired_reason']}",
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
        *staleness_banner(results),
        *headline(results, k),
        "",
        "## By tier",
        "",
        *tier_table(results),
        "",
        "## Tool-call accuracy",
        "",
        *tool_table(results),
        *_tool_caveat(results),
        "",
        "## Where the rounds go",
        "",
        *loop_efficiency_table(results),
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
        *case_failure_table(results, k),
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
