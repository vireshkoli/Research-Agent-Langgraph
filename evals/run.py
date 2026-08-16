"""The evaluation runner.

    python -m evals.run --dry-run                      validate everything, spend $0
    python -m evals.run --limit 8 --runs 1             shake the harness out cheaply
    python -m evals.run --variant full --runs 3 --no-cache --batch      the real thing

`--dry-run` is not a formality. It validates every case against the schema, checks
that referenced tools exist, prints the tier breakdown and estimates the spend — so
the answer to "what is this about to cost" arrives before the money does.

Two flags exist to protect the numbers rather than the wallet:

`--no-cache` forces cassettes off for the official run. Replayed runs are not
independent, and `pass^k` over non-independent runs is a meaningless statistic.

`--no-judge` runs the deterministic metrics only. The `no_overrule` ablation uses it,
because its headline number — how often reflect overruled a premature stop — is read
straight off the trace and needs no judge at all.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evals.judge import judge
from evals.metrics import (
    CaseResult,
    ToolMetrics,
    citation_metrics,
    is_success,
    step_efficiency,
    tool_metrics,
)
from evals.schema import EvalCase, load_cases, tier_counts
from research_agent import spend
from research_agent.agent import run as run_agent
from research_agent.config import Settings, settings
from research_agent.llm import CostTracker, cost_usd
from research_agent.trace import RunTrace, write_trace

RESULTS_DIR = Path(__file__).parent / "results"

# Measured from real runs during development, used only for the pre-flight estimate.
TOKENS_PER_RUN = (11_000, 1_000)  # input, output
JUDGE_TOKENS = (900, 250)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="evals.run", description=__doc__)
    parser.add_argument("--variant", choices=["full", "baseline", "no_overrule"], default="full")
    parser.add_argument("--runs", type=int, default=3, help="Independent runs per case (k).")
    parser.add_argument("--limit", type=int, help="Only the first N cases.")
    parser.add_argument("--case", action="append", help="Run specific case ids.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and estimate; spend $0.")
    parser.add_argument("--no-cache", action="store_true", help="Force cassettes off.")
    parser.add_argument("--no-judge", action="store_true", help="Deterministic metrics only.")
    parser.add_argument(
        "--batch", action="store_true", help="Judge via the Batch API (half price)."
    )
    parser.add_argument("--judge-samples", type=int, default=1)
    parser.add_argument("--model", help="Override the agent model.")
    parser.add_argument(
        "--out", type=Path, help="Results file (defaults to results/<variant>.json)."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip runs already in the checkpoint."
    )
    return parser.parse_args(argv)


def checkpoint_path(variant: str) -> Path:
    return RESULTS_DIR / f".{variant}.partial.jsonl"


def load_checkpoint(variant: str) -> list[dict[str, Any]]:
    """Scored runs recovered from an interrupted invocation.

    One line per run, appended and flushed as it completes, so a crash costs at most
    the run in flight. This exists because a transient DNS failure on a judge call
    once discarded four completed runs and twenty minutes of work: a 90-run job that
    cannot resume is a bet on the network staying up for half an hour.
    """
    path = checkpoint_path(variant)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # torn final line from a hard kill; that run is simply redone
    return rows


def select(cases: list[EvalCase], args: argparse.Namespace) -> list[EvalCase]:
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case.id in wanted]
    if args.limit:
        cases = cases[: args.limit]
    return cases


def estimate_usd(cases: list[EvalCase], args: argparse.Namespace, cfg: Settings) -> float:
    """Pre-flight spend estimate, from token counts measured during development."""
    runs = len(cases) * args.runs
    agent = runs * cost_usd(cfg.agent_model, TOKENS_PER_RUN[0], 0, TOKENS_PER_RUN[1])
    if args.no_judge:
        return agent
    judged = runs * args.judge_samples
    per_judge = cost_usd(cfg.judge_model, JUDGE_TOKENS[0], 0, JUDGE_TOKENS[1], batch=args.batch)
    return agent + judged * per_judge


def score(
    case: EvalCase,
    trace: RunTrace,
    run_index: int,
    args: argparse.Namespace,
    tracker: CostTracker,
) -> CaseResult:
    actual_calls = [call for step in trace.steps for call in (step.get("tool_calls") or [])]

    verdict: str | None = None
    failure_mode: str | None = None
    if not args.no_judge:
        verdict, failure_mode, _ = judge(
            case.question,
            case.reference_answer,
            trace.answer,
            case.judge_rubric_notes,
            samples=args.judge_samples,
            tracker=tracker,
        )

    success, reasons, anchors_ok, forbidden_ok, behavior_ok = is_success(case, trace, verdict)
    resolution, invented = citation_metrics(trace.answer, trace.sources)

    return CaseResult(
        case_id=case.id,
        tier=case.tier,
        run=run_index,
        success=success,
        judge_verdict=verdict,
        judge_failure_mode=failure_mode,
        anchors_ok=anchors_ok,
        forbidden_ok=forbidden_ok,
        behavior_ok=behavior_ok,
        tools=tool_metrics(case, actual_calls),
        steps=trace.usage["steps"],
        step_efficiency=step_efficiency(case, trace.usage["steps"]),
        citation_resolution=resolution,
        invented_url_rate=invented,
        cost_usd=trace.usage["totals"]["cost_usd"],
        latency_ms=trace.timings_ms["total"],
        early_exit_reason=trace.outcome.early_exit_reason,
        reasons=reasons,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = settings()
    if args.model:
        cfg = cfg.model_copy(update={"agent_model": args.model})

    cases = select(load_cases(), args)
    if not cases:
        print("no cases selected")
        return 1

    print(
        f"variant={args.variant}  cases={len(cases)}  runs={args.runs}  "
        f"judge={'off' if args.no_judge else f'{args.judge_samples}x {cfg.judge_model}'}"
    )
    print(f"tiers: {tier_counts(cases)}")
    print(f"agent model: {cfg.agent_model}")
    print(f"estimated spend: ${estimate_usd(cases, args, cfg):.2f}")

    if args.dry_run:
        print("\ndry run: every case validated against the schema, nothing spent.")
        print(f"{spend.summary()}")
        return 0

    if args.no_cache:
        import os

        os.environ["RA_LLM_CACHE"] = "off"
        print("cassettes: OFF (runs must be independent for pass^k to mean anything)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    traces_dir = RESULTS_DIR / "traces" / args.variant
    traces_dir.mkdir(parents=True, exist_ok=True)

    judge_tracker = CostTracker(budget_usd=10.0)
    started = time.perf_counter()

    recovered = load_checkpoint(args.variant) if args.resume else []
    done = {(row["case_id"], row["run"]) for row in recovered}
    if recovered:
        print(f"resuming: {len(recovered)} runs recovered from the checkpoint")
    else:
        checkpoint_path(args.variant).unlink(missing_ok=True)

    serialised: list[dict[str, Any]] = list(recovered)
    checkpoint = checkpoint_path(args.variant).open("a", encoding="utf-8")

    try:
        for index, case in enumerate(cases, start=1):
            for run_index in range(1, args.runs + 1):
                if (case.id, run_index) in done:
                    continue

                trace = run_agent(case.question, variant=args.variant, cfg=cfg)
                write_trace(trace, traces_dir / f"{case.id}-r{run_index}.json")
                result = score(case, trace, run_index, args, judge_tracker)

                row = _serialise(result)
                serialised.append(row)
                # Flushed immediately: an interrupted run must cost at most the one
                # in flight, never the twenty minutes before it.
                checkpoint.write(json.dumps(row) + "\n")
                checkpoint.flush()

                flag = "ok " if result.success else "FAIL"
                print(
                    f"  [{index:>2}/{len(cases)}] {case.id} r{run_index} [{flag}] "
                    f"{result.steps} steps  ${result.cost_usd:.5f}  "
                    f"{result.latency_ms / 1000:.1f}s"
                    + (f"  ({result.reasons[0]})" if result.reasons else ""),
                    flush=True,
                )
    finally:
        checkpoint.close()

    results = [_deserialise(row) for row in serialised]

    payload: dict[str, Any] = {
        "variant": args.variant,
        "runs_per_case": args.runs,
        "agent_model": cfg.agent_model,
        "judge_model": None if args.no_judge else cfg.judge_model,
        "judge_samples": 0 if args.no_judge else args.judge_samples,
        "cache": "off" if args.no_cache else "default",
        "elapsed_s": round(time.perf_counter() - started, 1),
        "agent_cost_usd": round(sum(r.cost_usd for r in results), 6),
        "judge_cost_usd": round(judge_tracker.total_cost_usd, 6),
        "results": serialised,
    }

    out = args.out or RESULTS_DIR / f"{args.variant}.json"
    out.write_text(json.dumps(payload, indent=2))

    print(f"\n{len(results)} runs in {payload['elapsed_s']}s")
    print(f"agent ${payload['agent_cost_usd']:.4f} + judge ${payload['judge_cost_usd']:.4f}")
    print(f"written to {out}")
    print(spend.summary())
    return 0


def _serialise(result: CaseResult) -> dict[str, Any]:
    data = asdict(result)
    data["tools"] = asdict(result.tools)
    return data


def _deserialise(row: dict[str, Any]) -> CaseResult:
    data = dict(row)
    data["tools"] = ToolMetrics(**row["tools"])
    return CaseResult(**data)


if __name__ == "__main__":
    sys.exit(main())
