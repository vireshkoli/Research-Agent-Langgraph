"""The evaluation runner.

    python -m evals.run --dry-run                      validate everything, spend $0
    python -m evals.run --limit 8 --runs 1             shake the harness out cheaply
    python -m evals.run --variant full --runs 3 --no-cache             the real thing

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
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evals.benchmarks import frames
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
    parser.add_argument(
        "--dataset",
        choices=["home", "frames"],
        default="home",
        help="home: the 30 hand-written cases. frames: the pre-registered FRAMES sample.",
    )
    parser.add_argument(
        "--search-backend",
        choices=["tavily", "ddgs"],
        help="Pin the search engine. Required for a real run: see main().",
    )
    parser.add_argument("--runs", type=int, default=3, help="Independent runs per case (k).")
    parser.add_argument("--limit", type=int, help="Only the first N cases.")
    parser.add_argument("--case", action="append", help="Run specific case ids.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and estimate; spend $0.")
    parser.add_argument("--no-cache", action="store_true", help="Force cassettes off.")
    parser.add_argument("--no-judge", action="store_true", help="Deterministic metrics only.")
    parser.add_argument("--judge-samples", type=int, default=1)
    parser.add_argument("--model", help="Override the agent model.")
    parser.add_argument(
        "--out", type=Path, help="Results file (defaults to results/<variant>.json)."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip runs already in the checkpoint."
    )
    return parser.parse_args(argv)


# Where FRAMES' questions and gold answers are published. A benchmark run that can
# read these is measuring lookup, not research.
ANSWER_KEY_HOSTS = (
    "huggingface.co",
    "hf.co",
    "github.com",
    "githubusercontent.com",
    "kaggle.com",
    "paperswithcode.com",
)


def tavily_quota_problem() -> str | None:
    """Why a Tavily-pinned run should not start, or None if it may. Costs no credit."""
    import httpx

    from research_agent.llm import _load_env_file

    _load_env_file()  # the key lives in .env, which is otherwise loaded lazily
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return "TAVILY_API_KEY is not set"
    try:
        account = httpx.get(
            "https://api.tavily.com/usage",
            headers={"Authorization": f"Bearer {key}"},
            timeout=20,
        ).json()["account"]
    except Exception as exc:  # noqa: BLE001 — a failed check is reported, not fatal
        return f"could not read the Tavily quota ({type(exc).__name__}); check it by hand"
    used, limit = account.get("plan_usage") or 0, account.get("plan_limit")
    if limit is not None and used >= limit:
        return (
            f"Tavily's quota is exhausted ({used}/{limit} this month). Wait for it to "
            "reset, or pin --search-backend ddgs for every variant being compared."
        )
    return None


def results_root(dataset: str) -> Path:
    """Each dataset keeps its own results, traces and checkpoints, so a benchmark run
    can never overwrite the hand-written set's committed evidence."""
    return RESULTS_DIR if dataset == "home" else RESULTS_DIR / dataset


def checkpoint_path(variant: str, dataset: str = "home") -> Path:
    return results_root(dataset) / f".{variant}.partial.jsonl"


def load_checkpoint(variant: str, dataset: str = "home") -> list[dict[str, Any]]:
    """Scored runs recovered from an interrupted invocation.

    One line per run, appended and flushed as it completes, so a crash costs at most
    the run in flight. This exists because a transient DNS failure on a judge call
    once discarded four completed runs and twenty minutes of work: a 90-run job that
    cannot resume is a bet on the network staying up for half an hour.
    """
    path = checkpoint_path(variant, dataset)
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
    per_judge = cost_usd(cfg.judge_model, JUDGE_TOKENS[0], 0, JUDGE_TOKENS[1])
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
        gold_source_recall=frames.gold_source_recall(case, trace.sources),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = settings()
    if args.model:
        cfg = cfg.model_copy(update={"agent_model": args.model})

    cases = select(frames.load() if args.dataset == "frames" else load_cases(), args)
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

    # An unpinned run falls back from Tavily to DuckDuckGo when the quota runs out,
    # which is right for the demo and wrong here: it happened mid-sweep, so later
    # variants ran on a different search engine from earlier ones and every number
    # still looked plausible. A real evaluation must say which engine it used.
    if args.search_backend is None:
        print(
            "refusing to run: pin the search engine with --search-backend tavily|ddgs. "
            "Unpinned, a quota running out mid-run silently switches engines and "
            "confounds every comparison that crosses the switch.",
            file=sys.stderr,
        )
        return 2
    os.environ["RA_SEARCH_BACKEND"] = args.search_backend
    settings.cache_clear()
    cfg = (
        settings() if not args.model else settings().model_copy(update={"agent_model": args.model})
    )
    if args.search_backend == "tavily" and (reason := tavily_quota_problem()):
        print(f"refusing to run: {reason}", file=sys.stderr)
        return 2
    print(f"search engine: {args.search_backend} (pinned, no fallback)")
    if args.dataset == "frames":
        # The benchmark's own questions and answers are published on these hosts.
        os.environ["RA_SEARCH_EXCLUDE_DOMAINS"] = json.dumps(list(ANSWER_KEY_HOSTS))
        settings.cache_clear()
        cfg = (
            settings()
            if not args.model
            else settings().model_copy(update={"agent_model": args.model})
        )
        print(f"excluded hosts: {', '.join(ANSWER_KEY_HOSTS)}")

    if args.no_cache:
        os.environ["RA_LLM_CACHE"] = "off"
        print("cassettes: OFF (runs must be independent for pass^k to mean anything)")

    root = results_root(args.dataset)
    root.mkdir(parents=True, exist_ok=True)
    traces_dir = root / "traces" / args.variant
    traces_dir.mkdir(parents=True, exist_ok=True)

    judge_tracker = CostTracker(budget_usd=10.0)
    started = time.perf_counter()

    recovered = load_checkpoint(args.variant, args.dataset) if args.resume else []
    done = {(row["case_id"], row["run"]) for row in recovered}
    if recovered:
        print(f"resuming: {len(recovered)} runs recovered from the checkpoint")
    else:
        checkpoint_path(args.variant, args.dataset).unlink(missing_ok=True)

    serialised: list[dict[str, Any]] = list(recovered)
    checkpoint = checkpoint_path(args.variant, args.dataset).open("a", encoding="utf-8")

    try:
        for index, case in enumerate(cases, start=1):
            for run_index in range(1, args.runs + 1):
                if (case.id, run_index) in done:
                    continue

                trace = run_agent(case.question, variant=args.variant, cfg=cfg)
                write_trace(trace, traces_dir / f"{case.id}-r{run_index}.json")
                judged_before = judge_tracker.total_cost_usd
                result = score(case, trace, run_index, args, judge_tracker)
                result.judge_cost_usd = round(judge_tracker.total_cost_usd - judged_before, 8)

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
        "search_backend": args.search_backend,
        "dataset": args.dataset,
        "elapsed_s": round(time.perf_counter() - started, 1),
        "agent_cost_usd": round(sum(r.cost_usd for r in results), 6),
        # Summed over results rather than read off this process's tracker: after a
        # --resume the tracker has only seen the runs made since, and the total would
        # quietly leave out the rest.
        "judge_cost_usd": round(sum(r.judge_cost_usd for r in results), 6),
        "results": serialised,
    }

    out = args.out or results_root(args.dataset) / f"{args.variant}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")

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
