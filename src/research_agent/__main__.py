"""CLI: `uv run python -m research_agent "<question>"`

Prints the cited answer, a per-step table, and what the run cost. Every budget is
overridable from here, which is how the graceful-degradation path is demonstrated
without waiting for a real breach:

    python -m research_agent --max-steps 2 --max-usd 0.002 "<a hard question>"

That exits 0 with a partial answer and an early_exit_reason, not a traceback.
"""

import argparse
import sys
from pathlib import Path

from research_agent import spend
from research_agent.agent import run
from research_agent.config import Settings, settings
from research_agent.trace import RunTrace


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="research_agent", description=__doc__)
    parser.add_argument("question", help="The research question to answer.")
    parser.add_argument("--variant", choices=["full", "baseline", "no_reflect"], default="full")
    parser.add_argument("--json", type=Path, metavar="PATH", help="Write the full trace here.")
    parser.add_argument("--max-steps", type=int, help="Override the step budget.")
    parser.add_argument("--max-usd", type=float, help="Override the cost budget.")
    parser.add_argument("--max-seconds", type=float, help="Override the wall-clock budget.")
    parser.add_argument("--model", help="Override the agent model.")
    parser.add_argument("--quiet", action="store_true", help="Print only the answer.")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Settings:
    overrides = {
        "max_steps": args.max_steps,
        "max_run_cost_usd": args.max_usd,
        "max_seconds": args.max_seconds,
        "agent_model": args.model,
    }
    given = {key: value for key, value in overrides.items() if value is not None}
    return settings().model_copy(update=given) if given else settings()


def render(trace: RunTrace, quiet: bool) -> str:
    if quiet:
        return trace.answer

    lines: list[str] = []
    if trace.plan["subquestions"]:
        lines.append("Plan")
        for question in trace.plan["subquestions"]:
            lines.append(f"  {'[x]' if trace.coverage.get(question) else '[ ]'} {question}")
        lines.append("")

    tool_steps = [s for s in trace.steps if s.get("observations")]
    if tool_steps:
        lines.append("Steps")
        for step in tool_steps:
            for observation in step["observations"]:
                status = "ok " if observation["ok"] else "ERR"
                args = ", ".join(f"{k}={str(v)[:50]}" for k, v in observation["args"].items())
                lines.append(
                    f"  {step['i']:>2}. [{status}] {observation['tool']}({args})"
                    f"  {observation['latency_ms']:.0f}ms"
                )
                detail = (
                    observation["error"]
                    if not observation["ok"]
                    else observation["content"][:120].replace("\n", " ")
                )
                lines.append(f"        {detail}")
        lines.append("")

    lines.append("Answer")
    lines.append(trace.answer)
    lines.append("")

    if trace.sources:
        lines.append("Sources")
        for source in trace.sources:
            cited = "*" if source["sid"] in trace.citations else " "
            lines.append(f"  {cited}[{source['sid']}] {source['title'][:70]}")
            lines.append(f"      {source['url']}")
        lines.append("")

    totals = trace.usage["totals"]
    lines.append(
        f"{trace.usage['steps']} steps · {totals['cost_usd']:.5f} USD · "
        f"{trace.timings_ms['total'] / 1000:.1f}s · "
        f"{totals['input_tokens']:,} in / {totals['output_tokens']:,} out · "
        f"cache {totals['cache_hit_rate']:.0%} · {trace.usage['search_credits']} search credits"
    )
    if trace.outcome.early_exit_reason:
        lines.append(f"stopped early: {trace.outcome.early_exit_reason}")
    if trace.outcome.used_deterministic_finalize:
        lines.append("answer assembled without an LLM (deterministic fallback)")
    if trace.outcome.error:
        lines.append(f"error: {trace.outcome.error}")
    lines.append(spend.summary())
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trace = run(args.question, variant=args.variant, cfg=build_config(args), trace_path=args.json)
    print(render(trace, args.quiet))
    if args.json:
        print(f"\ntrace written to {args.json}")
    # A budget breach is a successful outcome, not a failure: the run returned a
    # partial answer exactly as designed. Only a genuine crash is non-zero.
    return 1 if trace.outcome.status == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
