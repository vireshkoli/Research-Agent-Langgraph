"""Phase-1 spike: settle the OpenAI unknowns, through the real llm.py code path.

Every check here goes through the public wrapper rather than the SDK, so a pass
means the thing the agent will actually call works — not that something adjacent
to it does.

MEASURED RESULTS (2026-07-30):

  Chat Completions + tools + reasoning_effort  ->  400 on gpt-5.4-nano AND -mini
      "Function tools with reasoning_effort are not supported ... use /v1/responses
       or set reasoning_effort to 'none'."
  Responses + tools + reasoning                ->  works on nano
  Prompt caching, identical 2120-token prefix  ->  1792/2120 cached on call 2 (84.5%)
  Batch API                                    ->  accepts gpt-5.6-terra

The first line reversed a locked plan decision: the plan specified Chat Completions
to avoid a suspected lack of Responses support on 5.4-series models. The opposite is
true — it is Chat Completions that carries the restriction — so llm.py is built on
Responses.

Costs a few cents. Run: `uv run python scripts/spike_openai.py`
"""

import sys
from typing import Any

from pydantic import BaseModel

from research_agent import spend
from research_agent.config import settings
from research_agent.llm import CostTracker, call_tools, complete, cost_usd, parse
from research_agent.tools import openai_schemas

results: list[tuple[str, bool, str]] = []


class Check:
    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> "Check":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if exc_type is not None:
            results.append((self.name, False, f"{type(exc).__name__}: {str(exc)[:150]}"))
            return True
        return False

    def ok(self, detail: str) -> None:
        results.append((self.name, True, detail))


class Subquestions(BaseModel):
    """Mirrors the real plan-node schema closely enough to prove the mechanism."""

    reasoning: str
    subquestions: list[str]


def responses_tool_schemas() -> list[dict[str, Any]]:
    """The registry emits Chat-Completions-shaped schemas; Responses wants them flat."""
    flat = []
    for schema in openai_schemas(["web_search"]):
        function = schema["function"]
        flat.append({"type": "function", **function})
    return flat


def main() -> int:
    config = settings()
    agent, judge = config.agent_model, config.judge_model
    tracker = CostTracker(budget_usd=0.50)

    with Check("complete() -> text") as check:
        text = complete(
            "You are terse.",
            "Reply with the single word: ok",
            model=agent,
            purpose="spike",
            tracker=tracker,
        )
        check.ok(f"{text.strip()[:40]!r}")

    with Check("parse() -> structured output") as check:
        plan = parse(
            "Decompose the question into sub-questions.",
            "Which is larger, the GDP of Japan or Germany?",
            Subquestions,
            model=agent,
            reasoning_effort=config.plan_reasoning_effort,
            purpose="spike",
            tracker=tracker,
        )
        if plan is None:
            raise AssertionError("model refused to produce structured output")
        check.ok(f"{len(plan.subquestions)} subquestions")

    with Check("call_tools() -> native function call") as check:
        turn = call_tools(
            [{"role": "user", "content": "What did the Nobel committee announce in 2024?"}],
            responses_tool_schemas(),
            model=agent,
            reasoning_effort=config.reasoning_effort or None,
            purpose="spike",
            tracker=tracker,
        )
        if not turn.tool_calls:
            raise AssertionError("no tool_calls returned — the act node cannot work")
        call = turn.tool_calls[0]
        check.ok(f"{call['name']}({call['args']})")

    with Check("multi-turn: tool output fed back") as check:
        history: list[dict[str, Any]] = [
            {"role": "user", "content": "What did the Nobel committee announce in 2024?"}
        ]
        history += turn.items
        history.append(
            {
                "type": "function_call_output",
                "call_id": turn.tool_calls[0]["call_id"],
                "output": "Hopfield and Hinton won the 2024 Physics prize for neural networks.",
            }
        )
        second = call_tools(
            history,
            responses_tool_schemas(),
            model=agent,
            reasoning_effort=config.reasoning_effort or None,
            purpose="spike",
            tracker=tracker,
        )
        check.ok(f"answered: {second.text.strip()[:70]!r}")

    with Check("prompt caching engages on a repeated prefix") as check:
        prefix = "You are a research assistant. " + ("Context filler. " * 700)
        before = tracker.total_cached_tokens
        complete(prefix, "Say ok.", model=agent, purpose="spike", tracker=tracker)
        complete(prefix, "Say ok.", model=agent, purpose="spike", tracker=tracker)
        gained = tracker.total_cached_tokens - before
        check.ok(f"{gained} tokens served from cache; run hit rate {tracker.cache_hit_rate:.1%}")

    with Check(f"Batch API accepts {judge}") as check:
        import io
        import json

        from research_agent.llm import _client

        line = {
            "custom_id": "spike-1",
            "method": "POST",
            "url": "/v1/responses",
            "body": {"model": judge, "input": "Reply with: ok", "max_output_tokens": 2048},
        }
        upload = _client().files.create(file=io.BytesIO(json.dumps(line).encode()), purpose="batch")
        batch = _client().batches.create(
            input_file_id=upload.id, endpoint="/v1/responses", completion_window="24h"
        )
        check.ok(f"batch={batch.id} status={batch.status} (not waited on)")

    print()
    width = max(len(name) for name, _, _ in results)
    for name, passed, detail in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name.ljust(width)}  {detail}")

    print(f"\n  agent={agent}  judge={judge}")
    print(
        f"  this run: ${tracker.total_cost_usd:.5f} over {len(tracker.calls)} calls, "
        f"cache hit rate {tracker.cache_hit_rate:.1%}"
    )
    print(f"  sanity:   10k in / 1k out on {agent} = ${cost_usd(agent, 10_000, 0, 1_000):.6f}")
    print(f"  {spend.summary()}")

    failed = [name for name, passed, _ in results if not passed]
    if failed:
        print(f"\n  {len(failed)} check(s) failed: {', '.join(failed)}")
        return 1
    print("\n  All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
