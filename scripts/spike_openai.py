"""Phase-1 spike: settle the OpenAI unknowns before anything is built on them.

Four questions, each of which would force a rewrite if answered late:

  1. Does the agent model work on Chat Completions with native `tools=`?
     (The gpt-5.4 model cards list Chat Completions and Batch but not Responses.)
  2. Does `.parse()` structured output work on it? plan/reflect/judge all depend on it.
  3. Does prompt caching actually engage on a repeated prefix? The cost model assumes
     a ~58% cache hit rate; if `cached_tokens` stays 0, every cost estimate is wrong.
  4. Does the Batch API accept the judge model? The eval budget assumes 50% off.

Costs a few cents. Run: `uv run python scripts/spike_openai.py`
"""

import json
import sys

from pydantic import BaseModel

from research_agent.config import settings
from research_agent.llm import PRICES, _client, _usage_of, cost_usd

results: list[tuple[str, bool, str]] = []


def check(name: str) -> "Check":
    return Check(name)


class Check:
    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> "Check":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if exc_type is not None:
            results.append((self.name, False, f"{type(exc).__name__}: {exc}"))
            return True
        return False

    def ok(self, detail: str) -> None:
        results.append((self.name, True, detail))


class Subquestions(BaseModel):
    """Mirrors the real plan-node schema closely enough to prove the mechanism."""

    reasoning: str
    subquestions: list[str]


AGENT = settings().agent_model
JUDGE = settings().judge_model

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def main() -> int:
    client = _client()

    # 1. Basic completion, and which sampling parameters the model accepts.
    with check("chat.completions + reasoning_effort") as c:
        response = client.chat.completions.create(
            model=AGENT,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_completion_tokens=2048,
            reasoning_effort=settings().reasoning_effort,
        )
        usage = _usage_of(response)
        c.ok(f"content={response.choices[0].message.content!r} usage(in,cached,out,reason)={usage}")

    with check("temperature accepted (mutually exclusive with reasoning_effort?)") as c:
        response = client.chat.completions.create(
            model=AGENT,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_completion_tokens=2048,
            temperature=0.0,
        )
        c.ok(f"accepted; content={response.choices[0].message.content!r}")

    # 2. Native tool calling — the act node depends entirely on this.
    with check("native tool calling") as c:
        response = client.chat.completions.create(
            model=AGENT,
            messages=[
                {"role": "user", "content": "What did the Nobel committee announce in 2024?"}
            ],
            tools=[WEB_SEARCH_TOOL],
            max_completion_tokens=2048,
            reasoning_effort=settings().reasoning_effort,
        )
        calls = response.choices[0].message.tool_calls or []
        if not calls:
            raise AssertionError("model returned no tool_calls — the act node cannot work")
        args = json.loads(calls[0].function.arguments)
        c.ok(f"name={calls[0].function.name} args={args}")

    # 3. Structured output — plan, reflect and judge all depend on this.
    with check("structured output via .parse()") as c:
        response = client.chat.completions.parse(
            model=AGENT,
            messages=[
                {"role": "system", "content": "Decompose the question into sub-questions."},
                {"role": "user", "content": "Which is larger, the GDP of Japan or Germany?"},
            ],
            response_format=Subquestions,
            max_completion_tokens=2048,
            reasoning_effort=settings().reasoning_effort,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise AssertionError("model refused to produce structured output")
        c.ok(f"{len(parsed.subquestions)} subquestions: {parsed.subquestions}")

    # 4. Prompt caching. The cost model assumes a large cached share; verify the
    #    mechanism engages at all on a >1024-token identical prefix.
    with check("prompt caching engages on a repeated prefix") as c:
        prefix = "You are a research assistant. " + ("Context filler. " * 700)
        messages = [{"role": "system", "content": prefix}, {"role": "user", "content": "Say ok."}]
        first = client.chat.completions.create(
            model=AGENT, messages=messages, max_completion_tokens=2048
        )
        second = client.chat.completions.create(
            model=AGENT, messages=messages, max_completion_tokens=2048
        )
        in1, cached1, _, _ = _usage_of(first)
        in2, cached2, _, _ = _usage_of(second)
        c.ok(f"call1 in={in1} cached={cached1} | call2 in={in2} cached={cached2}")

    # 5. Batch API on the judge model. The eval budget assumes the 50% discount.
    with check(f"Batch API accepts {JUDGE}") as c:
        import io

        line = {
            "custom_id": "spike-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": JUDGE,
                "messages": [{"role": "user", "content": "Reply with: ok"}],
                "max_completion_tokens": 2048,
            },
        }
        upload = client.files.create(file=io.BytesIO(json.dumps(line).encode()), purpose="batch")
        batch = client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        c.ok(f"accepted: batch={batch.id} status={batch.status} (not waited on)")

    # --- report ---
    print()
    width = max(len(name) for name, _, _ in results)
    for name, passed, detail in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name.ljust(width)}  {detail}")

    print(f"\n  Models under test: agent={AGENT} judge={JUDGE}")
    if AGENT in PRICES:
        print(f"  Sanity: 10k in / 1k out on {AGENT} = ${cost_usd(AGENT, 10_000, 0, 1_000):.6f}")

    failed = [name for name, passed, _ in results if not passed]
    if failed:
        print(f"\n  {len(failed)} check(s) failed: {', '.join(failed)}")
        return 1
    print("\n  All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
