"""Phase-1 spike: confirm Tavily works on a free key and report the credit cost.

Tavily is the primary web_search backend because its free tier is 1000 credits per
month, recurring, with no credit card on file — so the worst case for a public demo
is zero dollars rather than an open bar.

Three things to confirm before the tool is built on it:
  1. A free key authenticates and returns results.
  2. `include_answer` returns a synthesised answer at no extra credit cost.
  3. `basic` search depth costs 1 credit, which is what the eval budget assumes
     (~360 credits for a 30-case x 3-run evaluation).

Costs 2-3 credits. Run: `uv run python scripts/spike_tavily.py`
"""

import os
import sys

import httpx

from research_agent.llm import _load_env_file

QUERY = "2024 Nobel Prize in Physics laureates"
ENDPOINT = "https://api.tavily.com/search"


def search(api_key: str, **overrides: object) -> dict:
    payload: dict[str, object] = {
        "query": QUERY,
        "search_depth": "basic",  # 1 credit; "advanced" is 2
        "max_results": 5,
        "include_answer": "basic",
    }
    payload.update(overrides)
    response = httpx.post(
        ENDPOINT,
        json=payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def main() -> int:
    _load_env_file()
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        print("  TAVILY_API_KEY is not set. Get a free key (no card) at https://app.tavily.com")
        return 1

    data = search(api_key)
    results = data.get("results", [])
    answer = data.get("answer")

    print(f"\n  query          {QUERY!r}")
    print(f"  results        {len(results)}")
    print(f"  include_answer {'yes — ' + answer[:120] if answer else 'NO ANSWER RETURNED'}")
    print(f"  response_time  {data.get('response_time')}s")

    if not results:
        print("\n  FAIL: no results returned.")
        return 1

    print("\n  top results:")
    for result in results[:3]:
        print(f"    - {result.get('title', '')[:70]}")
        print(f"      {result.get('url', '')}")
        print(f"      content: {len(result.get('content') or '')} chars")

    # The raw_content option is what would blow the context window without the
    # max_observation_chars cap, so measure how big it actually gets.
    raw = search(api_key, include_raw_content=True)
    sizes = [len(r.get("raw_content") or "") for r in raw.get("results", [])]
    total = sum(sizes)
    print(f"\n  include_raw_content total: {total:,} chars across {len(sizes)} results")
    print(
        f"  -> truncation cap is {'justified' if total > 20_000 else 'less critical'} "
        f"(RA_MAX_OBSERVATION_CHARS)"
    )

    print("\n  All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
