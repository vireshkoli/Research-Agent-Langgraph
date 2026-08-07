"""Web search via Tavily, with a circuit-broken free fallback.

Tavily is primary because its free tier is 1000 credits per month, recurring, with
no credit card on file. For a demo anyone on the internet can hit, the worst case is
therefore zero dollars rather than an open bar. `basic` depth costs 1 credit;
`include_answer` is free and gives the agent a synthesised summary alongside the
raw results.

`ddgs` is wired as a fallback but is not advertised as a feature. It is heavily
rate-limited and will be worse on a shared cloud egress IP than on a laptop. It
exists so that quota exhaustion degrades into a slightly worse search rather than a
dead tool, and so the tool-failure path in `observe` gets exercised for real.

Every failure returns `ToolResult(ok=False, ...)`. Nothing here raises: the agent
should reason about a failed search, and `observe` counts consecutive failures and
trips a circuit breaker.
"""

import json
import os
from pathlib import Path
from typing import Any

import httpx

from research_agent.config import settings
from research_agent.tools.base import Source, ToolResult, ToolSpec, truncate

TAVILY_ENDPOINT = "https://api.tavily.com/search"
MAX_CONTENT_CHARS = 4000
CACHE_DIR = Path(".cache/search")

# Tripped when Tavily reports quota exhaustion, so the remaining steps of a run stop
# paying the latency of a call that is going to fail.
_tavily_exhausted = False


def _reset_circuit() -> None:
    """Test-only hook."""
    global _tavily_exhausted
    _tavily_exhausted = False


def _cache_path(query: str, depth: str, max_results: int) -> Path:
    import hashlib

    key = hashlib.sha256(f"{query}|{depth}|{max_results}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"{key}.json"


def _search_tavily(query: str, max_results: int, depth: str) -> dict[str, Any]:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set")
    response = httpx.post(
        TAVILY_ENDPOINT,
        json={
            "query": query,
            "search_depth": depth,
            "max_results": max_results,
            "include_answer": "basic",
            # Raw content is deliberately off: it returns 100k+ chars per result and
            # fetch_page exists for when the agent genuinely needs a full page.
            "include_raw_content": False,
        },
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=settings().fetch_timeout_seconds,
    )
    if response.status_code in (429, 432, 433):
        raise QuotaExhausted(f"Tavily quota or rate limit hit (HTTP {response.status_code})")
    response.raise_for_status()
    return response.json()


class QuotaExhausted(Exception):
    """Tavily is out of credits or rate-limiting us; trip the circuit breaker."""


def _search_ddgs(query: str, max_results: int) -> dict[str, Any]:
    """Free, keyless, and unreliable. Shaped to look like a Tavily response."""
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise RuntimeError("ddgs is not installed") from exc

    with DDGS() as client:
        hits = list(client.text(query, max_results=max_results))
    return {
        "answer": None,
        "results": [
            {
                "title": hit.get("title", ""),
                "url": hit.get("href") or hit.get("url", ""),
                "content": hit.get("body", ""),
            }
            for hit in hits
        ],
    }


def _render(query: str, payload: dict[str, Any], backend: str) -> ToolResult:
    results = payload.get("results") or []
    if not results:
        return ToolResult.failure(f"no results for {query!r}", backend=backend)

    sources = [
        Source(
            url=result.get("url", ""),
            title=result.get("title", "") or result.get("url", ""),
            snippet=(result.get("content") or "")[:300],
            tool="web_search",
        )
        for result in results
        if result.get("url")
    ]

    lines = []
    if answer := payload.get("answer"):
        lines.append(f"Summary: {answer}\n")
    for index, result in enumerate(results, start=1):
        lines.append(f"[{index}] {result.get('title', '')}")
        lines.append(f"    {result.get('url', '')}")
        lines.append(f"    {(result.get('content') or '').strip()}")
        lines.append("")

    content, raw_chars, was_truncated = truncate("\n".join(lines).strip(), MAX_CONTENT_CHARS)
    return ToolResult(
        ok=True,
        content=content,
        sources=sources,
        raw_chars=raw_chars,
        truncated=was_truncated,
        # credits is what the run-level search budget counts against Tavily's
        # 1000/month free tier.
        meta={"backend": backend, "credits": 1 if backend == "tavily" else 0},
    )


def web_search(query: str) -> ToolResult:
    """Search the web and return ranked results with a synthesised summary."""
    global _tavily_exhausted

    query = (query or "").strip()
    if not query:
        return ToolResult.failure("empty query")

    config = settings()
    max_results, depth = config.search_max_results, config.search_depth

    cache_file = _cache_path(query, depth, max_results)
    if config.search_cache_enabled and cache_file.is_file():
        payload = json.loads(cache_file.read_text())
        return _render(query, payload, backend="cache")

    if not _tavily_exhausted:
        try:
            payload = _search_tavily(query, max_results, depth)
            if config.search_cache_enabled:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(payload))
            return _render(query, payload, backend="tavily")
        except QuotaExhausted:
            _tavily_exhausted = True
        except (httpx.HTTPError, RuntimeError, ValueError):
            pass  # fall through to the fallback rather than failing the step

    try:
        return _render(query, _search_ddgs(query, max_results), backend="ddgs")
    except Exception as exc:  # noqa: BLE001 — a dead search must not end the run
        return ToolResult.failure(
            f"search unavailable ({type(exc).__name__}). Tavily "
            f"{'is out of quota' if _tavily_exhausted else 'failed'} and the fallback "
            f"did not respond. Answer from what you already have.",
            backend="none",
        )


SPEC = ToolSpec(
    name="web_search",
    description=(
        "Search the web and return the top results with titles, URLs and snippets, "
        "plus a short synthesised summary. Use specific, keyword-style queries. Issue "
        "one search per distinct fact you need rather than one broad search for "
        "everything. Do not use it for arithmetic — use the calculator."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search terms, e.g. 'Llama 3.1 405B parameter count'.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    fn=web_search,
    max_calls=8,
)
