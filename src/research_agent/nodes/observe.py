"""observe — run the tools the model asked for and fold the results into state.

Hand-built rather than `langgraph.prebuilt.ToolNode`, because four things have to
happen here that ToolNode does not offer cleanly: per-tool budget accounting,
per-call timing, source-id minting, and a failure counter feeding a circuit breaker.

**Fan-out stays inside this node.** Parallel tool calls run in a thread pool, never
as parallel graph branches. Two reasons, and both matter: branches writing the same
reducer-less state key in one super-step raise `InvalidUpdateError`, and each branch
would burn a super-step against the recursion limit.

**Sources are minted here and stored in state, not in message text.** That is the
structural reason compaction cannot lose a citation — the registry is not part of
what gets compacted. Deduplication is by URL against both the existing registry and
the current batch, so re-finding a page reuses its id rather than issuing a second.

**Failures are data.** A tool that errors produces an error observation the model can
read and react to, plus `consecutive_tool_failures` which resets to zero on any
success. Three in a row trips the circuit breaker in `budget_verdict`.
"""

import concurrent.futures
import time
from typing import Any

from research_agent.config import Settings, settings
from research_agent.nodes import make_step
from research_agent.state import Observation, ResearchState, Source, ToolCall
from research_agent.tools.base import ToolResult
from research_agent.tools.registry import dispatch

MAX_PARALLEL = 4


def observe_node(state: ResearchState, cfg: Settings | None = None) -> ResearchState:
    cfg = cfg or settings()
    pending = state.get("pending_calls") or []

    if not pending:
        # act emitted no tool calls. A no-op keeps the act -> observe edge static,
        # which keeps the diagram honest for one cheap super-step.
        return {
            "pending_calls": [],
            "scratchpad": [make_step(state, "observe", note="no tool calls")],
        }

    results = _run_all(pending, cfg)

    observations: list[Observation] = []
    new_sources: list[Source] = []
    messages: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    credits = 0
    failures = 0

    known_urls = {source["url"]: source["sid"] for source in state.get("sources", [])}
    next_index = len(state.get("sources", [])) + 1

    for call, result, latency_ms in results:
        counts[call["name"]] = counts.get(call["name"], 0) + 1
        credits += int(result.meta.get("credits", 0))
        failures = failures + 1 if not result.ok else 0

        source_ids: list[str] = []
        for source in result.sources:
            if not source.url:
                continue
            if source.url in known_urls:
                source_ids.append(known_urls[source.url])
                continue
            sid = f"S{next_index}"
            next_index += 1
            known_urls[source.url] = sid
            new_sources.append(
                Source(
                    sid=sid,
                    url=source.url,
                    title=source.title,
                    snippet=source.snippet,
                    tool=source.tool,
                    first_seen_step=state.get("step", 0),
                )
            )
            source_ids.append(sid)

        observations.append(
            Observation(
                call_id=call["call_id"],
                tool=call["name"],
                args=call["args"],
                ok=result.ok,
                content=result.content,
                error=result.error,
                latency_ms=latency_ms,
                raw_chars=result.raw_chars,
                truncated=result.truncated,
                source_ids=source_ids,
            )
        )
        messages.append(
            {
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": _for_model(result, source_ids),
            }
        )

    return {
        "messages": messages,
        "sources": new_sources,
        "pending_calls": [],  # drained; this key is reducer-less so it empties
        "tool_calls_by_type": counts,
        "search_credits": credits,
        # Reducer-less on purpose: a success must reset the streak to zero, which
        # an additive reducer could never do.
        "consecutive_tool_failures": (
            state.get("consecutive_tool_failures", 0) + failures if failures else 0
        ),
        "scratchpad": [make_step(state, "observe", observations=observations)],
    }


def _for_model(result: ToolResult, source_ids: list[str]) -> str:
    """What the model sees. Source ids are attached so it can cite them later."""
    if not result.ok:
        return result.content
    if not source_ids:
        return result.content
    return f"{result.content}\n\n(cite as: {', '.join(f'[{sid}]' for sid in source_ids)})"


def _run_all(pending: list[ToolCall], cfg: Settings) -> list[tuple[ToolCall, ToolResult, float]]:
    """Execute calls in parallel, preserving request order in the results.

    Order is preserved because the model's own numbering of its calls is the order
    the observations should read back in; a thread pool would otherwise return them
    by completion time and make traces hard to follow.
    """
    if len(pending) == 1:
        return [_run_one(pending[0], cfg)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = [pool.submit(_run_one, call, cfg) for call in pending]
        return [future.result() for future in futures]


def _run_one(call: ToolCall, cfg: Settings) -> tuple[ToolCall, ToolResult, float]:
    started = time.perf_counter()
    args = call["args"]
    if "__malformed__" in args:
        # The model emitted arguments that were not valid JSON. Telling it so is
        # more useful than a crash, and it usually recovers on the next turn.
        result = ToolResult.failure(f"arguments were not valid JSON: {args['__malformed__'][:200]}")
    else:
        result = dispatch(call["name"], args)
    return call, result, (time.perf_counter() - started) * 1000
