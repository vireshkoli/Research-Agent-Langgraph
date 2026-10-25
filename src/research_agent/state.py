"""The graph's state, and the reducers that decide how nodes combine into it.

Two structural decisions are worth knowing before reading:

**No `add_messages`, no langchain message objects.** `llm.py` speaks the raw OpenAI
wire format, so `add_messages` would coerce dicts into `langchain_core` objects that
every call then converts back. Its one real value-add, ID-based message dedup, is
irrelevant here because messages are never mutated by ID. `Annotated[list[dict],
operator.add]` says exactly what is happening.

**Tool fan-out happens inside the `observe` node, never as parallel graph branches.**
That property is load-bearing and fragile. LangGraph raises `InvalidUpdateError` when
two branches write the same reducer-less key in one super-step, and most of the keys
below are deliberately reducer-less. Adding a parallel branch to this graph will
break it in exactly that way.

Nodes return **deltas, not totals**. Under `operator.add`, a node that returns the
whole list instead of its new items silently duplicates everything, and nothing will
tell you — see `test_additive_reducers_receive_deltas_not_totals`.
"""

import operator
import time
import uuid
from typing import Annotated, Any, Literal, TypedDict

# Closed set so the eval can assert on it and the UI can render it.
EarlyExitReason = Literal[
    "max_steps",
    "wall_clock",
    "budget_usd",
    "tool_failures",
    "max_replans",
    "tool_cap",
    "recursion_limit",
    "internal_error",
]

Variant = Literal["full", "baseline", "no_overrule"]

# What reflect decided to do next. "replan" is capped by max_replans.
ReflectDecision = Literal["continue", "replan", "finalize"]


class ToolCall(TypedDict):
    call_id: str
    name: str
    args: dict[str, Any]


class Observation(TypedDict):
    call_id: str
    tool: str
    args: dict[str, Any]
    ok: bool
    content: str  # truncated, model-facing
    error: str | None
    latency_ms: float
    raw_chars: int  # size before truncation, so a debugger sees what was dropped
    truncated: bool
    source_ids: list[str]


class Source(TypedDict):
    """A citable document, addressed by a short id like "S3".

    Sources live here rather than in message text. That is what makes compaction
    structurally unable to drop a citation: it rewrites the scratchpad, and the
    registry is not in the scratchpad.
    """

    sid: str
    url: str
    title: str
    snippet: str
    tool: str
    first_seen_step: int


class Step(TypedDict):
    """One node execution, as it will appear in the trace."""

    i: int
    node: str
    plan_version: int
    ts_offset_ms: float
    tool_calls: list[ToolCall]
    observations: list[Observation]
    reflection: dict[str, Any] | None
    note: str | None


def merge_counts(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    """Sum per-tool call counts.

    A plain ``{**left, **right}`` would *overwrite* counts rather than add them,
    which would silently disable the per-tool budget caps.
    """
    merged = dict(left)
    for key, value in right.items():
        merged[key] = merged.get(key, 0) + value
    return merged


class ResearchState(TypedDict, total=False):
    # --- inputs, written once by the caller ---
    question: str
    run_id: str
    variant: Variant
    # perf_counter() at run start. Set by the caller, NOT by `plan`: plan is
    # re-entrant on replan and would reset the wall clock every time it ran.
    t_start: float

    # --- conversation, append-only ---
    messages: Annotated[list[dict[str, Any]], operator.add]

    # --- the plan. Deliberately reducer-less: on replan the list is *replaced*.
    # operator.add would concatenate the stale plan onto the new one, and a merge
    # reducer over `covered` would resurrect keys belonging to a dead plan.
    plan: list[str]
    plan_version: int
    covered: dict[str, bool]
    open_gaps: list[str]
    # What reflect concluded. Reducer-less and replaced every time reflect runs;
    # the router reads it rather than re-deriving intent from the gap list.
    reflect_decision: ReflectDecision | None

    # --- working memory ---
    scratchpad: Annotated[list[Step], operator.add]
    compacted_upto: int  # index into scratchpad; earlier steps are summarised
    summary: str
    compactions: Annotated[int, operator.add]
    sources: Annotated[list[Source], operator.add]

    # --- act -> observe handoff. Reducer-less because `observe` drains it by
    # returning []; additive would never empty.
    pending_calls: list[ToolCall]
    act_requested_stop: bool

    # --- counters used by budget_verdict ---
    step: Annotated[int, operator.add]
    spend_usd: Annotated[float, operator.add]  # additive: last-write-wins would
    # let any node that forgets to add prior spend silently reset the budget
    tool_calls_by_type: Annotated[dict[str, int], merge_counts]
    search_credits: Annotated[int, operator.add]
    llm_calls: Annotated[list[dict[str, Any]], operator.add]
    replans: Annotated[int, operator.add]
    reflections: Annotated[int, operator.add]
    # The number that justifies the reflect node existing. If it is 0 across the
    # whole eval, the node gets cut and the README says so.
    reflect_overrules: Annotated[int, operator.add]
    # Reducer-less: must reset to 0 on any success, which is not additive.
    consecutive_tool_failures: int

    # --- outputs ---
    early_exit_reason: EarlyExitReason | None
    answer: str
    citations: list[str]
    used_deterministic_finalize: bool


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def initial_state(question: str, variant: Variant = "full") -> ResearchState:
    """A fully-populated starting state.

    Every key is initialised even though the TypedDict is ``total=False``, so no
    node ever has to guess whether a key exists yet.
    """
    return ResearchState(
        question=question,
        run_id=new_run_id(),
        variant=variant,
        t_start=time.perf_counter(),
        messages=[],
        plan=[],
        plan_version=0,
        covered={},
        open_gaps=[],
        reflect_decision=None,
        scratchpad=[],
        compacted_upto=0,
        summary="",
        compactions=0,
        sources=[],
        pending_calls=[],
        act_requested_stop=False,
        step=0,
        spend_usd=0.0,
        tool_calls_by_type={},
        search_credits=0,
        llm_calls=[],
        replans=0,
        reflections=0,
        reflect_overrules=0,
        consecutive_tool_failures=0,
        early_exit_reason=None,
        answer="",
        citations=[],
        used_deterministic_finalize=False,
    )


def elapsed_seconds(state: ResearchState) -> float:
    start = state.get("t_start")
    return time.perf_counter() - start if start else 0.0


def next_source_id(existing: list[Source]) -> str:
    return f"S{len(existing) + 1}"


def source_by_id(state: ResearchState, sid: str) -> Source | None:
    return next((s for s in state.get("sources", []) if s["sid"] == sid), None)
