"""Every prompt in the project, in one place, plus the helpers that render state.

**Prompt ordering is a cost decision.** OpenAI routes a request to a cache by
hashing its prefix, so the layout of a turn is deliberate:

    [ system + tool guidance ]   stable for the whole run  -> cached
    [ user: the question      ]   stable for the whole run  -> cached
    [ ...the conversation...  ]   append-only               -> cached from turn 2
    [ user: current context   ]   re-rendered every turn    -> the only uncached part

Sources, plan coverage and the rolling summary go in that *last* block rather than
being baked into the first. That keeps two properties at once: the citation
registry is re-rendered fresh on every turn (so compaction can never strip a
citation out of the model's view), and everything before it stays byte-identical
and therefore cached. A spike measured 1792 of 2120 tokens served from cache on a
repeated prefix; cached input bills at 10% of list.

`prompt_hash()` goes into every trace so a committed result names the prompt
version that produced it.
"""

import hashlib
from typing import Any

from research_agent.state import ResearchState, Source

SYSTEM = """\
You are a research agent. You answer questions by searching for evidence and \
citing it, never from memory alone.

How you work:
- Call tools to gather evidence. Prefer several specific searches over one broad one.
- Every factual claim in your final answer must be traceable to a source you \
actually retrieved.
- Use the calculator for arithmetic. Do not compute in your head and do not use \
code_execution for a single expression.
- If a search fails or returns nothing useful, try different terms rather than \
repeating the same query.
- Stop calling tools once you can answer. Do not keep searching to feel thorough.

You are on a hard budget of steps, wall-clock time and money. Running out is normal \
and is not a failure: an honest partial answer that says what is still unknown is \
worth more than a padded one. Never invent a source or a URL."""

PLAN_SYSTEM = """\
Break a research question into the minimum set of sub-questions that must each be \
answered before the whole question can be.

Rules:
- Between 1 and 5 sub-questions. Use 1 if the question is genuinely single-hop.
- Each must be independently searchable and factual.
- Do not include steps like "combine the results" or "verify the answer". Those are \
not sub-questions.
- Prefer fewer. Every extra sub-question costs a search."""

REFLECT_SYSTEM = """\
You are checking whether a research agent can stop yet.

You see the sub-questions and a one-line digest of each step taken. Decide:
- "continue" — evidence is still missing for at least one sub-question and another \
tool call would plausibly find it.
- "replan"   — the original sub-questions were the wrong decomposition. Expensive; \
only when the evidence shows the plan itself was misconceived.
- "finalize" — every sub-question is answered, OR further searching is unlikely to \
help.

Two failure modes to avoid, in order of how common they are:
1. Stopping early with a sub-question still unanswered.
2. Continuing to search when the evidence is already sufficient, or when repeated \
searches have returned nothing useful. If the last two steps added nothing, \
finalize.

Mark a sub-question covered only if the observations actually contain its answer."""

FINALIZE_SYSTEM = """\
Write the final answer from the evidence gathered.

Rules:
- Cite with bracketed source ids exactly as given, like [S1] or [S2][S4]. Cite the \
specific source each claim came from.
- Never cite an id that is not in the source list. Never invent a URL.
- Answer the question directly in the first sentence. No preamble.
- If the evidence is incomplete, give what is supported and state plainly what \
could not be established. A short honest answer beats a padded one.
- If the evidence does not support an answer at all, say so instead of guessing."""

COMPACT_SYSTEM = """\
Compress the earlier steps of a research run into a short summary, so the agent can \
keep working without re-reading everything.

Preserve, in this order of priority:
1. Facts found, each with the [Sn] source id it came from.
2. Which sub-questions those facts answer.
3. Approaches that FAILED — "searched X, nothing useful". This is the most valuable \
line in the summary: it is what stops the agent re-issuing a query that already \
came back empty.

Drop: full page text, repeated boilerplate, anything already superseded. Be terse. \
Bullet points, no preamble."""


def render_sources(sources: list[Source]) -> str:
    if not sources:
        return "(no sources gathered yet)"
    return "\n".join(f"[{s['sid']}] {s['title']}\n      {s['url']}" for s in sources)


def render_plan(state: ResearchState) -> str:
    plan = state.get("plan") or []
    if not plan:
        return "(no plan)"
    covered = state.get("covered") or {}
    return "\n".join(f"{'[x]' if covered.get(q) else '[ ]'} {q}" for q in plan)


def render_digest(state: ResearchState, limit: int | None = None) -> str:
    """One line per step: what was called, and what came back.

    This is what `reflect` sees instead of the raw scratchpad — around 1-2k tokens
    against act's 5-7k. Reflect is answering a different and much smaller question
    ("is anything still missing?"), so it does not need the full observations.
    """
    steps = state.get("scratchpad") or []
    lines: list[str] = []
    for step in steps:
        for observation in step.get("observations") or []:
            args = ", ".join(f"{k}={str(v)[:60]!r}" for k, v in (observation["args"] or {}).items())
            outcome = (
                observation["content"][:200].replace("\n", " ")
                if observation["ok"]
                else f"FAILED: {observation['error']}"
            )
            sids = f" -> {','.join(observation['source_ids'])}" if observation["source_ids"] else ""
            lines.append(f"{step['i']}. {observation['tool']}({args}){sids}: {outcome}")
    if not lines:
        return "(no steps taken yet)"
    return "\n".join(lines[-limit:] if limit else lines)


def render_context(state: ResearchState) -> str:
    """The per-turn block: plan coverage, rolling summary, and the source registry.

    Rendered last in the turn so everything before it stays cacheable, and rendered
    fresh so the model always sees the current source list.
    """
    parts = [f"Question: {state['question']}", "", "Sub-questions:", render_plan(state)]
    if summary := state.get("summary"):
        parts += ["", "Summary of earlier steps:", summary]
    parts += [
        "",
        "Sources gathered so far (cite these ids):",
        render_sources(state.get("sources", [])),
    ]
    if gaps := state.get("open_gaps"):
        parts += ["", "Still missing:", "\n".join(f"- {g}" for g in gaps)]
    return "\n".join(parts)


def reflect_user(state: ResearchState) -> str:
    return "\n\n".join(
        [
            f"Question: {state['question']}",
            "Sub-questions:\n" + render_plan(state),
            "Steps taken:\n" + render_digest(state),
            f"Steps used: {state.get('step', 0)}",
        ]
    )


def finalize_user(state: ResearchState) -> str:
    return "\n\n".join(
        [
            f"Question: {state['question']}",
            "Sub-questions:\n" + render_plan(state),
            "Evidence gathered:\n" + render_digest(state),
            "Sources you may cite:\n" + render_sources(state.get("sources", [])),
        ]
    )


def compact_user(state: ResearchState, steps_text: str) -> str:
    previous = state.get("summary") or "(none yet)"
    return f"Existing summary:\n{previous}\n\nNew steps to fold in:\n{steps_text}"


def prompt_hash() -> str:
    """Fingerprint of every template, stamped into each trace.

    Turns a committed result file into something that names the code version that
    produced it — a result whose prompts have since changed is identifiable rather
    than silently stale.
    """
    blob = "\x00".join([SYSTEM, PLAN_SYSTEM, REFLECT_SYSTEM, FINALIZE_SYSTEM, COMPACT_SYSTEM])
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()[:12]


def responses_tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-shape registry schemas for the Responses API, which flattens the function."""
    return [{"type": "function", **schema["function"]} for schema in schemas]
