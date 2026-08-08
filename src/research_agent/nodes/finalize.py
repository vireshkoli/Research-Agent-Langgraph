"""finalize — produce the cited answer, and always produce one.

Two implementations, and the second is the reason the budget story holds up:

`_synthesize_llm` is the normal path. `_synthesize_deterministic` uses **no LLM at
all** — it assembles the question, the plan with coverage, the findings from the
scratchpad digest and the full source registry into a readable answer. It runs when
the budget is genuinely gone, when the LLM call raises anything, or when
`agent.run()` recovers from a crash.

Without it, "a budget breach returns a partial result" would be a hope. With it,
every terminal state of the graph produces something a reader (and the eval judge)
can score, which is also what makes the adversarial eval tier meaningful.

`finalize` re-runs `budget_verdict` rather than trusting the router that sent it
here. Both call the same pure function on the same state, so they agree by
construction — and the router, being a pure router, could not have written the
reason down even if it wanted to.
"""

import re

from research_agent.budget import budget_verdict
from research_agent.config import Settings, settings
from research_agent.llm import CostTracker, QueryBudgetExceeded, complete
from research_agent.nodes import call_records, make_step, spend_delta
from research_agent.prompts import FINALIZE_SYSTEM, finalize_user, render_digest
from research_agent.state import ResearchState

CITATION = re.compile(r"\[(S\d+)\]")

EXIT_EXPLANATION = {
    "max_steps": "the step budget was reached",
    "wall_clock": "the time budget was reached",
    "budget_usd": "the cost budget was reached",
    "tool_failures": "several tool calls failed in a row",
    "max_replans": "the plan was revised as often as allowed",
    "tool_cap": "a tool hit its per-run call limit",
    "recursion_limit": "the graph hit its recursion limit",
    "internal_error": "an internal error interrupted the run",
}


def finalize_node(
    state: ResearchState, tracker: CostTracker, cfg: Settings | None = None
) -> ResearchState:
    cfg = cfg or settings()
    seen = len(tracker.calls)
    reason = budget_verdict(state, cfg)

    # Hand the withheld reserve over: the loop stopped short of the cap precisely
    # so this call could afford to happen.
    tracker.release_reserve()

    try:
        answer = _synthesize_llm(state, tracker, cfg)
        used_fallback = False
    except (QueryBudgetExceeded, Exception):  # noqa: B014 - intent is "anything at all"
        answer = _synthesize_deterministic(state, reason)
        used_fallback = True

    if reason:
        why = EXIT_EXPLANATION.get(reason, reason)
        answer = f"{answer}\n\n_Note: this run stopped early because {why}._"

    return {
        "answer": answer,
        "citations": resolve_citations(state, answer),
        "early_exit_reason": reason,
        "used_deterministic_finalize": used_fallback,
        "llm_calls": call_records(tracker, seen),
        "spend_usd": spend_delta(tracker, seen),
        "scratchpad": [
            make_step(state, "finalize", note="deterministic" if used_fallback else None)
        ],
    }


def _synthesize_llm(state: ResearchState, tracker: CostTracker, cfg: Settings) -> str:
    text = complete(
        FINALIZE_SYSTEM,
        finalize_user(state),
        model=cfg.agent_model,
        purpose="finalize",
        tracker=tracker,
    ).strip()
    if not text:
        raise ValueError("empty synthesis")
    return text


def _synthesize_deterministic(state: ResearchState, reason: str | None) -> str:
    """Assemble an answer from state alone. No LLM, so it cannot fail for money."""
    lines = [
        f"Could not complete a full synthesis for: {state['question']}",
        "",
        "Here is what was established before the run stopped"
        + (f" ({EXIT_EXPLANATION.get(reason, reason)})" if reason else "")
        + ":",
        "",
    ]

    plan = state.get("plan") or []
    covered = state.get("covered") or {}
    if plan:
        lines.append("Sub-questions:")
        lines += [f"- [{'answered' if covered.get(q) else 'unanswered'}] {q}" for q in plan]
        lines.append("")

    digest = render_digest(state)
    if digest and "no steps" not in digest:
        lines += ["Evidence gathered:", digest, ""]

    sources = state.get("sources") or []
    if sources:
        lines.append("Sources:")
        lines += [f"[{s['sid']}] {s['title']} — {s['url']}" for s in sources]

    return "\n".join(lines).strip()


def resolve_citations(state: ResearchState, answer: str) -> list[str]:
    """Source ids cited in the answer that actually exist in the registry.

    Ids the model invented are dropped here and counted by the eval's
    citation-validity metric, which is the cheapest hallucination check in the
    project and catches what a careless judge misses.
    """
    known = {source["sid"] for source in state.get("sources", [])}
    seen: list[str] = []
    for sid in CITATION.findall(answer):
        if sid in known and sid not in seen:
            seen.append(sid)
    return seen


def unresolved_citations(state: ResearchState, answer: str) -> list[str]:
    """Ids cited that do not exist. Non-empty means the model invented a source."""
    known = {source["sid"] for source in state.get("sources", [])}
    return sorted({sid for sid in CITATION.findall(answer) if sid not in known})
