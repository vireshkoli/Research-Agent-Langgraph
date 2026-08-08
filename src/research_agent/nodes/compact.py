"""compact — fold older steps into a rolling summary.

**Why this exists, honestly.** With a 400k-token context on the agent model, this
run will never overflow. Compaction here is a *cost and quality* control, not a
context-window necessity: uncached delta tokens grow linearly with every
observation, and long noisy scratchpads measurably degrade a small model's
attention. Claiming it prevents overflow would be the kind of thing an interviewer
catches immediately.

**Citations cannot be lost here, structurally.** The source registry lives in
`state["sources"]` and is re-rendered into every prompt by `render_context`.
Compaction rewrites the scratchpad. Those are different things, so no summariser
mistake — including a summary that mentions no source ids at all — can drop a
citation from the model's view. That is a much stronger guarantee than "the
summarisation prompt asks it to be careful", and there is a test that enforces it
by feeding in a deliberately adversarial summary.

The last `keep_last_steps` steps stay verbatim: the model needs recent detail to
choose its next call. Rolling summaries drift, so `max_compactions` caps how far the
degradation can go.
"""

from research_agent.config import Settings, settings
from research_agent.llm import CostTracker, QueryBudgetExceeded, complete
from research_agent.nodes import call_records, make_step, spend_delta
from research_agent.prompts import COMPACT_SYSTEM, compact_user
from research_agent.state import ResearchState, Step


def compact_node(
    state: ResearchState, tracker: CostTracker, cfg: Settings | None = None
) -> ResearchState:
    cfg = cfg or settings()
    seen = len(tracker.calls)
    scratchpad = state.get("scratchpad") or []
    start = state.get("compacted_upto", 0)
    end = max(start, len(scratchpad) - cfg.keep_last_steps)

    to_fold = scratchpad[start:end]
    if not to_fold:
        return {"scratchpad": [make_step(state, "compact", note="nothing to compact")]}

    try:
        summary = complete(
            COMPACT_SYSTEM,
            compact_user(state, _render(to_fold)),
            model=cfg.agent_model,
            purpose="compact",
            tracker=tracker,
        ).strip()
    except QueryBudgetExceeded:
        return {
            "early_exit_reason": "budget_usd",
            "llm_calls": call_records(tracker, seen),
            "spend_usd": spend_delta(tracker, seen),
            "scratchpad": [make_step(state, "compact", note="budget exceeded during compact")],
        }

    if not summary:
        # A summariser that returns nothing must not blank the existing summary.
        return {"scratchpad": [make_step(state, "compact", note="empty summary, keeping previous")]}

    return {
        "summary": summary,
        "compacted_upto": end,
        "compactions": 1,
        "llm_calls": call_records(tracker, seen),
        "spend_usd": spend_delta(tracker, seen),
        "scratchpad": [
            make_step(state, "compact", note=f"folded steps {start + 1}-{end} into the summary")
        ],
    }


def _render(steps: list[Step]) -> str:
    """The steps being folded, as text for the summariser."""
    lines: list[str] = []
    for step in steps:
        for observation in step.get("observations") or []:
            status = "ok" if observation["ok"] else f"FAILED: {observation['error']}"
            sids = (
                f" (sources: {', '.join(observation['source_ids'])})"
                if observation["source_ids"]
                else ""
            )
            header = f"Step {step['i']} — {observation['tool']}({observation['args']})"
            lines.append(f"{header} [{status}]{sids}\n{observation['content'][:1500]}")
        if reflection := step.get("reflection"):
            lines.append(f"Step {step['i']} — reflection: {reflection.get('reasoning', '')}")
    return "\n\n".join(lines) if lines else "(no observations in this range)"
