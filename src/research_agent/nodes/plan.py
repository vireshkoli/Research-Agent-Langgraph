"""plan — decompose the question into sub-questions.

This node earns its place because its output is a **data structure**, not because
planning is inherently good. Three other subsystems consume the sub-question list:
`reflect`'s coverage check, the eval's coverage metric, and the UI's progress
display. If planning were left implicit in the system prompt it would live as prose
inside an assistant message, and none of those could read it.

It is re-entrant: `reflect` can route back here once (capped by `max_replans`) when
the original decomposition turns out to be wrong. On that second pass the plan is
*replaced*, which is why `plan` and `covered` are reducer-less in the state schema.

An unparseable response falls back to treating the whole question as one
sub-question. That is a worse plan, not a dead run.
"""

from pydantic import BaseModel, Field

from research_agent.config import Settings, settings
from research_agent.llm import CostTracker, QueryBudgetExceeded, parse
from research_agent.nodes import call_records, make_step, spend_delta
from research_agent.prompts import PLAN_SYSTEM
from research_agent.state import ResearchState

MAX_SUBQUESTIONS = 5


class PlanOutput(BaseModel):
    """`reasoning` is first on purpose: structured outputs preserve field order, so
    the model reasons before it commits to a decomposition."""

    reasoning: str = Field(description="Why this decomposition, in one or two sentences.")
    subquestions: list[str] = Field(description="1-5 independently searchable sub-questions.")


def plan_node(
    state: ResearchState, tracker: CostTracker, cfg: Settings | None = None
) -> ResearchState:
    cfg = cfg or settings()
    seen = len(tracker.calls)
    is_replan = state.get("plan_version", 0) > 0

    user = state["question"]
    if is_replan:
        gaps = "\n".join(f"- {g}" for g in state.get("open_gaps", []))
        user = (
            f"{state['question']}\n\n"
            f"A previous decomposition did not work. It was:\n"
            + "\n".join(f"- {q}" for q in state.get("plan", []))
            + f"\n\nWhat is still missing:\n{gaps}\n\nProduce a better decomposition."
        )

    try:
        result = parse(
            PLAN_SYSTEM,
            user,
            PlanOutput,
            model=cfg.agent_model,
            reasoning_effort=cfg.plan_reasoning_effort or None,
            purpose="plan",
            tracker=tracker,
        )
    except QueryBudgetExceeded:
        # Out of money before the first tool call. Still produce a usable plan so
        # finalize has something to structure its answer around.
        return _result(
            state,
            [state["question"]],
            tracker,
            seen,
            note="budget exceeded during plan",
            early_exit="budget_usd",
        )

    subquestions = _clean(result.subquestions if result else [], state["question"])
    return _result(state, subquestions, tracker, seen, note="replan" if is_replan else None)


def _clean(subquestions: list[str], question: str) -> list[str]:
    """Trim, drop blanks and duplicates, cap the count.

    A model that returns 12 sub-questions would spend the whole step budget on
    searches before answering any of them.
    """
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in subquestions:
        text = (item or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            cleaned.append(text)
    return cleaned[:MAX_SUBQUESTIONS] or [question]


def _result(
    state: ResearchState,
    subquestions: list[str],
    tracker: CostTracker,
    seen: int,
    note: str | None = None,
    early_exit: str | None = None,
) -> ResearchState:
    update: ResearchState = {
        "plan": subquestions,
        "plan_version": state.get("plan_version", 0) + 1,
        # Replaced wholesale: a merge would resurrect coverage from a dead plan.
        "covered": {q: False for q in subquestions},
        "open_gaps": list(subquestions),
        "reflect_decision": None,
        "replans": 1 if state.get("plan_version", 0) > 0 else 0,
        "llm_calls": call_records(tracker, seen),
        "spend_usd": spend_delta(tracker, seen),
        "scratchpad": [make_step(state, "plan", note=note)],
    }
    if early_exit:
        update["early_exit_reason"] = early_exit  # type: ignore[typeddict-item]
    return update
