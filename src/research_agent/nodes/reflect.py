"""reflect — decide whether the run can stop.

This node exists for one reason: **to catch premature termination.** In a loop where
`act` decides everything, the stop condition is "the model emitted no tool call",
and small models get that wrong in both directions — they bail after one search, or
they loop on query variants forever. "Should I stop?" and "what do I do next?" are
genuinely different questions, and the first has a much smaller input.

Kept cheap and gated so it earns its cost:

- It sees a one-line-per-step digest, not the raw scratchpad — roughly 1-2k tokens
  against act's 5-7k.
- It is skipped on the first iteration; no research question worth asking is
  finished in one hop.
- `reflect_overrules` counts the times it turned a proposed stop into another
  round. That number, measured over the eval, is what justifies the node — which is
  why the `no_overrule` variant exists.

**It did not justify itself.** The count came out at 35 over 90 runs, so the node
fires; what the traces then showed is that firing is not the same as being right.
A third of all rounds call no tools at all, because `act` said it could answer and
reflect sent it back to say so again, and one adversarial case repeats that eight
times. The node was keeping a stop decision in play long after the run had stopped
producing evidence. The `gated` variant caps the overrules at
`max_reflect_overrules` for that reason: reflect gets its chance to catch a
premature stop, but not to argue with the same one indefinitely.
"""

from pydantic import BaseModel, Field

from research_agent.config import Settings, settings
from research_agent.llm import CostTracker, QueryBudgetExceeded, parse
from research_agent.nodes import call_records, make_step, spend_delta
from research_agent.prompts import REFLECT_SYSTEM, reflect_user
from research_agent.state import ReflectDecision, ResearchState


class ReflectOutput(BaseModel):
    """Field order matters: structured outputs preserve it, so the model works
    through coverage before committing to a decision."""

    reasoning: str = Field(description="One or two sentences on what is and is not covered.")
    covered: list[str] = Field(description="Sub-questions the observations actually answer.")
    open_gaps: list[str] = Field(description="Sub-questions still unanswered.")
    decision: ReflectDecision = Field(description="continue, replan, or finalize.")


def reflect_node(
    state: ResearchState, tracker: CostTracker, cfg: Settings | None = None
) -> ResearchState:
    cfg = cfg or settings()
    seen = len(tracker.calls)
    plan = state.get("plan") or []
    proposed_stop = bool(state.get("act_requested_stop"))

    # Skipped on the first round when act wants to continue: reflect could only agree,
    # and paying for a coverage check that cannot change the outcome is waste. Note
    # this is *not* a claim that one hop is never enough — the baseline variant answers
    # in one hop and outscores this agent. A proposed stop on round one is still
    # checked, which is why the condition is conjunctive.
    if state.get("step", 0) <= 1 and not proposed_stop:
        return {
            "reflect_decision": "continue",
            "scratchpad": [make_step(state, "reflect", note="skipped on first step")],
        }

    try:
        result = parse(
            REFLECT_SYSTEM,
            reflect_user(state),
            ReflectOutput,
            model=cfg.agent_model,
            reasoning_effort=cfg.plan_reasoning_effort or None,
            purpose="reflect",
            tracker=tracker,
        )
    except QueryBudgetExceeded:
        return {
            "early_exit_reason": "budget_usd",
            "reflect_decision": "finalize",
            "llm_calls": call_records(tracker, seen),
            "spend_usd": spend_delta(tracker, seen),
            "scratchpad": [make_step(state, "reflect", note="budget exceeded during reflect")],
        }

    if result is None:
        # Unparseable output is a worse decision, not a dead run: continue if there
        # is budget left to continue with, otherwise answer.
        return {
            "reflect_decision": "continue" if not proposed_stop else "finalize",
            "reflections": 1,
            "llm_calls": call_records(tracker, seen),
            "spend_usd": spend_delta(tracker, seen),
            "scratchpad": [make_step(state, "reflect", note="unparseable reflection")],
        }

    covered = {question: question in set(result.covered) for question in plan}
    gaps = [gap for gap in result.open_gaps if gap in set(plan)] or [
        question for question, done in covered.items() if not done
    ]
    decision: ReflectDecision = result.decision

    # The measurement that justifies this node existing: act wanted to stop and
    # reflect sent it back for more evidence.
    overruled = proposed_stop and decision == "continue"

    return {
        "covered": covered,
        "open_gaps": gaps,
        "reflect_decision": decision,
        "reflections": 1,
        "reflect_overrules": 1 if overruled else 0,
        # act's stop is consumed here; leaving it set would make the next router
        # treat a fresh round as another proposed stop.
        "act_requested_stop": False,
        "llm_calls": call_records(tracker, seen),
        "spend_usd": spend_delta(tracker, seen),
        "scratchpad": [
            make_step(
                state,
                "reflect",
                reflection={
                    "decision": decision,
                    "reasoning": result.reasoning,
                    "covered": covered,
                    "open_gaps": gaps,
                    "overruled_stop": overruled,
                },
            )
        ],
    }
