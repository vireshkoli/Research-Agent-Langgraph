"""LLM-as-judge: binary verdict, anchored rubric, majority of three.

**Binary, not a 0-1 score.** The agreement literature this project cites is about
binary criteria; a hand label is naturally binary; and majority-of-three on a binary
*is* the self-consistency being sought. Continuous scores from an LLM are badly
calibrated and invite fake precision like "0.87 faithfulness".

**Three samples at temperature 1.0, not three at 0.** Three samples at temperature 0
are identical roughly 95% of the time — that is cargo-culted self-consistency, not
measurement. At 1.0 the disagreement rate is real information, and it gets published:
if the judge disagrees with itself more than ~10% of the time, the *rubric* is
ambiguous and the fix is the rubric, not more samples.

**Field order is load-bearing.** Structured outputs preserve declaration order, so
`reasoning` before `verdict` before `failure_mode` forces the model to work through
the evidence before committing. The failure_mode enum costs nothing extra and turns
a bare pass rate into "of 21 failures, 11 were missing_key_fact".

The judge runs through the Batch API where possible: eval scoring has no latency
requirement, and Batch is half price.
"""

import time
from collections import Counter
from typing import Literal

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from pydantic import BaseModel, Field

from research_agent.config import Settings, settings
from research_agent.llm import CostTracker, parse

Verdict = Literal["CORRECT", "INCORRECT"]
FailureMode = Literal[
    "none",
    "missing_key_fact",
    "wrong_value",
    "fabricated_source",
    "premature_stop",
    "unbounded_or_padded",
    "refused_when_answerable",
    "other",
]

JUDGE_SYSTEM = """\
You are grading a research agent's answer against a reference answer.

Return CORRECT only if the answer is factually right on every point the question \
asked for. Otherwise return INCORRECT.

Grade the substance, not the presentation:
- Wording need not match the reference. Different phrasing of the same fact is CORRECT.
- Extra correct detail is fine. Extra *incorrect* detail is INCORRECT.
- A number that is wrong, or a name that is wrong, is INCORRECT even if the rest is right.
- An answer that honestly states it could not establish something is CORRECT **only \
if** the question was genuinely unanswerable or the reference says so. Giving up on \
an answerable question is INCORRECT.
- Fabricating a source, a URL or a citation is INCORRECT regardless of whether the \
underlying fact happens to be right.
- Length is irrelevant. Do not reward thoroughness or penalise brevity.

Worked examples:

Q: Who won the 2024 Nobel Prize in Physics?
Reference: John J. Hopfield and Geoffrey E. Hinton.
Answer: "Hopfield and Hinton won it for work on neural networks [S1]."
-> CORRECT (both named; extra detail is accurate)

Q: Who won the 2024 Nobel Prize in Physics?
Answer: "Geoffrey Hinton won the 2024 prize [S1]."
-> INCORRECT, missing_key_fact (Hopfield omitted; the question asked who won)

Q: List every 2025 paper citing 'Attention Is All You Need'.
Reference: Unbounded request; the correct behaviour is to say so.
Answer: "Here is the complete list: [five invented titles]"
-> INCORRECT, unbounded_or_padded

Q: List every 2025 paper citing 'Attention Is All You Need'.
Answer: "This cannot be enumerated — the paper has tens of thousands of citations \
and no complete list is retrievable [S1]. Here is where such lists live."
-> CORRECT (acknowledges the scope problem instead of fabricating)"""


class Judgement(BaseModel):
    """Order matters: structured outputs preserve it, so reasoning comes first."""

    reasoning: str = Field(
        description="One or two sentences weighing the answer against the reference."
    )
    verdict: Verdict = Field(description="CORRECT or INCORRECT.")
    failure_mode: FailureMode = Field(description="'none' when CORRECT.")


def judge_user(question: str, reference: str, answer: str, notes: str = "") -> str:
    parts = [
        f"Question:\n{question}",
        f"Reference answer:\n{reference}",
        f"Agent's answer:\n{answer or '(the agent produced no answer)'}",
    ]
    if notes:
        parts.append(f"Case-specific grading notes:\n{notes}")
    return "\n\n".join(parts)


def judge_once(
    question: str,
    reference: str,
    answer: str,
    notes: str = "",
    model: str | None = None,
    cfg: Settings | None = None,
    tracker: CostTracker | None = None,
    batch: bool = True,
    attempts: int = 3,
) -> Judgement | None:
    """One judgement, retrying transient transport failures.

    A DNS hiccup or a dropped connection is not a judgement — it is noise, and
    letting it propagate once killed an entire 90-run evaluation on run five. It
    retries with backoff and, if the network is genuinely down, returns None so the
    caller records "unparseable" and carries on. The count of those is published.
    """
    cfg = cfg or settings()

    for attempt in range(attempts):
        try:
            return parse(
                JUDGE_SYSTEM,
                judge_user(question, reference, answer, notes),
                Judgement,
                model=model or cfg.judge_model,
                purpose="judge",
                tracker=tracker,
                batch=batch,
            )
        except (APIConnectionError, APITimeoutError, RateLimitError, APIStatusError):
            if attempt == attempts - 1:
                return None
            time.sleep(2**attempt)
    return None


def judge(
    question: str,
    reference: str,
    answer: str,
    notes: str = "",
    samples: int = 1,
    model: str | None = None,
    cfg: Settings | None = None,
    tracker: CostTracker | None = None,
) -> tuple[str, str | None, float]:
    """Return (verdict, failure_mode, disagreement_rate).

    An unparseable judgement is reported as "unparseable" and *excluded* from the
    metric rather than coerced to INCORRECT — coercing would bias the result in the
    direction of whichever way the coercion went, and the count is published instead.
    """
    verdicts: list[str] = []
    modes: list[str] = []

    for _ in range(max(1, samples)):
        result = judge_once(
            question, reference, answer, notes, model=model, cfg=cfg, tracker=tracker
        )
        if result is None:
            verdicts.append("unparseable")
        else:
            verdicts.append(result.verdict)
            modes.append(result.failure_mode)

    usable = [v for v in verdicts if v != "unparseable"]
    if not usable:
        return "unparseable", None, 0.0

    counts = Counter(usable)
    verdict, top = counts.most_common(1)[0]
    disagreement = 1.0 - top / len(usable)
    mode = Counter(modes).most_common(1)[0][0] if modes else None
    return verdict, (None if verdict == "CORRECT" else mode), disagreement
