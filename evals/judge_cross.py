"""The cross-family judge: the same rubric, a different model family.

**Why this module exists separately.** `evals/judge.py` is the OpenAI judge and
stays that way. Both judges must grade against a *byte-identical* rubric and
return the same schema, or a disagreement between them measures the prompt rather
than the model — so `JUDGE_SYSTEM`, `judge_user` and `Judgement` are imported from
there rather than restated here. The only thing that differs is which company's
model reads them.

**What it buys.** The README has carried "self-preference is not controlled" as a
limitation since the first evaluation: the agent is an OpenAI model and so is its
judge. They are different generations, not different families, and no amount of
agreement between a model and itself rules out a shared bias. Re-grading the
committed trajectories with a judge from another provider and reporting the
agreement between the two turns that disclaimer into a number.

**What it is not.** It is not a replacement judge and the committed verdicts are
not regraded in place. The OpenAI judge remains the one the published pass rates
come from; this is a second opinion on the same evidence, reported beside it.

**Isolation.** Nothing in `src/` imports this, `anthropic` is a dev-group
dependency, and `scripts/requirements.py` writes the Space's requirements from
`project.dependencies` only — so the agent, the Docker image and the live demo
behave identically whether this module, its dependency or its key exist at all.
A missing key is a skipped section in the report, never a failed run.
"""

import os
from dataclasses import dataclass

from evals.judge import JUDGE_SYSTEM, Judgement, judge_user
from research_agent.llm import _load_env_file

# Claude Opus 5.5, USD per million tokens. Kept beside the OpenAI PRICES table in
# llm.py in spirit but not in place: that module is the agent's single OpenAI
# wrapper and this cost never touches the agent's budgets or the project ledger,
# which bound OpenAI spend specifically.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

DEFAULT_MODEL = "claude-opus-5-5"


class CrossJudgeUnavailable(RuntimeError):
    """Raised when the key or the package is missing. Callers skip, never crash."""


@dataclass
class CrossVerdict:
    """One judgement, plus what it cost to obtain it."""

    verdict: str
    failure_mode: str
    reasoning: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in PRICES:
        return 0.0
    price_in, price_out = PRICES[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def available() -> bool:
    """True when a cross-family judgement could actually be made."""
    _load_env_file()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _client() -> object:
    _load_env_file()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise CrossJudgeUnavailable(
            "ANTHROPIC_API_KEY is not set — see .env.example. The cross-family "
            "judge is optional; every other part of the evaluation runs without it."
        )
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - dev dependency is installed in CI
        raise CrossJudgeUnavailable(
            "the `anthropic` package is not installed — it lives in the dev group, "
            "so `uv sync --dev` is what brings it in."
        ) from exc

    # The SDK already retries 429 and 5xx with exponential backoff; raising the
    # count rather than writing a retry loop keeps one behaviour in one place.
    return anthropic.Anthropic(max_retries=5)


def judge_cross(
    question: str,
    reference: str,
    answer: str,
    notes: str = "",
    model: str = DEFAULT_MODEL,
    effort: str = "medium",
) -> CrossVerdict | None:
    """Grade one answer. Returns None if the model declined to produce a verdict.

    None is data, exactly as it is for the OpenAI judge: an unparseable or refused
    judgement is reported and excluded rather than coerced to INCORRECT, because
    coercion biases the comparison in whichever direction it goes.
    """
    client = _client()

    response = client.messages.parse(  # type: ignore[attr-defined]
        model=model,
        max_tokens=4096,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": judge_user(question, reference, answer, notes)}],
        # Opus 5.5 defaults to medium rather than high, so it is set explicitly:
        # a reader should not have to know the model's default to know what ran.
        output_config={"effort": effort},
        output_format=Judgement,
    )

    # Safety classifiers can decline a request outright. That is a non-verdict, not
    # an INCORRECT, and it is counted separately.
    if getattr(response, "stop_reason", None) == "refusal":
        return None

    parsed = getattr(response, "parsed_output", None)
    if parsed is None:
        return None

    usage = response.usage
    return CrossVerdict(
        verdict=parsed.verdict,
        failure_mode=parsed.failure_mode,
        reasoning=parsed.reasoning,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=cost_usd(model, usage.input_tokens, usage.output_tokens),
    )
