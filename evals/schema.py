"""The evaluation case format, validated on load.

One rule governs this dataset above all others: **every case carries at least one
deterministic `must_include` anchor.** If a correct answer cannot be pinned to a
string that must appear in it, the case is unscoreable and the judge ends up
measuring its own taste rather than the agent's behaviour. Cases that cannot meet
that bar are cut rather than kept and hoped for. The loader enforces it, so the rule
cannot quietly erode.

Human labels live in a separate file. The dataset is the input contract; labels are
measurements. Keeping them together would mean editing a question silently
invalidates the label attached to it.
"""

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

DATASET = Path(__file__).parent / "dataset.json"
HUMAN_LABELS = Path(__file__).parent / "human_labels.json"

Tier = Literal["easy", "multi_hop", "adversarial"]
MatcherName = Literal[
    "exact", "contains_any", "contains_all", "regex", "numeric_close", "lte", "gte", "any"
]


class ArgMatcher(BaseModel):
    """How one tool argument is compared.

    Structural, field-by-field comparison with per-field semantics — the BFCL idea —
    rather than `json.dumps(actual) == json.dumps(expected)`, which fails on key
    order, whitespace and 5 vs "5" while telling you nothing about why.
    """

    matcher: MatcherName
    value: Any = None
    tolerance: float = 0.01  # numeric_close only, relative


class ExpectedTool(BaseModel):
    name: str
    args: dict[str, ArgMatcher] = Field(default_factory=dict)
    required: bool = True


class ExpectedBehavior(BaseModel):
    should_answer: bool = True
    should_abstain: bool = False
    should_early_exit: bool = False
    # None inside the list means "finishing normally is also acceptable".
    allowed_early_exit_reasons: list[str | None] = Field(default_factory=list)
    must_cite: bool = True
    min_citations: int = 1


class EvalCase(BaseModel):
    id: str
    tier: Tier
    question: str
    reference_answer: str
    must_include: list[str]
    must_not_include: list[str] = Field(default_factory=list)
    expected_tools: list[ExpectedTool] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    tool_order: list[str] | None = None
    min_steps: int = 1
    strict_args: bool = False
    expected_behavior: ExpectedBehavior = Field(default_factory=ExpectedBehavior)
    judge_rubric_notes: str = ""
    tags: list[str] = Field(default_factory=list)

    @field_validator("must_include")
    @classmethod
    def _needs_an_anchor(cls, value: list[str], info: Any) -> list[str]:
        # Adversarial cases are the one exception: "refuses to fabricate a list" has
        # no string that must appear, so they are pinned by must_not_include instead.
        if not value and info.data.get("tier") != "adversarial":
            raise ValueError(
                "every non-adversarial case needs at least one must_include anchor; "
                "if you cannot write one the case is unscoreable and should be cut"
            )
        return value

    @model_validator(mode="after")
    def _adversarial_cases_need_some_anchor(self) -> "EvalCase":
        if self.tier == "adversarial" and not (self.must_include or self.must_not_include):
            raise ValueError(
                f"{self.id}: an adversarial case still needs must_include or "
                "must_not_include, otherwise nothing distinguishes pass from fail"
            )
        return self


class HumanLabel(BaseModel):
    """One hand judgement, keyed by case id and run index.

    Labelled per *run*, not per case: the judge produces one verdict per run, so
    agreement has to be computed over the same unit. Labelling only the cases would
    give kappa on a third of the sample, where the standard error spans several
    Landis-Koch bands.
    """

    case_id: str
    run: int
    label: Literal[0, 1]
    note: str = ""


def load_cases(path: Path = DATASET) -> list[EvalCase]:
    """Load and validate the dataset. Fails loudly on drift rather than skipping."""
    raw = json.loads(path.read_text())
    cases = [EvalCase.model_validate(item) for item in raw]

    ids = [case.id for case in cases]
    duplicates = {name for name in ids if ids.count(name) > 1}
    if duplicates:
        raise ValueError(f"duplicate case ids: {sorted(duplicates)}")
    return cases


def load_human_labels(path: Path = HUMAN_LABELS) -> dict[tuple[str, int], int]:
    """{(case_id, run): label}. Missing file is fine — agreement is then skipped."""
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text())
    labels = [HumanLabel.model_validate(item) for item in raw]
    return {(label.case_id, label.run): label.label for label in labels}


def tier_counts(cases: list[EvalCase]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.tier] = counts.get(case.tier, 0) + 1
    return counts
