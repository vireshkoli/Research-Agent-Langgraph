"""Does the judge agree with a human? Reported the way the 2026 literature asks.

Raw agreement on its own is the standard mistake. Cohen's kappa corrects for the
agreement you would get by chance, and the deflation between the two is large and
universal — the 541k-judgement study behind this design measures 33-41 percentage
points on MT-Bench. Publishing raw agreement alone would flatter the judge by
roughly that much.

Kappa has its own failure mode, though: with skewed marginals (an agent that
succeeds most of the time, so almost everything is labelled 1) it becomes unstable
and can collapse even when agreement is high. Gwet's AC1 is designed for exactly
that case and costs one extra function, so both are reported and the confusion
matrix is printed underneath so a reader can see the marginals for themselves.

Deliberately *not* reported: Pearson, Spearman, Kendall, phi and MCC. For a binary
criterion they all collapse to the same number, so printing five of them is theatre.
"""

import math
import random
from dataclasses import dataclass, field


@dataclass
class Agreement:
    n: int
    raw_agreement: float
    cohens_kappa: float
    kappa_ci: tuple[float, float]
    gwets_ac1: float
    confusion: dict[str, int] = field(default_factory=dict)
    coverage: float = 1.0
    excluded: int = 0

    @property
    def landis_koch(self) -> str:
        """The conventional reading of a kappa value (Landis & Koch, 1977)."""
        k = self.cohens_kappa
        if k < 0.0:
            return "poor"
        if k < 0.21:
            return "slight"
        if k < 0.41:
            return "fair"
        if k < 0.61:
            return "moderate"
        if k < 0.81:
            return "substantial"
        return "almost perfect"


def _confusion(judge: list[int], human: list[int]) -> dict[str, int]:
    return {
        "both_correct": sum(1 for j, h in zip(judge, human, strict=True) if j == 1 and h == 1),
        "judge_correct_human_incorrect": sum(
            1 for j, h in zip(judge, human, strict=True) if j == 1 and h == 0
        ),
        "judge_incorrect_human_correct": sum(
            1 for j, h in zip(judge, human, strict=True) if j == 0 and h == 1
        ),
        "both_incorrect": sum(1 for j, h in zip(judge, human, strict=True) if j == 0 and h == 0),
    }


def cohens_kappa(judge: list[int], human: list[int]) -> float:
    n = len(judge)
    if n == 0:
        return 0.0
    observed = sum(1 for j, h in zip(judge, human, strict=True) if j == h) / n
    # Expected agreement if both raters labelled independently at their own base rates.
    judge_positive, human_positive = sum(judge) / n, sum(human) / n
    expected = judge_positive * human_positive + (1 - judge_positive) * (1 - human_positive)
    if expected >= 1.0:
        # Both raters were unanimous; kappa is undefined, and 1.0 would overstate it.
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def gwets_ac1(judge: list[int], human: list[int]) -> float:
    """Chance-corrected agreement that does not collapse on skewed marginals."""
    n = len(judge)
    if n == 0:
        return 0.0
    observed = sum(1 for j, h in zip(judge, human, strict=True) if j == h) / n
    pi = (sum(judge) + sum(human)) / (2 * n)
    expected = 2 * pi * (1 - pi)
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1 - expected)


def bootstrap_kappa_ci(
    judge: list[int], human: list[int], iterations: int = 2000, seed: int = 42
) -> tuple[float, float]:
    """Percentile bootstrap CI. Seeded, so the reported interval is reproducible."""
    n = len(judge)
    if n < 2:
        return 0.0, 0.0
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        picks = [rng.randrange(n) for _ in range(n)]
        samples.append(cohens_kappa([judge[i] for i in picks], [human[i] for i in picks]))
    samples.sort()
    return samples[int(0.025 * iterations)], samples[int(0.975 * iterations) - 1]


def compare(
    judge_verdicts: dict[tuple[str, int], str], human_labels: dict[tuple[str, int], int]
) -> Agreement | None:
    """Align judge verdicts with human labels and score the agreement.

    Only items with both a human label and a parseable verdict are compared;
    everything else is counted in `excluded` and reported, rather than silently
    dropped or coerced to a value that would move the number.
    """
    judge_side: list[int] = []
    human_side: list[int] = []
    excluded = 0

    for key, label in sorted(human_labels.items()):
        verdict = judge_verdicts.get(key)
        if verdict not in ("CORRECT", "INCORRECT"):
            excluded += 1
            continue
        judge_side.append(1 if verdict == "CORRECT" else 0)
        human_side.append(label)

    if not judge_side:
        return None

    n = len(judge_side)
    return Agreement(
        n=n,
        raw_agreement=sum(1 for j, h in zip(judge_side, human_side, strict=True) if j == h) / n,
        cohens_kappa=cohens_kappa(judge_side, human_side),
        kappa_ci=bootstrap_kappa_ci(judge_side, human_side),
        gwets_ac1=gwets_ac1(judge_side, human_side),
        confusion=_confusion(judge_side, human_side),
        coverage=n / len(human_labels) if human_labels else 0.0,
        excluded=excluded,
    )


def deflation(agreement: Agreement) -> float:
    """Percentage points by which raw agreement overstates kappa.

    Published because reporting raw agreement alone is the mistake the literature
    calls out, and naming the size of the gap is more useful than just avoiding it.
    """
    return (agreement.raw_agreement - agreement.cohens_kappa) * 100


def verbosity_correlation(scores: list[int], lengths: list[int]) -> float:
    """Point-biserial correlation between verdict and answer length.

    A cheap bias probe. The 2026 study measured verbosity bias below 0.011 — small
    enough to contradict the folk wisdom that judges reward length — so this is
    reported as a check rather than assumed to be a problem.
    """
    n = len(scores)
    if n < 2 or len(set(scores)) < 2:
        return 0.0
    mean_length = sum(lengths) / n
    mean_score = sum(scores) / n
    numerator = sum(
        (s - mean_score) * (length - mean_length) for s, length in zip(scores, lengths, strict=True)
    )
    denominator = math.sqrt(
        sum((s - mean_score) ** 2 for s in scores)
        * sum((length - mean_length) ** 2 for length in lengths)
    )
    return numerator / denominator if denominator else 0.0
