"""FRAMES: a public multi-hop benchmark, sampled large enough that a result is not luck.

The 30 hand-written cases in `evals/dataset.json` answer the question they were
written for, and cannot answer whether one variant beats another. Every variant
scores 100% on the easy tier, so the cases that could separate them are a handful,
and at that size a real difference and a lucky streak look the same.

FRAMES (Krishna et al., 2024; Apache-2.0) is 824 questions, each needing facts from
two to fifteen Wikipedia articles, each with a gold answer and the gold articles
themselves. That last part matters: it lets a run be scored on whether it *found*
the right pages, separately from whether it reasoned correctly once it had them.

**Sample size is chosen, not convenient.** A paired comparison needs about 200
questions to detect a 10-point difference with 80% power at p < 0.05, and at that
size a smaller difference is still bounded — the 95% interval on the gap narrows to
roughly ±7 points. `evals/prereg/frames.md` records the arithmetic, and was committed
before any FRAMES run.

**The sample is stratified and frozen.** Questions are drawn in proportion to their
primary reasoning type, with a fixed seed, and only their ids are committed. The
source file's hash is recorded beside them, so a changed upstream file is an error
rather than a silently different benchmark.
"""

import ast
import csv
import hashlib
import json
import random
from pathlib import Path
from urllib.parse import unquote

from evals.schema import EvalCase

SOURCE = "google/frames-benchmark"
LICENCE = "Apache-2.0"
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache" / "frames" / "test.tsv"
SAMPLE = Path(__file__).parent / "frames_sample.json"
SEED = 20261025
SIZE = 200

RUBRIC_NOTE = (
    "FRAMES benchmark item; the reference is its gold answer. Equivalent wording, number "
    "formatting or unit notation of that answer is CORRECT. An answer that does not commit "
    "to the gold answer — that stops at the facts leading to it, or hedges between "
    "options — is INCORRECT."
)


URL = f"https://huggingface.co/datasets/{SOURCE}/resolve/main/test.tsv"


def download() -> Path:
    """Fetch the benchmark into the gitignored cache. The file itself is never committed.

    The hash check in `load` is what makes fetching from `main` safe: if upstream
    changes, the committed sample stops resolving instead of quietly meaning
    different questions.
    """
    if not CACHE.is_file():
        import httpx

        CACHE.parent.mkdir(parents=True, exist_ok=True)
        response = httpx.get(URL, follow_redirects=True, timeout=60)
        response.raise_for_status()
        CACHE.write_bytes(response.content)
    return CACHE


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows() -> list[dict[str, str]]:
    with download().open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def primary_type(row: dict[str, str]) -> str:
    return row["reasoning_types"].split("|")[0].strip()


def draw(size: int = SIZE, seed: int = SEED) -> dict[str, object]:
    """A stratified sample: each primary reasoning type in proportion to its share.

    Largest-remainder rounding, so the strata sum to exactly `size`. Pure given the
    file and the seed, which is what lets anyone reproduce the committed ids.
    """
    data = rows()
    strata: dict[str, list[int]] = {}
    for row in data:
        strata.setdefault(primary_type(row), []).append(int(row[""]))

    exact = {name: size * len(ids) / len(data) for name, ids in strata.items()}
    quota = {name: int(share) for name, share in exact.items()}
    shortfall = size - sum(quota.values())
    for name in sorted(exact, key=lambda n: exact[n] - quota[n], reverse=True)[:shortfall]:
        quota[name] += 1

    rng = random.Random(seed)
    chosen: list[int] = []
    for name in sorted(strata):
        chosen += rng.sample(sorted(strata[name]), quota[name])

    return {
        "source": SOURCE,
        "licence": LICENCE,
        "source_sha256": _sha256(CACHE),
        "seed": seed,
        "size": size,
        "method": "stratified by primary reasoning type, largest-remainder allocation",
        "strata": dict(sorted(quota.items())),
        "ids": sorted(chosen),
    }


def load(path: Path = SAMPLE) -> list[EvalCase]:
    """The committed sample as eval cases. Refuses if the upstream file has changed."""
    sample = json.loads(path.read_text())
    if _sha256(download()) != sample["source_sha256"]:
        raise ValueError(
            f"{SOURCE} has changed since the sample was drawn; the committed ids no "
            "longer name the same questions. Redraw and re-register before running."
        )
    by_id = {int(row[""]): row for row in rows()}
    return [_case(by_id[index]) for index in sample["ids"]]


def _case(row: dict[str, str]) -> EvalCase:
    links = ast.literal_eval(row["wiki_links"]) if row["wiki_links"].strip() else []
    types = [part.strip() for part in row["reasoning_types"].split("|") if part.strip()]
    return EvalCase(
        id=f"frames-{int(row['']):03d}",
        tier="frames",
        question=row["Prompt"].strip(),
        reference_answer=row["Answer"].strip(),
        must_include=[],
        judge_rubric_notes=RUBRIC_NOTE,
        tags=types,
        gold_links=[str(link) for link in links],
    )


def gold_source_recall(case: EvalCase, sources: list[dict[str, object]]) -> float | None:
    """Share of the case's gold articles that the run actually retrieved.

    Separates "never found the page" from "found it and reasoned wrong" — the two
    have different fixes, and a pass rate cannot tell them apart.
    """
    if not case.gold_links:
        return None
    found = {_canonical(str(source.get("url", ""))) for source in sources}
    return sum(_canonical(link) in found for link in case.gold_links) / len(case.gold_links)


def _canonical(url: str) -> str:
    """Wikipedia URLs compare equal across scheme, mobile host, fragment and case."""
    url = url.strip().split("#")[0].split("?")[0].rstrip("/")
    url = url.replace("http://", "https://").replace("://en.m.wikipedia.org", "://en.wikipedia.org")
    if "/wiki/" in url:
        head, title = url.split("/wiki/", 1)
        title = unquote(title).replace(" ", "_")
        url = f"{head.lower()}/wiki/{title[:1].upper()}{title[1:]}"
    return url


if __name__ == "__main__":
    print(json.dumps(draw(), indent=2))
