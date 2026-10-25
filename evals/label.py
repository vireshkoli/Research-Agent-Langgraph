"""Hand-label agent answers, so the judge can be validated against a human.

    python -m evals.label            label the unlabelled ones, resuming where you left off
    python -m evals.label --review   re-read what you already labelled
    python -m evals.label --stats    progress and the label balance

**The judge's verdict is never shown.** Seeing it first is the single easiest way
to contaminate the agreement statistic — anchoring on a verdict turns Cohen's kappa
from a measurement into a confirmation. This tool cannot display it even with a
flag, which is deliberate.

Labels are written to `evals/human_labels.json` after every answer, so this can be
done in several sittings without losing work. Labelling is per *run*, not per case:
the judge produces one verdict per run, so agreement has to be computed over the
same unit. Labelling only the 30 cases would give kappa on a third of the sample,
where the standard error spans several Landis-Koch bands.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evals.schema import HUMAN_LABELS, EvalCase, answer_fingerprint, load_cases
from research_agent.trace import load_trace

TRACES = Path(__file__).parent / "results" / "traces" / "full"


Label = dict[str, Any]


def existing() -> dict[str, Label]:
    if not HUMAN_LABELS.is_file():
        return {}
    return {
        f"{item['case_id']}#{item['run']}": item for item in json.loads(HUMAN_LABELS.read_text())
    }


def save(labels: dict[str, Label]) -> None:
    ordered = sorted(labels.values(), key=lambda item: (item["case_id"], item["run"]))
    HUMAN_LABELS.write_text(json.dumps(ordered, indent=2) + "\n")


def available() -> list[tuple[str, int, Path]]:
    if not TRACES.is_dir():
        return []
    items = []
    for path in sorted(TRACES.glob("*-r*.json")):
        case_id, _, run = path.stem.rpartition("-r")
        items.append((case_id, int(run), path))
    return items


def show(case: EvalCase, answer: str, index: int, total: int) -> None:
    rule = "─" * 78
    print(f"\n{rule}")
    print(f"[{index}/{total}]  {case.id}  ({case.tier})")
    print(rule)
    print(f"\nQUESTION\n  {case.question}")
    print(f"\nREFERENCE ANSWER\n  {case.reference_answer}")
    if case.judge_rubric_notes:
        print(f"\nGRADING NOTES\n  {case.judge_rubric_notes}")
    print(f"\n{rule}\nAGENT'S ANSWER\n{rule}")
    print(answer.strip() or "(the agent produced no answer)")
    print(rule)


def prompt() -> str:
    while True:
        raw = (
            input("\n  1 = correct   0 = incorrect   s = skip   q = save and quit\n  > ")
            .strip()
            .lower()
        )
        if raw in ("1", "0", "s", "q"):
            return raw
        print("  please answer 1, 0, s or q")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.label", description=__doc__)
    parser.add_argument("--review", action="store_true", help="Re-read labelled items.")
    parser.add_argument("--stats", action="store_true", help="Progress only.")
    args = parser.parse_args(argv)

    cases = {case.id: case for case in load_cases()}
    items = available()
    labels = existing()

    if not items:
        print("no traces found — run `python -m evals.run` first", file=sys.stderr)
        return 1

    if args.stats:
        live = {
            f"{c}#{r}": labels[f"{c}#{r}"]
            for c, r, path in items
            if f"{c}#{r}" in labels
            and labels[f"{c}#{r}"].get("answer_sha256")
            == answer_fingerprint(load_trace(path).answer)
        }
        ones = sum(1 for item in live.values() if item["label"] == 1)
        print(f"labelled {len(live)}/{len(items)} of the answers committed now")
        if stale := len(labels) - len(live):
            print(f"  {stale} older labels grade answers that have since been re-run")
        if live:
            print(f"  correct   {ones} ({ones / len(live):.0%})")
            print(f"  incorrect {len(live) - ones} ({1 - ones / len(live):.0%})")
        return 0

    def current(case_id: str, run: int, path: Path) -> bool:
        """A label only counts as done if it graded the answer committed now.

        Keys are reused by every re-run, so "a label exists for (case, run)" says
        nothing about whether a human ever read *this* answer. Without this check a
        re-run leaves every label stale and this tool reports all of them done.
        """
        label = labels.get(f"{case_id}#{run}")
        if label is None:
            return False
        return label.get("answer_sha256") == answer_fingerprint(load_trace(path).answer)

    stale = sum(1 for c, r, path in items if f"{c}#{r}" in labels and not current(c, r, path))
    if stale:
        print(
            f"{stale} existing labels grade answers that have since been re-run; "
            "they are queued for re-labelling."
        )

    todo = [
        (case_id, run, path)
        for case_id, run, path in items
        if args.review or not current(case_id, run, path)
    ]
    if not todo:
        print(f"all {len(items)} runs already labelled. `--review` to go through them again.")
        return 0

    print(f"{len(todo)} to label ({len(labels)}/{len(items)} done).")
    print("The judge's verdict is deliberately not shown — seeing it first would")
    print("contaminate the agreement statistic. Progress saves after every answer.")

    for position, (case_id, run, path) in enumerate(todo, start=1):
        case = cases.get(case_id)
        if case is None:
            continue
        trace = load_trace(path)
        show(case, trace.answer, position, len(todo))

        choice = prompt()
        if choice is None or choice == "q":
            break
        if choice == "s":
            continue

        labels[f"{case_id}#{run}"] = {
            "case_id": case_id,
            "run": run,
            "label": int(choice),
            "note": "",
            "answer_sha256": answer_fingerprint(trace.answer),
        }
        save(labels)

    save(labels)
    print(f"\nsaved {len(labels)}/{len(items)} labels to {HUMAN_LABELS}")
    if len(labels) < len(items):
        print("run again to continue where you left off.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
