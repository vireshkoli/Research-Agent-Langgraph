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

from evals.schema import HUMAN_LABELS, EvalCase, load_cases
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
        ones = sum(1 for item in labels.values() if item["label"] == 1)
        print(f"labelled {len(labels)}/{len(items)}")
        if labels:
            print(f"  correct   {ones} ({ones / len(labels):.0%})")
            print(f"  incorrect {len(labels) - ones} ({1 - ones / len(labels):.0%})")
        return 0

    todo = [
        (case_id, run, path)
        for case_id, run, path in items
        if args.review or f"{case_id}#{run}" not in labels
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
        }
        save(labels)

    save(labels)
    print(f"\nsaved {len(labels)}/{len(items)} labels to {HUMAN_LABELS}")
    if len(labels) < len(items):
        print("run again to continue where you left off.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
