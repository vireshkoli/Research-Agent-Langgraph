"""A persistent ledger of every dollar this project has spent on the OpenAI API.

`CostTracker` bounds a single run and the process guard bounds a single process.
Neither survives a restart, so neither can answer the question that actually matters
on a tight budget: *how much has this project cost me in total?*

This appends every call to `.spend.json` and refuses to make more once a hard
ceiling is reached. Development is a few hundred short runs across a few hundred
processes; without a persistent counter the honest answer to "what have I spent" is
a shrug and a look at the billing dashboard tomorrow.

The ledger is also what backs the README's "total spend to build and evaluate this
project" line, which is a real number rather than a reconstruction.
"""

import json
import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

LEDGER_ENV = "RA_SPEND_LEDGER"
DEFAULT_LEDGER = Path(".spend.json")

_lock = threading.Lock()


class ProjectBudgetExceeded(Exception):
    """Total project spend has reached RA_MAX_PROJECT_COST_USD.

    Deliberately not catchable by the agent's own budget handling: this is a stop
    sign for the developer, not a condition the graph should degrade around.
    """


@dataclass(frozen=True)
class Ledger:
    total_usd: float
    calls: int
    by_purpose: dict[str, float]
    updated: str


def ledger_path() -> Path:
    return Path(os.environ.get(LEDGER_ENV) or DEFAULT_LEDGER)


def read() -> Ledger:
    path = ledger_path()
    if not path.is_file():
        return Ledger(total_usd=0.0, calls=0, by_purpose={}, updated="never")
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        # A corrupt ledger must not block work; it under-reports rather than crashing.
        return Ledger(total_usd=0.0, calls=0, by_purpose={}, updated="unreadable")
    return Ledger(
        total_usd=float(raw.get("total_usd", 0.0)),
        calls=int(raw.get("calls", 0)),
        by_purpose=dict(raw.get("by_purpose", {})),
        updated=str(raw.get("updated", "unknown")),
    )


def record(amount_usd: float, purpose: str, cap_usd: float) -> float:
    """Add `amount_usd` to the ledger and return the new total.

    Raises ProjectBudgetExceeded *after* recording, so the ledger stays truthful
    about what was actually spent even on the call that crossed the line.
    """
    with _lock:
        current = read()
        by_purpose = dict(current.by_purpose)
        by_purpose[purpose] = round(by_purpose.get(purpose, 0.0) + amount_usd, 8)
        total = round(current.total_usd + amount_usd, 8)
        payload = {
            "total_usd": total,
            "calls": current.calls + 1,
            "by_purpose": by_purpose,
            "updated": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        path = ledger_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename so an interrupted write cannot truncate the ledger.
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2))
            temporary.replace(path)
        except OSError:
            pass  # an unwritable ledger must not stop the work

    if cap_usd > 0 and total > cap_usd:
        raise ProjectBudgetExceeded(
            f"total project spend ${total:.4f} has reached the ${cap_usd:.2f} ceiling "
            f"(RA_MAX_PROJECT_COST_USD). Raise it deliberately in .env to continue."
        )
    return total


def summary() -> str:
    """One-line report, printed by the CLI and the eval runner."""
    current = read()
    if not current.calls:
        return "Project spend: $0.0000 (no calls recorded)"
    parts = ", ".join(
        f"{purpose} ${amount:.4f}"
        for purpose, amount in sorted(current.by_purpose.items(), key=lambda kv: -kv[1])
    )
    return (
        f"Project spend: ${current.total_usd:.4f} across {current.calls} calls "
        f"[{parts}] as of {current.updated}"
    )
