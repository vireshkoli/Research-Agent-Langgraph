"""Spend controls for the public demo.

`concurrency_limit` bounds how many requests run *at once*. It says nothing about
total spend, so on its own it is not a budget — a public URL with a working API key
behind it is an open bar no matter how narrow the tap.

Three controls, in order of how much they actually protect you:

1. **A daily dollar cap**, counted in a SQLite file that survives restarts. Once
   today's spend crosses it the demo declines politely instead of billing you.
2. **Bring-your-own-key.** A visitor supplying their own key is not counted against
   the cap at all, because they are paying. The key is used for that request and
   never stored.
3. **An input length cap**, so a pasted novel cannot turn into a large prompt.

None of these is the real backstop. That is a monthly budget limit set on a
project-scoped key at the provider, which is the only control that cannot be
defeated by a bug in this file. The README says so.
"""

import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from research_agent.config import settings

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_spend (
    day   TEXT PRIMARY KEY,
    usd   REAL NOT NULL DEFAULT 0,
    runs  INTEGER NOT NULL DEFAULT 0
)
"""


def _db_path() -> Path:
    return Path(os.environ.get("RA_DEMO_DB") or settings().demo_db)


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(SCHEMA)
    return connection


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def spent_today() -> tuple[float, int]:
    """(usd, runs) recorded for today."""
    with _lock, _connect() as connection:
        row = connection.execute(
            "SELECT usd, runs FROM daily_spend WHERE day = ?", (_today(),)
        ).fetchone()
    return (row[0], row[1]) if row else (0.0, 0)


def record(usd: float) -> None:
    with _lock, _connect() as connection:
        connection.execute(
            "INSERT INTO daily_spend (day, usd, runs) VALUES (?, ?, 1) "
            "ON CONFLICT(day) DO UPDATE SET usd = usd + excluded.usd, runs = runs + 1",
            (_today(), usd),
        )


def check(question: str, own_key: str | None) -> str | None:
    """Why this request must be refused, or None to proceed.

    A visitor with their own key bypasses the cap entirely — the cap exists to
    protect the owner's key, and theirs is not it.
    """
    config = settings()

    question = (question or "").strip()
    if not question:
        return "Ask a question first."
    if len(question) > config.max_question_chars:
        return (
            f"That question is {len(question):,} characters; the demo accepts up to "
            f"{config.max_question_chars:,}. Try a shorter one."
        )

    if own_key:
        return None

    if config.daily_cap_usd <= 0:
        return None

    usd, runs = spent_today()
    if usd >= config.daily_cap_usd:
        return (
            f"The shared demo budget for today (${config.daily_cap_usd:.2f}) is spent — "
            f"{runs} runs so far. It resets at midnight UTC.\n\n"
            "You can keep going immediately by pasting your own OpenAI API key in the "
            "box below: it is used for your request only and never stored. Or run the "
            "project locally — the README has a one-command quickstart."
        )
    return None


@contextmanager
def borrowed_key(own_key: str | None) -> Iterator[None]:
    """Temporarily swap in a visitor's key, and always put the original back.

    The key lives in the environment for the duration of one request and is not
    written anywhere. `_client()` is cached, so it is cleared on the way in and out
    to stop a borrowed key leaking into the next visitor's request.
    """
    if not own_key:
        yield
        return

    from research_agent.llm import _client

    previous = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = own_key.strip()
    _client.cache_clear()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = previous
        _client.cache_clear()


def status_line() -> str:
    config = settings()
    if config.daily_cap_usd <= 0:
        return "Demo budget: uncapped (local run)."
    usd, runs = spent_today()
    return (
        f"Shared demo budget today: ${usd:.3f} of ${config.daily_cap_usd:.2f} across {runs} runs."
    )
