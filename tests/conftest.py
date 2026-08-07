"""Test isolation for the two pieces of global state in this project.

The spend ledger is a real file and the process guard is a module global. Without
this fixture every test that prices a call would append to the developer's actual
`.spend.json` and leak spend into the next test — so the suite would both corrupt
the number the README quotes and fail differently depending on run order.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from research_agent import llm
from research_agent.config import settings


@pytest.fixture(autouse=True)
def isolate_global_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("RA_SPEND_LEDGER", str(tmp_path / "spend.json"))
    monkeypatch.setenv("RA_LLM_CACHE", "off")
    monkeypatch.setenv("RA_LLM_CACHE_DIR", str(tmp_path / "llm-cache"))
    # Caps off by default; the tests that exercise them set their own.
    monkeypatch.setenv("RA_MAX_PROCESS_COST_USD", "0")
    monkeypatch.setenv("RA_MAX_PROJECT_COST_USD", "0")
    settings.cache_clear()
    llm._reset_process_spend()
    yield
    llm._reset_process_spend()
    settings.cache_clear()
