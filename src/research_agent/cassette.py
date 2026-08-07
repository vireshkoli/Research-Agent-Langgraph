"""Record LLM responses once, replay them for free.

Development is the most expensive part of this project, not the evaluation: the
same handful of prompts get re-run hundreds of times while the graph, the prompts
and the UI are shaken out. Each of those repeats costs the same as the first.

A cassette keys a call on everything that determines its response — model, input,
tools, reasoning effort, output schema — and stores the *derived* result rather
than the raw SDK object, so replay does not depend on the SDK's internal shapes
staying still.

Two rules keep this honest:

  - Default is `off`. Caching is opt-in, so nothing silently replays.
  - The official evaluation runs with `off`, exactly as it runs `--no-cache` for
    search. Replayed runs are not independent, and `pass^k` over non-independent
    runs is a meaningless number.

Replayed calls are recorded with `cost_usd=0.0` and `replayed=True`, because that
is what they actually cost.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

MODE_ENV = "RA_LLM_CACHE"
DIR_ENV = "RA_LLM_CACHE_DIR"
DEFAULT_DIR = Path(".cache/llm")
VALID_MODES = frozenset({"off", "auto", "record", "replay"})


def mode() -> str:
    """off (default) | auto (replay if present, else call and record) | record | replay."""
    value = (os.environ.get(MODE_ENV) or "off").strip().lower()
    return value if value in VALID_MODES else "off"


def cache_dir() -> Path:
    return Path(os.environ.get(DIR_ENV) or DEFAULT_DIR)


def key(payload: dict[str, Any]) -> str:
    """Stable hash of everything that determines the response.

    `default=str` keeps unserialisable values (a pydantic class, say) from raising;
    they contribute their repr, which is stable enough for a dev cache.
    """
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def load(cache_key: str) -> dict[str, Any] | None:
    if mode() not in ("auto", "replay"):
        return None
    path = cache_dir() / f"{cache_key}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def save(cache_key: str, value: dict[str, Any]) -> None:
    if mode() not in ("auto", "record"):
        return
    path = cache_dir() / f"{cache_key}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, default=str))
    except OSError:
        pass  # an unwritable cache is a missed saving, not an error


def misses_are_fatal() -> bool:
    """In strict `replay` mode a miss means the recording is stale."""
    return mode() == "replay"
