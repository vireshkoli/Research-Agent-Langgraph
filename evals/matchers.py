"""Structural comparison of tool-call arguments.

The alternative — `json.dumps(actual) == json.dumps(expected)` — fails on key order,
on whitespace, and on `5` versus `"5"`, and when it fails it tells you nothing about
which field was wrong. Seven matchers with explicit per-field semantics is what
"AST-style structural matching" means in practice.

Normalisation is deliberately narrow: lowercase, collapse whitespace, strip wrapping
quotes, and coerce numeric strings. Anything cleverer starts hiding genuine failures,
which is the one thing an eval harness must never do.
"""

import re
from typing import Any

from evals.schema import ArgMatcher

_WHITESPACE = re.compile(r"\s+")


def normalize(value: Any) -> Any:
    """Canonical form for comparison. Lists normalise elementwise and sort."""
    if isinstance(value, str):
        text = _WHITESPACE.sub(" ", value).strip().strip("\"'").lower()
        # "5" and 5 should compare equal; a model may emit either.
        try:
            return float(text) if _looks_numeric(text) else text
        except ValueError:
            return text
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, list | tuple):
        return sorted((normalize(item) for item in value), key=repr)
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in sorted(value.items())}
    return value


def _looks_numeric(text: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", text))


def as_text(value: Any) -> str:
    """Flattened lowercase text, for the `contains_*` family."""
    normalized = normalize(value)
    if isinstance(normalized, list):
        return " ".join(str(item) for item in normalized)
    if isinstance(normalized, dict):
        return " ".join(f"{k} {v}" for k, v in normalized.items())
    return str(normalized)


def matches(actual: Any, spec: ArgMatcher) -> bool:
    """Whether one actual argument satisfies one matcher."""
    kind = spec.matcher

    if kind == "any":
        return True

    if kind == "exact":
        return normalize(actual) == normalize(spec.value)

    if kind in ("contains_any", "contains_all"):
        haystack = as_text(actual)
        needles = spec.value if isinstance(spec.value, list) else [spec.value]
        hits = (str(normalize(needle)) in haystack for needle in needles)
        return (
            any(hits)
            if kind == "contains_any"
            else all(str(normalize(needle)) in haystack for needle in needles)
        )

    if kind == "regex":
        return bool(re.search(str(spec.value), str(actual), re.IGNORECASE | re.DOTALL))

    if kind in ("numeric_close", "lte", "gte"):
        left, right = _as_number(actual), _as_number(spec.value)
        if left is None or right is None:
            return False
        if kind == "lte":
            return left <= right
        if kind == "gte":
            return left >= right
        # Relative tolerance, falling back to absolute when the target is zero.
        scale = abs(right) if right else 1.0
        return abs(left - right) <= spec.tolerance * scale

    return False


def _as_number(value: Any) -> float | None:
    normalized = normalize(value)
    return normalized if isinstance(normalized, float) else None


def call_matches(
    actual: dict[str, Any],
    expected_name: str,
    expected_args: dict[str, ArgMatcher],
    strict_args: bool = False,
) -> bool:
    """Whether an actual tool call satisfies an expectation.

    Extra arguments are tolerated by default — superset semantics, from the
    agentevals taxonomy — because a model passing an optional parameter we did not
    think to specify has not done anything wrong. `strict_args` opts a case out.
    """
    if actual.get("name") != expected_name:
        return False

    args = actual.get("args") or {}
    if strict_args and set(args) - set(expected_args):
        return False

    return all(key in args and matches(args[key], spec) for key, spec in expected_args.items())
