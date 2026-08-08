"""The run trace: one canonical document per run, written at the end.

Deliberately **not** the same thing as the live UI event stream. The UI needs
low-latency partials with no consistency guarantees; the eval harness needs one
complete, validated document. Conflating them forces the eval to replay an event log
to reconstruct state, which is how eval harnesses start quietly lying about what
happened.

Pydantic at the disk boundary — `model_validate` gives free schema-drift detection
when an older result file is loaded — while `LLMCall` and `CostTracker` stay plain
dataclasses in memory.

`git_sha` and `prompt_hash` are what turn a committed result into an *auditable*
one: a number in the README can be traced to the code and prompts that produced it.
"""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from research_agent.config import Settings
from research_agent.prompts import prompt_hash
from research_agent.state import ResearchState

SCHEMA_VERSION = "1"


class UsageTotals(BaseModel):
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    cache_hit_rate: float = 0.0


class Outcome(BaseModel):
    status: str  # completed | early_exit | error
    early_exit_reason: str | None = None
    used_deterministic_finalize: bool = False
    reflect_overrules: int = 0
    error: str | None = None


class RunTrace(BaseModel):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    ts: str
    question: str
    variant: str
    config: dict[str, Any]
    plan: dict[str, Any]
    steps: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    answer: str
    citations: list[str]
    coverage: dict[str, bool]
    outcome: Outcome
    usage: dict[str, Any]
    timings_ms: dict[str, float] = Field(default_factory=dict)
    compaction: dict[str, Any] = Field(default_factory=dict)


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def build_trace(
    state: ResearchState, cfg: Settings, elapsed_ms: float, error: str | None = None
) -> RunTrace:
    calls = state.get("llm_calls", [])
    totals = UsageTotals(
        input_tokens=sum(c.get("input_tokens", 0) for c in calls),
        cached_tokens=sum(c.get("cached_tokens", 0) for c in calls),
        output_tokens=sum(c.get("output_tokens", 0) for c in calls),
        reasoning_tokens=sum(c.get("reasoning_tokens", 0) for c in calls),
        cost_usd=sum(c.get("cost_usd", 0.0) for c in calls),
    )
    if totals.input_tokens:
        totals.cache_hit_rate = totals.cached_tokens / totals.input_tokens

    by_purpose: dict[str, dict[str, Any]] = {}
    for call in calls:
        bucket = by_purpose.setdefault(
            call.get("purpose", "?"), {"calls": 0, "cost_usd": 0.0, "output_tokens": 0}
        )
        bucket["calls"] += 1
        bucket["cost_usd"] += call.get("cost_usd", 0.0)
        bucket["output_tokens"] += call.get("output_tokens", 0)

    reason = state.get("early_exit_reason")
    status = "error" if error else ("early_exit" if reason else "completed")

    return RunTrace(
        run_id=state.get("run_id", "unknown"),
        ts=datetime.now(UTC).isoformat(timespec="seconds"),
        question=state.get("question", ""),
        variant=state.get("variant", "full"),
        config={
            "agent_model": cfg.agent_model,
            "reasoning_effort": cfg.reasoning_effort,
            "max_steps": cfg.max_steps,
            "max_seconds": cfg.max_seconds,
            "max_run_cost_usd": cfg.max_run_cost_usd,
            "finalize_reserve_usd": cfg.finalize_reserve_usd,
            "compact_threshold_tokens": cfg.compact_threshold_tokens,
            "git_sha": git_sha(),
            "prompt_hash": prompt_hash(),
        },
        plan={
            "version": state.get("plan_version", 0),
            "subquestions": state.get("plan", []),
            "replans": state.get("replans", 0),
        },
        steps=[dict(step) for step in state.get("scratchpad", [])],
        sources=[dict(source) for source in state.get("sources", [])],
        answer=state.get("answer", ""),
        citations=state.get("citations", []),
        coverage=state.get("covered", {}),
        outcome=Outcome(
            status=status,
            early_exit_reason=reason,
            used_deterministic_finalize=state.get("used_deterministic_finalize", False),
            reflect_overrules=state.get("reflect_overrules", 0),
            error=error,
        ),
        usage={
            "llm_calls": calls,
            "totals": totals.model_dump(),
            "by_purpose": by_purpose,
            "search_credits": state.get("search_credits", 0),
            "steps": state.get("step", 0),
        },
        timings_ms={"total": elapsed_ms},
        compaction={
            "count": state.get("compactions", 0),
            "compacted_upto": state.get("compacted_upto", 0),
        },
    )


def write_trace(trace: RunTrace, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(trace.model_dump_json(indent=2))
    return path


def load_trace(path: Path) -> RunTrace:
    """Validates on load, so a schema change surfaces as an error rather than a
    silently missing field in a metric."""
    return RunTrace.model_validate(json.loads(path.read_text()))
