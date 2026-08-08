"""Central configuration. All values overridable via environment (prefix RA_) or .env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RA_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Models. API keys are read by their SDKs from OPENAI_API_KEY / TAVILY_API_KEY
    # directly; they are not RA_-prefixed settings.
    agent_model: str = "gpt-5.4-nano"
    # luna at $1/$0.10/$6 rather than terra at $2.50/$0.25/$15. Phase 7 judges 30
    # items with both and publishes the agreement, so this is a validated choice
    # rather than an assumed one.
    judge_model: str = "gpt-5.6-luna"
    judge_reference_model: str = "gpt-5.6-terra"
    # none|low|medium|high|xhigh. Reasoning tokens bill as output at the full rate,
    # so the loop defaults to none and the planning steps opt in.
    reasoning_effort: str = "none"
    plan_reasoning_effort: str = "low"

    # Per-run budgets. A breach never crashes: it routes to finalize and returns
    # a partial answer. See budget.py for the layering.
    max_steps: int = 8  # act->observe rounds
    max_seconds: float = 120.0
    max_run_cost_usd: float = 0.05
    # Unbounded replanning is a loop generator, so one corrective replan only.
    max_replans: int = 1
    # Circuit breaker: three failures in a row means the tool is down or the model
    # is stuck calling it wrong. Either way, more attempts will not help.
    max_tool_failures: int = 3
    # Steps kept verbatim at the tail of the scratchpad when compacting; the model
    # needs recent detail to choose the next call.
    keep_last_steps: int = 3
    # Compaction fires on the last act call's real prompt_tokens, not an estimate.
    # Low enough to actually trigger on multi-hop runs: with a 400k context this is
    # a cost and attention control, not a context-overflow necessity.
    compact_threshold_tokens: int = 12_000
    max_compactions: int = 3
    # Held back from the loop so finalize can always afford to synthesise an answer.
    # Without this, exhausting the budget mid-loop would return nothing at all.
    finalize_reserve_usd: float = 0.008

    # Process-level kill switch, independent of any CostTracker. Guards a runaway
    # dev loop even when a caller forgets to pass a tracker. 0 disables it.
    max_process_cost_usd: float = 0.50
    # Hard ceiling on everything this project ever spends, persisted across restarts
    # in .spend.json. Neither of the guards above survives a process exit, so
    # neither can bound a few hundred short development runs. 0 disables it.
    max_project_cost_usd: float = 5.00

    # Observations are truncated before entering the prompt. A Tavily raw-content
    # response is routinely 100k+ chars; one uncapped search would blow both the
    # context window and the budget in a single step.
    max_observation_chars: int = 4000

    # Tools
    # Everything file_ops can reach. Resolved to an absolute path and used as a
    # confinement root, so nothing outside it is addressable.
    workspace_dir: Path = Path("workspace")
    search_max_results: int = 5
    # "basic" costs 1 Tavily credit, "advanced" 2. The free tier is 1000/month and
    # a full 30-case x 3-run evaluation spends ~360 of them.
    search_depth: str = "basic"
    fetch_timeout_seconds: float = 15.0
    # Dev-only. The official evaluation runs with --no-cache: a shared search cache
    # would make repeated runs non-independent and invalidate pass^k.
    search_cache_enabled: bool = False

    # Public demo. concurrency_limit bounds simultaneous calls, not total spend, so
    # a public URL needs an actual dollar ceiling. Visitors who supply their own key
    # bypass it, since the cap exists to protect this project's key.
    daily_cap_usd: float = 0.25
    max_question_chars: int = 500
    demo_db: Path = Path("data/demo.sqlite3")
    concurrency_limit: int = 2
    queue_max_size: int = 20


@lru_cache
def settings() -> Settings:
    return Settings()
