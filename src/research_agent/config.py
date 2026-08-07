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
    judge_model: str = "gpt-5.6-terra"
    reasoning_effort: str = "low"  # none|low|medium|high|xhigh — "low" is the cost sweet spot
    temperature: float = 0.0

    # Per-run budgets. A breach never crashes: it routes to finalize and returns
    # a partial answer. See budget.py for the layering.
    max_steps: int = 8  # act->observe rounds
    max_seconds: float = 120.0
    max_run_cost_usd: float = 0.05
    # Held back from the loop so finalize can always afford to synthesise an answer.
    # Without this, exhausting the budget mid-loop would return nothing at all.
    finalize_reserve_usd: float = 0.008

    # Process-level kill switch, independent of any CostTracker. Guards a runaway
    # dev loop even when a caller forgets to pass a tracker. 0 disables it.
    max_process_cost_usd: float = 2.00

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


@lru_cache
def settings() -> Settings:
    return Settings()
