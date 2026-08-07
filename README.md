# Tool-Using Research Agent

A ReAct research agent built on an explicit LangGraph state graph: it decomposes a question, calls
tools in a loop, and returns a cited answer — with hard budgets on steps, wall clock, and dollars, so
it can never run away.

> **Status: under construction.** The results table, architecture diagram, live demo link, and
> evaluation methodology land as the build progresses. Every number published here will be
> reproducible from the committed traces in `evals/results/` — nothing is estimated.

## What is here so far

- `src/research_agent/llm.py` — the single OpenAI wrapper. Per-call token and cost accounting at
  list prices, a per-run budget guard with a reserve so an exhausted run still returns an answer,
  and a process-level kill switch that fires even when a caller forgets to pass a tracker.
- `src/research_agent/config.py` — all configuration, overridable via `RA_`-prefixed env vars.

## Quickstart

```bash
uv sync --dev
cp .env.example .env      # add OPENAI_API_KEY and TAVILY_API_KEY
make check                # ruff, mypy, pytest — no API key needed
```

## Licence

MIT — see [LICENSE](LICENSE).
