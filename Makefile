.PHONY: dev test lint fmt typecheck check spike eval report requirements clean

# Install the project plus dev tooling into .venv.
dev:
	uv sync --dev

# Unit tests. LLM calls are faked; no API key required.
test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy

# What CI runs. Run this before every commit.
check: lint typecheck test

# Phase-1 de-risking spikes. These make real API calls (a few cents).
spike:
	uv run python scripts/spike_openai.py
	uv run python scripts/spike_tavily.py

# Validates every eval case and spends $0. Always run this before a real eval.
eval-dry:
	uv run python -m evals.run --dry-run

# The official evaluation. Costs real money — see README before running.
eval:
	uv run python -m evals.run --variant full --runs 3 --no-cache --batch

# Regenerates evals/REPORT.md from committed results. Must be a pure function
# of evals/results/ — CI asserts `git diff --exit-code` after running this.
report:
	uv run python -m evals.report

# Hugging Face Gradio Spaces install from requirements.txt, not pyproject.toml.
requirements:
	uv export --no-hashes --no-dev --format requirements-txt -o requirements.txt

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
