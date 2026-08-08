"""Trace-level evaluation over multi-step runs.

Hand-rolled rather than built on an agent-eval library, and the README says why:
`agentevals` has had no release in a year, Ragas changed GitHub orgs and is
deprecating its own ToolCallAccuracy, and OpenAI's hosted Evals platform shuts down
on 30 November 2026. A few hundred lines that are owned outright have a longer
half-life than any of them, and the metric definitions stay legible instead of
living behind someone else's `.evaluate()`.
"""
