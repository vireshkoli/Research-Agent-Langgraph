"""The tool registry: one place that knows every tool, its schema and its budget.

`observe` dispatches through here rather than importing tools directly, so adding a
tool is a one-line change and the per-tool call caps stay next to the tools they cap.

Dispatch never raises. Bad arguments from the model — a missing field, a misspelled
one, the wrong type — come back as `ToolResult(ok=False, ...)`, because "you called
that wrong" is something the agent can read and correct on the next turn, whereas an
exception ends the run.
"""

import inspect
from typing import Any

from research_agent.tools import calculator, code_exec, fetch_page, file_ops, web_search
from research_agent.tools.base import ToolResult, ToolSpec

REGISTRY: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        web_search.SPEC,
        fetch_page.SPEC,
        calculator.SPEC,
        code_exec.SPEC,
        file_ops.SPEC,
    )
}


def openai_schemas(names: list[str] | None = None) -> list[dict[str, Any]]:
    """Function definitions for the `tools=` parameter, in registry order."""
    selected = names or list(REGISTRY)
    return [REGISTRY[name].openai_schema() for name in selected if name in REGISTRY]


def dispatch(name: str, arguments: dict[str, Any]) -> ToolResult:
    """Run one tool call. Returns a ToolResult for every outcome, including bad input."""
    spec = REGISTRY.get(name)
    if spec is None:
        return ToolResult.failure(
            f"unknown tool {name!r}; available: {', '.join(sorted(REGISTRY))}"
        )
    if not isinstance(arguments, dict):
        return ToolResult.failure(
            f"{name} expects an object of arguments, got {type(arguments).__name__}"
        )

    signature = inspect.signature(spec.fn)
    accepted = set(signature.parameters)
    unexpected = set(arguments) - accepted
    if unexpected:
        return ToolResult.failure(
            f"{name} got unexpected argument(s) {', '.join(sorted(unexpected))}; "
            f"expected {', '.join(sorted(accepted))}"
        )
    missing = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.default is inspect.Parameter.empty and parameter.name not in arguments
    ]
    if missing:
        return ToolResult.failure(
            f"{name} is missing required argument(s) {', '.join(p.name for p in missing)}"
        )

    try:
        return spec.fn(**arguments)
    except Exception as exc:  # noqa: BLE001 — a broken tool must not end the run
        return ToolResult.failure(f"{name} raised {type(exc).__name__}: {exc}")
