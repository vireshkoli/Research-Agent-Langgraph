"""The contract every tool implements.

Tools never raise at the caller. A failure is a `ToolResult(ok=False, error=...)`,
because a tool failure is an observation the agent should reason about — "that search
returned nothing, try different terms" — not an exception that ends the run. The
`observe` node counts consecutive failures and trips a circuit breaker; that only
works if failures arrive as data.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Source:
    """A citable document. Minted by tools, registered by `observe`, cited by `finalize`.

    Sources live in graph state rather than in message text, which is what makes
    compaction structurally unable to drop a citation.
    """

    url: str
    title: str
    snippet: str
    tool: str


@dataclass
class ToolResult:
    ok: bool
    content: str  # model-facing text, already truncated
    error: str | None = None
    sources: list[Source] = field(default_factory=list)
    raw_chars: int = 0  # size before truncation, so a debugger sees what was dropped
    truncated: bool = False
    meta: dict[str, Any] = field(default_factory=dict)  # e.g. search credits burned

    @classmethod
    def failure(cls, error: str, **meta: Any) -> "ToolResult":
        return cls(ok=False, content=f"Tool failed: {error}", error=error, meta=meta)


@dataclass(frozen=True)
class ToolSpec:
    """A tool plus everything the agent loop and the budget guard need to know about it."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the arguments object
    fn: Callable[..., ToolResult]
    max_calls: int  # per-run cap for this tool type, enforced by budget_verdict

    def openai_schema(self) -> dict[str, Any]:
        """The function definition as the Chat Completions API wants it.

        `strict` requires `additionalProperties: false` and every property listed in
        `required`; optional arguments are expressed as a nullable type instead.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "strict": True,
            },
        }


def truncate(text: str, limit: int) -> tuple[str, int, bool]:
    """Cap model-facing text. Returns (text, original_length, was_truncated).

    A Tavily raw-content response is routinely 100k+ characters. Without this cap a
    single search would blow both the context window and the run budget in one step.
    """
    raw_chars = len(text)
    if raw_chars <= limit:
        return text, raw_chars, False
    kept = text[:limit].rsplit(" ", 1)[0]
    return (
        f"{kept}\n\n[truncated: {raw_chars - len(kept):,} of {raw_chars:,} chars]",
        raw_chars,
        True,
    )
