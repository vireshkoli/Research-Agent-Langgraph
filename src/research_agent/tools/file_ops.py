"""Read, write and list files, confined to a single workspace directory.

Confinement is enforced by resolving the requested path to an absolute real path and
checking that it is under the resolved workspace root. Resolving first is what makes
this robust: it collapses `..` segments *and* follows symlinks, so neither
`../../etc/passwd` nor a symlink planted inside the workspace escapes. Checking the
string for ".." before resolving — the common version of this code — catches the
first and misses the second.
"""

from pathlib import Path

from research_agent.config import settings
from research_agent.tools.base import ToolResult, ToolSpec, truncate

MAX_READ_CHARS = 8000
MAX_WRITE_CHARS = 100_000
MAX_LISTED_ENTRIES = 200


class PathEscape(Exception):
    """Raised when a requested path resolves outside the workspace root."""


def workspace_root() -> Path:
    root = Path(settings().workspace_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve(path: str) -> Path:
    """Resolve `path` under the workspace root, or raise PathEscape."""
    root = workspace_root()
    if not path or not path.strip():
        raise PathEscape("empty path")
    # strict=False so a not-yet-created file still resolves for writes.
    target = (root / path).resolve()
    if target != root and not target.is_relative_to(root):
        raise PathEscape(f"{path!r} resolves outside the workspace")
    return target


def file_ops(operation: str, path: str, content: str | None = None) -> ToolResult:
    try:
        target = _resolve(path)
    except PathEscape as exc:
        return ToolResult.failure(str(exc), rejected_by="path_confinement")
    except (OSError, RuntimeError) as exc:  # RuntimeError: symlink loop
        return ToolResult.failure(f"could not resolve {path!r}: {exc}")

    root = workspace_root()
    relative = target.relative_to(root) if target != root else Path(".")

    if operation == "read":
        if not target.is_file():
            return ToolResult.failure(f"{relative} does not exist or is not a file")
        try:
            text = target.read_text(errors="replace")
        except OSError as exc:
            return ToolResult.failure(f"could not read {relative}: {exc}")
        body, raw_chars, was_truncated = truncate(text, MAX_READ_CHARS)
        return ToolResult(
            ok=True,
            content=f"{relative}:\n{body}",
            raw_chars=raw_chars,
            truncated=was_truncated,
        )

    if operation == "write":
        if content is None:
            return ToolResult.failure("write requires content")
        if len(content) > MAX_WRITE_CHARS:
            return ToolResult.failure(f"content exceeds {MAX_WRITE_CHARS} characters")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        except OSError as exc:
            return ToolResult.failure(f"could not write {relative}: {exc}")
        return ToolResult(ok=True, content=f"Wrote {len(content):,} characters to {relative}.")

    if operation == "list":
        base = target if target.is_dir() else root
        try:
            entries = sorted(p for p in base.rglob("*") if p.is_file())
        except OSError as exc:
            return ToolResult.failure(f"could not list {relative}: {exc}")
        if not entries:
            return ToolResult(ok=True, content="The workspace is empty.")
        shown = entries[:MAX_LISTED_ENTRIES]
        lines = [f"{p.relative_to(root)} ({p.stat().st_size:,} bytes)" for p in shown]
        if len(entries) > len(shown):
            lines.append(f"... and {len(entries) - len(shown)} more")
        return ToolResult(ok=True, content="\n".join(lines))

    return ToolResult.failure(f"unknown operation {operation!r}; use read, write or list")


SPEC = ToolSpec(
    name="file_ops",
    description=(
        "Read, write or list files in a scratch workspace. Use it to save intermediate "
        "notes or data across steps. Paths are relative to the workspace; nothing "
        "outside it is reachable. Operations: 'read' (needs path), 'write' (needs path "
        "and content), 'list' (path may be '.')."
    ),
    parameters={
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["read", "write", "list"],
                "description": "What to do.",
            },
            "path": {
                "type": "string",
                "description": "Workspace-relative path, e.g. 'notes.md' or '.' to list.",
            },
            "content": {
                "type": ["string", "null"],
                "description": "Text to write. Required for 'write', null otherwise.",
            },
        },
        "required": ["operation", "path", "content"],
        "additionalProperties": False,
    },
    fn=file_ops,
    max_calls=10,
)
