"""Run model-generated Python in a hardened subprocess.

READ THIS BEFORE TRUSTING IT: this is a resource-abuse guardrail, not a security
boundary. The child runs on the same kernel, as the same user, with the same
filesystem view as the app. Real isolation — network namespaces (`unshare -n`),
bubblewrap, nsjail, gVisor, Firecracker — all require CAP_SYS_ADMIN or a privileged
container, which is not available in the unprivileged containers this deploys to.

What it does provide, in descending order of how much it actually matters:

1. **A scrubbed environment.** The child is handed an explicit dict, never
   `os.environ`. OPENAI_API_KEY and TAVILY_API_KEY simply do not exist in the
   child, so even a complete escape finds no credentials to exfiltrate. This is
   the single highest-value line of code in the module.
2. **An import allowlist**, checked by walking the AST before anything runs. The
   plan called for a denylist; an allowlist is strictly safer and, for a research
   agent that needs arithmetic and data munging, costs nothing real. Dunder
   attribute access is rejected too, since `().__class__.__bases__[0].__subclasses__()`
   is the classic route from a "safe" expression to arbitrary imports.
3. **Resource limits set by the child on itself.** Soft and hard limits are set to
   the same value, which makes them irreversible — a process may lower a limit but
   never raise it above its hard limit.
4. **A wall-clock timeout that kills the whole process group.** A bare `proc.kill()`
   leaks grandchildren; `start_new_session=True` plus `killpg` does not.

The limits are applied by a bootstrap in the child rather than through
`preexec_fn`, because `preexec_fn` is documented as unsafe in the presence of
threads and `observe` runs tools in a thread pool.
"""

import ast
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from research_agent.tools.base import ToolResult, ToolSpec, truncate

CPU_SECONDS = 5
WALL_SECONDS = 10.0
MAX_OUTPUT_CHARS = 4000
MAX_FILE_BYTES = 1 * 1024 * 1024
MAX_ADDRESS_SPACE_BYTES = 512 * 1024 * 1024
MAX_OPEN_FILES = 64
MAX_SOURCE_CHARS = 20_000

# Everything the child is permitted to import. Anything reaching the filesystem,
# the network, the process table, or the import machinery is absent by design.
ALLOWED_MODULES = frozenset(
    {
        "math",
        "cmath",
        "statistics",
        "decimal",
        "fractions",
        "numbers",
        "random",
        "itertools",
        "functools",
        "operator",
        "collections",
        "heapq",
        "bisect",
        "array",
        "json",
        "re",
        "string",
        "textwrap",
        "unicodedata",
        "difflib",
        "datetime",
        "calendar",
        "time",
        "zoneinfo",
        "dataclasses",
        "enum",
        "typing",
        "copy",
        "pprint",
        "uuid",
        "hashlib",
        "base64",
    }
)

# Names that reopen the door the import allowlist just closed.
BLOCKED_NAMES = frozenset(
    {
        "__import__",
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "memoryview",
        "breakpoint",
        "exit",
        "quit",
        "help",
        "copyright",
        "credits",
    }
)

# The child sets its own limits, then runs the user's file. Keeping the user's
# source in a separate file means tracebacks point at real line numbers and a
# `from __future__` import at the top of it stays legal.
BOOTSTRAP = """\
import resource, sys

# (name, soft, hard). soft == hard makes a limit irreversible, since a process may
# lower a limit but never raise it above its hard limit.
#
# RLIMIT_CPU is the deliberate exception. Exceeding the *soft* limit raises
# SIGXCPU, which the parent can recognise and report as "you used too much CPU";
# exceeding the *hard* limit is an unconditional SIGKILL. With soft == hard, Linux
# escalates to SIGKILL before CPython reaches a bytecode boundary where the signal
# would surface, and the parent only ever sees an opaque -9. One second of grace
# makes the diagnosable signal arrive first and keeps SIGKILL as the backstop.
LIMITS = [
    ("RLIMIT_CPU", {cpu}, {cpu} + 1),
    ("RLIMIT_FSIZE", {fsize}, {fsize}),
    ("RLIMIT_NOFILE", {nofile}, {nofile}),
    ("RLIMIT_CORE", 0, 0),
]
if sys.platform.startswith("linux"):
    # RLIMIT_AS is skipped on macOS, where it is unreliable and can kill the
    # interpreter during start-up. RLIMIT_NPROC is per-*user*, not per-process,
    # so on a development machine it would count the developer's own shells.
    LIMITS += [
        ("RLIMIT_AS", {address_space}, {address_space}),
        ("RLIMIT_NPROC", 64, 64),
    ]

for name, soft, hard in LIMITS:
    limit = getattr(resource, name, None)
    if limit is not None:
        try:
            resource.setrlimit(limit, (soft, hard))
        except (ValueError, OSError):
            pass

source = open({user_file!r}).read()
sys.argv = [{user_file!r}]
exec(compile(source, {user_file!r}, "exec"), {{"__name__": "__main__"}})
"""


class UnsafeCode(Exception):
    """Raised by the AST pre-screen. Always surfaced as a ToolResult, never propagated."""


def _screen(source: str) -> None:
    """Reject code that the sandbox is not willing to even start.

    Runs before the subprocess, so the cheapest rejections cost nothing.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise UnsafeCode(f"syntax error on line {exc.lineno}: {exc.msg}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_MODULES:
                    raise UnsafeCode(
                        f"import of {alias.name!r} is not allowed. Permitted modules: "
                        f"{', '.join(sorted(ALLOWED_MODULES))}"
                    )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in ALLOWED_MODULES:
                raise UnsafeCode(f"import from {node.module!r} is not allowed")
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            raise UnsafeCode(f"use of {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            # Blocks ().__class__.__bases__[0].__subclasses__() and friends.
            raise UnsafeCode(f"access to dunder attribute {node.attr!r} is not allowed")


def _child_env(workdir: str) -> dict[str, str]:
    """The child's entire environment. Deliberately built up, never inherited.

    Nothing from os.environ reaches here, so no API key does either.
    """
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": workdir,
        "TMPDIR": workdir,
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }


def execute(code: str) -> ToolResult:
    """Run `code` and return whatever it printed."""
    code = code or ""
    if not code.strip():
        return ToolResult.failure("empty code")
    if len(code) > MAX_SOURCE_CHARS:
        return ToolResult.failure(f"code exceeds {MAX_SOURCE_CHARS} characters")

    try:
        _screen(code)
    except UnsafeCode as exc:
        return ToolResult.failure(str(exc), rejected_by="static_screen")

    with tempfile.TemporaryDirectory(prefix="ra-sandbox-") as workdir:
        user_file = Path(workdir) / "user_code.py"
        user_file.write_text(code)
        runner = Path(workdir) / "_runner.py"
        runner.write_text(
            BOOTSTRAP.format(
                cpu=CPU_SECONDS,
                fsize=MAX_FILE_BYTES,
                nofile=MAX_OPEN_FILES,
                address_space=MAX_ADDRESS_SPACE_BYTES,
                user_file=str(user_file),
            )
        )

        process = subprocess.Popen(
            # -I is isolated mode: ignores PYTHONPATH, user site-packages, and the
            # current directory. -S skips site initialisation.
            [sys.executable, "-I", "-S", str(runner)],
            cwd=workdir,
            env=_child_env(workdir),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            # Makes the child a process-group leader so killpg reaps grandchildren.
            start_new_session=True,
        )

        try:
            stdout, stderr = process.communicate(timeout=WALL_SECONDS)
        except subprocess.TimeoutExpired:
            _kill_group(process)
            stdout, stderr = process.communicate()
            return ToolResult.failure(
                f"execution exceeded the {WALL_SECONDS:g}s wall-clock limit", timed_out=True
            )

    if process.returncode != 0:
        detail = (stderr or "").strip() or f"exited with status {process.returncode}"
        # Without these the agent sees an opaque negative return code and tends to
        # retry the same expensive thing. SIGXCPU is the soft CPU limit. SIGKILL is
        # the hard backstop, and is also what RLIMIT_AS and the OOM killer produce,
        # so it is reported as the resource kill it is without over-claiming which.
        if process.returncode == -signal.SIGXCPU:
            detail = f"exceeded the {CPU_SECONDS}s CPU limit"
        elif process.returncode == -signal.SIGKILL:
            detail = f"killed after exceeding a resource limit ({CPU_SECONDS}s CPU, or memory)"
        content, raw_chars, was_truncated = truncate(detail, MAX_OUTPUT_CHARS)
        return ToolResult(
            ok=False,
            content=f"Execution failed: {content}",
            error=content,
            raw_chars=raw_chars,
            truncated=was_truncated,
            meta={"returncode": process.returncode},
        )

    output = (stdout or "").strip()
    if not output:
        return ToolResult(
            ok=True,
            content="Ran successfully but printed nothing. Use print() to return a value.",
            meta={"returncode": 0},
        )
    content, raw_chars, was_truncated = truncate(output, MAX_OUTPUT_CHARS)
    return ToolResult(
        ok=True,
        content=content,
        raw_chars=raw_chars,
        truncated=was_truncated,
        meta={"returncode": 0},
    )


def _kill_group(process: subprocess.Popen[str]) -> None:
    """SIGKILL the child's whole process group; a bare kill() would leak grandchildren."""
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()


SPEC = ToolSpec(
    name="code_execution",
    description=(
        "Execute a short Python program and return everything it prints. Use print() "
        "to return results. Standard library only, from a restricted set (math, "
        "statistics, json, re, datetime, itertools, collections, decimal and similar) "
        "— no file, network or OS access. There is a 5s CPU limit. For plain "
        "arithmetic prefer the calculator tool, which is faster and always available."
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python source to run. Must print() its result.",
            }
        },
        "required": ["code"],
        "additionalProperties": False,
    },
    fn=execute,
    max_calls=5,
)
