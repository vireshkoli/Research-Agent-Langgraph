"""Run model-written Python, contained by the kernel rather than by Python.

Who writes this code? The model does — and the model reads web pages that anyone
can write. So the code that reaches this tool is ultimately chosen by strangers,
and it is reachable from the public demo. The containment is layered, and only the
first two layers are claimed to be sound:

1. **seccomp-bpf** (`containment.py`): no sockets, no exec, no signalling or tracing
   another process, no creating, writing, renaming or deleting files. Enforced by
   the kernel on every syscall. **Fails closed on Linux**: if the filter cannot be
   installed the code does not run.
2. **Landlock** (`containment.py`): reads confined to the standard library and the
   scratch directory, which keeps every `/proc/<pid>/environ` out of reach. Applied
   where the kernel supports it; each result records whether it was.
3. **A non-dumpable parent**: the app's own `/proc` entries are unreadable to its
   children even if Landlock is unavailable.
4. **A scrubbed environment and redacted output**: the child is never handed an API
   key, and any secret value that reaches stdout anyway is replaced before the
   model sees it. A backstop only — an encoding defeats it.
5. **Restricted builtins, a guarded importer and module proxies.** These close the
   four escapes an audit found in the earlier version (`uuid.os.popen`,
   `__builtins__['__import__']`, string-built dunders, reading `.env`) and two more
   found while fixing them (`__loader__.load_module`, and `typing.get_type_hints`,
   which evaluates strings). They are **not** a security boundary and are not
   claimed as one: CPython has too many ways back to the full interpreter, and
   in-process Python sandboxes have no record of staying sound. Their job is to make
   the obvious attempts fail fast and legibly.
6. **Resource limits and a process-group kill**, set by the child on itself.

On macOS layers 1-3 do not exist; the tool still runs for local development, and
every result's `meta["isolation"]` says exactly which layers applied.
"""

import ast
import base64
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from research_agent.tools import containment
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
        "copy",
        "pprint",
        "hashlib",
        "base64",
    }
)

# Names that reopen the door the import allowlist just closed.
BLOCKED_NAMES = frozenset(
    {
        "__import__",
        "__builtins__",
        "__loader__",
        "__spec__",
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

# Builtins the model's code may use, by name. An allowlist rather than "everything
# minus a blocklist": `__loader__` and `__spec__` are builtins too, and either one
# reaches the import machinery (`__loader__.load_module("os")`), which a blocklist
# that forgot them would have handed over.
SAFE_BUILTINS = frozenset(
    {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "classmethod",
        "complex",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "hash",
        "hex",
        "id",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "object",
        "oct",
        "ord",
        "pow",
        "print",
        "property",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "staticmethod",
        "str",
        "sum",
        "super",
        "tuple",
        "type",
        "zip",
        "__build_class__",
        "Ellipsis",
        "NotImplemented",
    }
)

# Attributes of allowlisted modules that look up an attribute or evaluate a string
# chosen at run time, which walks straight around a static screen.
GADGETS = {
    "operator": ["attrgetter", "methodcaller"],
    "string": ["Formatter"],
    "functools": ["singledispatch", "singledispatchmethod"],
}

CONTAINMENT_FAILED_EXIT = 97
ISOLATION_MARKER = "\x00ISOLATION "

# Static text, deliberately: parameters arrive in _config.json beside it rather than
# through str.format, because a security-critical script full of doubled braces is
# where a bug hides. The order is the point — load what may be imported, then let
# the kernel confine the process, then hand the model's code a restricted world.
BOOTSTRAP = r"""
import builtins, importlib, json, os, resource, sys, sysconfig, types

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "_config.json")) as handle:
    config = json.load(handle)
with open(os.path.join(here, "user_code.py")) as handle:
    source = handle.read()

# 1. Resource limits. soft == hard makes a limit irreversible: a process may lower
#    a limit but never raise it above its hard limit. RLIMIT_CPU is the exception —
#    one second of grace lets the diagnosable SIGXCPU arrive before SIGKILL.
limits = [
    ("RLIMIT_CPU", config["cpu"], config["cpu"] + 1),
    ("RLIMIT_FSIZE", config["fsize"], config["fsize"]),
    ("RLIMIT_NOFILE", config["nofile"], config["nofile"]),
    ("RLIMIT_CORE", 0, 0),
]
if sys.platform.startswith("linux"):
    # Skipped on macOS, where RLIMIT_AS can kill the interpreter at start-up.
    limits += [
        ("RLIMIT_AS", config["address_space"], config["address_space"]),
        ("RLIMIT_NPROC", 64, 64),
    ]
for name, soft, hard in limits:
    limit = getattr(resource, name, None)
    if limit is not None:
        try:
            resource.setrlimit(limit, (soft, hard))
        except (ValueError, OSError):
            pass

# 2. Import everything the model may use now, while the filesystem is open, so no
#    later import depends on reading anything.
allowed = set(config["allowed_modules"])
for name in sorted(allowed):
    try:
        importlib.import_module(name)
    except Exception:
        pass

# 3. Kernel containment. Fails closed on Linux.
sys.path.insert(0, here)
import _containment
sys.path.pop(0)
paths = sysconfig.get_paths()
read_paths = sorted({paths["stdlib"], paths["platstdlib"], here, *config["extra_read_paths"]})
try:
    isolation = _containment.contain(read_paths)
except Exception as exc:
    sys.stderr.write("containment could not be established: %s\n" % exc)
    sys.stderr.flush()
    os._exit(config["failed_exit"])
sys.stdout.write(config["marker"] + json.dumps(isolation) + "\n")
sys.stdout.flush()

# 4. A restricted world for the model's code. Defence in depth, not a boundary.
gadgets = {name: set(hidden) for name, hidden in config["gadgets"].items()}
proxies = {}

def proxy(module):
    # Public, non-module attributes only, so `statistics.sys` or `pprint.re` cannot
    # lead somewhere the allowlist does not. Allowed submodules are proxied in turn.
    key = module.__name__
    if key in proxies:
        return proxies[key]
    namespace = types.SimpleNamespace()
    proxies[key] = namespace
    hidden = gadgets.get(key, set())
    for attribute in dir(module):
        if attribute.startswith("_") or attribute in hidden:
            continue
        try:
            value = getattr(module, attribute)
        except Exception:
            continue
        if isinstance(value, types.ModuleType):
            if value.__name__.split(".")[0] in allowed:
                setattr(namespace, attribute, proxy(value))
            continue
        setattr(namespace, attribute, value)
    return namespace

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in allowed:
        raise ImportError("import of %r is not allowed" % name)
    module = importlib.import_module(name)
    return proxy(module if fromlist else importlib.import_module(name.split(".")[0]))

safe = {
    name: getattr(builtins, name)
    for name in config["safe_builtins"]
    if hasattr(builtins, name)
}
safe.update(
    (name, value)
    for name, value in vars(builtins).items()
    if isinstance(value, type) and issubclass(value, BaseException)
)
safe["__import__"] = guarded_import
user_globals = {"__name__": "__main__", "__builtins__": safe}

code = compile(source, os.path.join(here, "user_code.py"), "exec")
del here, config, source, handle, limits, paths, read_paths, isolation
exec(code, user_globals)
"""

# Read access the child keeps after Landlock, beyond the stdlib and its scratch
# directory: shared libraries an extension module may load, and timezone data.
EXTRA_READ_PATHS = (
    "/usr/lib",
    "/usr/lib64",
    "/lib",
    "/lib64",
    "/etc/ld.so.cache",
    "/usr/share/zoneinfo",
    "/etc/localtime",
)


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

    # Before any child exists: from here on this process's /proc entries, its
    # environment included, are unreadable to the children it spawns.
    containment.protect_parent()

    with tempfile.TemporaryDirectory(prefix="ra-sandbox-") as workdir:
        scratch = Path(workdir)
        (scratch / "user_code.py").write_text(code)
        (scratch / "_containment.py").write_text(Path(containment.__file__).read_text())
        (scratch / "_config.json").write_text(
            json.dumps(
                {
                    "cpu": CPU_SECONDS,
                    "fsize": MAX_FILE_BYTES,
                    "nofile": MAX_OPEN_FILES,
                    "address_space": MAX_ADDRESS_SPACE_BYTES,
                    "allowed_modules": sorted(ALLOWED_MODULES),
                    "safe_builtins": sorted(SAFE_BUILTINS),
                    "gadgets": GADGETS,
                    "extra_read_paths": list(EXTRA_READ_PATHS),
                    "failed_exit": CONTAINMENT_FAILED_EXIT,
                    "marker": ISOLATION_MARKER,
                }
            )
        )
        runner = scratch / "_runner.py"
        runner.write_text(BOOTSTRAP)

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

    isolation, stdout = _split_isolation(stdout or "")
    stdout, stderr = _redact(stdout), _redact(stderr or "")

    if process.returncode == CONTAINMENT_FAILED_EXIT:
        # Fail closed: on a host where the kernel cannot contain it, model-written
        # code does not run at all.
        return ToolResult.failure(
            "code execution is unavailable on this host: kernel isolation could not be "
            "established. Use the calculator for arithmetic.",
            isolation={"seccomp": False, "detail": stderr.strip()[:300]},
        )

    if process.returncode != 0:
        detail = stderr.strip() or f"exited with status {process.returncode}"
        # Without these the agent sees an opaque negative return code and tends to
        # retry the same expensive thing. SIGXCPU is the soft CPU limit. SIGKILL is
        # the hard backstop, and is also what RLIMIT_AS and the OOM killer produce,
        # so it is reported as the resource kill it is without over-claiming which.
        # SIGSYS is seccomp killing a syscall it treats as an attack, such as the
        # x32 ABI.
        if process.returncode == -signal.SIGXCPU:
            detail = f"exceeded the {CPU_SECONDS}s CPU limit"
        elif process.returncode == -signal.SIGKILL:
            detail = f"killed after exceeding a resource limit ({CPU_SECONDS}s CPU, or memory)"
        elif process.returncode == -signal.SIGSYS:
            detail = "killed by the sandbox for a forbidden system call"
        content, raw_chars, was_truncated = truncate(detail, MAX_OUTPUT_CHARS)
        return ToolResult(
            ok=False,
            content=f"Execution failed: {content}",
            error=content,
            raw_chars=raw_chars,
            truncated=was_truncated,
            meta={"returncode": process.returncode, "isolation": isolation},
        )

    output = stdout.strip()
    if not output:
        return ToolResult(
            ok=True,
            content="Ran successfully but printed nothing. Use print() to return a value.",
            meta={"returncode": 0, "isolation": isolation},
        )
    content, raw_chars, was_truncated = truncate(output, MAX_OUTPUT_CHARS)
    return ToolResult(
        ok=True,
        content=content,
        raw_chars=raw_chars,
        truncated=was_truncated,
        meta={"returncode": 0, "isolation": isolation},
    )


def _split_isolation(stdout: str) -> tuple[dict[str, object], str]:
    """Separate the bootstrap's report of which layers applied from the program's output."""
    if not stdout.startswith(ISOLATION_MARKER):
        return {}, stdout
    line, _, rest = stdout.partition("\n")
    try:
        report = json.loads(line[len(ISOLATION_MARKER) :])
    except ValueError:
        return {}, stdout
    return (report if isinstance(report, dict) else {}), rest


def _secrets() -> list[str]:
    """Every secret-shaped value in this process's environment."""
    found = []
    for name, value in os.environ.items():
        if len(value) >= 12 and any(
            marker in name.upper() for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD")
        ):
            found.append(value)
    return found


def _redact(text: str) -> str:
    """Replace any secret value — plain, base64 or reversed — before the model sees it.

    A backstop, not a control: the layers above are what keep a secret out of the
    child's reach. This catches the case where one gets there anyway and is
    printed, and an encoding it does not anticipate defeats it.
    """
    for secret in _secrets():
        for form in (secret, base64.b64encode(secret.encode()).decode(), secret[::-1]):
            if form in text:
                text = text.replace(form, "[redacted]")
    return text


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
