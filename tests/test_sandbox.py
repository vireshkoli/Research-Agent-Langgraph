"""The code_execution sandbox. These tests are the claim — without them the README
cannot honestly say anything about what the sandbox does.

Layered on purpose: the static screen is the first layer, and the scrubbed
environment is the second. `test_env_scrub_holds_when_the_static_screen_is_bypassed`
is the important one, because it proves the second layer works on its own.
"""

import os
import sys

import pytest

from research_agent.tools import code_exec
from research_agent.tools.code_exec import UnsafeCode, _child_env, _screen, execute

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="POSIX resource limits and process groups"
)


# --- the environment scrub: the layer that actually protects the keys -----------


def test_child_env_contains_no_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-leak")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-must-not-leak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-leak")

    env = _child_env("/tmp/whatever")

    assert "OPENAI_API_KEY" not in env
    assert "TAVILY_API_KEY" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert not any("must-not-leak" in value for value in env.values())


def test_child_env_is_an_explicit_allowlist_not_a_filtered_copy() -> None:
    # A filtered copy of os.environ grows a hole every time someone adds a new
    # secret-bearing variable. Building the dict up means new variables are absent
    # by default rather than present by default.
    assert set(_child_env("/tmp/x")) == {
        "PATH",
        "HOME",
        "TMPDIR",
        "LC_ALL",
        "LANG",
        "PYTHONIOENCODING",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHASHSEED",
    }


def test_env_scrub_holds_when_the_static_screen_is_bypassed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Defence in depth, stated as a test: disable the import allowlist entirely and
    # confirm the child still cannot read a key out of the real environment.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-leak")
    monkeypatch.setattr(code_exec, "_screen", lambda source: None)

    result = execute(
        "import os\nprint('OPENAI_API_KEY' in os.environ, os.environ.get('OPENAI_API_KEY'))\n"
    )

    assert result.ok, result.content
    assert "False None" in result.content
    assert "sk-must-not-leak" not in result.content


def test_the_child_cannot_see_the_parents_working_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(code_exec, "_screen", lambda source: None)
    result = execute("import os\nprint(os.getcwd())")
    assert result.ok, result.content
    assert "ra-sandbox-" in result.content
    assert os.getcwd() not in result.content


# --- the static screen ---------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "import os",
        "import socket",
        "import subprocess",
        "import shutil",
        "import ctypes",
        "import importlib",
        "import urllib.request",
        "from os import environ",
        "from subprocess import run",
        "import os.path",
    ],
    ids=lambda s: s.replace(" ", "_"),
)
def test_screen_rejects_dangerous_imports(source: str) -> None:
    with pytest.raises(UnsafeCode):
        _screen(source)


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os')",
        "eval('1+1')",
        "exec('x=1')",
        "compile('1', '<s>', 'eval')",
        "open('/etc/passwd')",
        "getattr(int, 'x')",
        "globals()",
        # The classic escape chain: reach type objects through a dunder and walk
        # to a subclass that can open files.
        "().__class__.__bases__[0].__subclasses__()",
        "(1).__class__",
        "[].__getattribute__('append')",
    ],
    ids=lambda s: s[:28],
)
def test_screen_rejects_escape_primitives(source: str) -> None:
    with pytest.raises(UnsafeCode):
        _screen(source)


@pytest.mark.parametrize(
    "source",
    [
        "import math\nprint(math.sqrt(2))",
        "import json, re\nprint(json.dumps({'a': 1}))",
        "from decimal import Decimal\nprint(Decimal('0.1') + Decimal('0.2'))",
        "from collections import Counter\nprint(Counter('aab'))",
        "import statistics\nprint(statistics.mean([1, 2, 3]))",
    ],
    ids=lambda s: s.split("\n")[0],
)
def test_screen_allows_the_analysis_toolkit(source: str) -> None:
    _screen(source)  # must not raise


def test_screen_reports_syntax_errors_with_a_line_number() -> None:
    with pytest.raises(UnsafeCode, match="line 1"):
        _screen("def broken(:")


# --- resource limits -----------------------------------------------------------


def test_runaway_cpu_is_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(code_exec, "CPU_SECONDS", 1)
    monkeypatch.setattr(code_exec, "WALL_SECONDS", 30.0)

    result = execute("x = 0\nwhile True:\n    x += 1")

    assert not result.ok
    assert "CPU limit" in result.content


def test_wall_clock_timeout_kills_a_sleeping_child(monkeypatch: pytest.MonkeyPatch) -> None:
    # CPU limits do not catch a process that is idle rather than busy, which is why
    # the wall clock is a separate limit rather than a larger CPU budget.
    monkeypatch.setattr(code_exec, "WALL_SECONDS", 1.0)

    result = execute("import time\ntime.sleep(30)")

    assert not result.ok
    assert result.meta.get("timed_out") is True
    assert "wall-clock" in result.error or ""


def test_memory_hog_is_contained_rather_than_taking_down_the_host() -> None:
    # On Linux RLIMIT_AS turns this into a MemoryError in the child. On macOS the
    # limit is not applied, so the assertion is only that we survive and report.
    result = execute("x = 'a' * (2 * 1024 * 1024 * 1024)\nprint(len(x))")
    assert isinstance(result.ok, bool)  # the point is that we returned at all


# --- ordinary behaviour --------------------------------------------------------


def test_prints_are_returned() -> None:
    result = execute("print(405 - 123)")
    assert result.ok
    assert "282" in result.content


def test_silent_success_explains_itself() -> None:
    # A model that computes the answer and forgets to print it gets told so,
    # instead of an empty observation it cannot act on.
    result = execute("x = 1 + 1")
    assert result.ok
    assert "print()" in result.content


def test_runtime_errors_come_back_as_data_not_exceptions() -> None:
    result = execute("print(1 / 0)")
    assert not result.ok
    assert "ZeroDivisionError" in result.content


def test_large_output_is_truncated_and_says_so() -> None:
    result = execute("print('x' * 50000)")
    assert result.ok
    assert result.truncated
    assert result.raw_chars > 40_000
    assert len(result.content) < 6_000
    assert "truncated" in result.content


def test_blocked_import_is_a_failed_result_not_a_raise() -> None:
    result = execute("import os\nprint(os.listdir('/'))")
    assert not result.ok
    assert result.meta.get("rejected_by") == "static_screen"
    assert "not allowed" in result.content


def test_empty_and_oversized_source_are_rejected_cheaply() -> None:
    assert not execute("").ok
    assert not execute("   \n  ").ok
    assert not execute("x = 1\n" * 20_000).ok
