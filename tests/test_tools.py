"""Calculator, file_ops, fetch_page's extraction, web_search routing, and dispatch.

Nothing here touches the network: web_search is exercised by monkeypatching its two
backend functions, which is also how the fallback and circuit-breaker paths get
tested without waiting on a real rate limit.
"""

from pathlib import Path
from typing import Any

import httpx
import pytest

from research_agent.tools import registry, web_search
from research_agent.tools.calculator import calculate
from research_agent.tools.fetch_page import fetch_page, html_to_text
from research_agent.tools.file_ops import PathEscape, _resolve, file_ops
from research_agent.tools.registry import REGISTRY, dispatch, openai_schemas

# --- calculator ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("405 - 123", "282"),
        ("2 ** 10", "1024"),
        ("17 % 5", "2"),
        ("7 // 2", "3"),
        ("-(3 + 4)", "-7"),
        ("sqrt(16)", "4.0"),
        ("round(pi, 4)", "3.1416"),
        ("max(1, 99, 3)", "99"),
        ("factorial(5)", "120"),
    ],
)
def test_calculator_arithmetic(expression: str, expected: str) -> None:
    result = calculate(expression)
    assert result.ok, result.content
    assert expected in result.content


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "().__class__.__bases__[0].__subclasses__()",
        "open('/etc/passwd').read()",
        "eval('1+1')",
        "os.getcwd()",
        "lambda: 1",
        "[x for x in range(10)]",
        "'a' * 5",  # strings are not numbers
    ],
    ids=lambda s: s[:30],
)
def test_calculator_rejects_anything_that_is_not_arithmetic(expression: str) -> None:
    # eval() on model output would be remote code execution; the whitelist walk is
    # what makes this tool safe rather than merely convenient.
    result = calculate(expression)
    assert not result.ok
    assert result.error


def test_calculator_refuses_to_hang_on_a_huge_exponent() -> None:
    # 9**9**9 is short to type and allocates until the machine dies.
    result = calculate("9 ** 9 ** 9")
    assert not result.ok
    assert "exponent" in result.error or ""


def test_calculator_reports_division_by_zero_as_data() -> None:
    result = calculate("1 / 0")
    assert not result.ok
    assert "zero" in result.error.lower()


# --- file_ops ------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from research_agent import config

    monkeypatch.setattr(config, "settings", lambda: config.Settings(workspace_dir=tmp_path / "ws"))
    from research_agent.tools import file_ops as module

    monkeypatch.setattr(module, "settings", config.settings)
    return tmp_path / "ws"


def test_write_then_read_round_trips(workspace: Path) -> None:
    assert file_ops("write", "notes/findings.md", "405B vs 123B").ok
    result = file_ops("read", "notes/findings.md")
    assert result.ok
    assert "405B vs 123B" in result.content


@pytest.mark.parametrize(
    "path",
    [
        "../escape.txt",
        "../../etc/passwd",
        "/etc/passwd",
        "notes/../../../escape.txt",
        "./../../escape.txt",
    ],
)
def test_path_traversal_is_refused(workspace: Path, path: str) -> None:
    result = file_ops("read", path)
    assert not result.ok
    assert result.meta.get("rejected_by") == "path_confinement"


def test_symlink_out_of_the_workspace_is_refused(workspace: Path) -> None:
    # The reason paths are resolved before the prefix check rather than scanned for
    # ".." — a string check passes this and the file is read anyway.
    workspace.mkdir(parents=True, exist_ok=True)
    outside = workspace.parent / "secret.txt"
    outside.write_text("should never be readable")
    (workspace / "innocent.txt").symlink_to(outside)

    result = file_ops("read", "innocent.txt")

    assert not result.ok
    assert "should never be readable" not in result.content


def test_write_refuses_a_traversing_path_before_creating_anything(workspace: Path) -> None:
    result = file_ops("write", "../escaped.txt", "nope")
    assert not result.ok
    assert not (workspace.parent / "escaped.txt").exists()


def test_reading_a_missing_file_is_a_failure_not_a_crash(workspace: Path) -> None:
    result = file_ops("read", "nope.md")
    assert not result.ok
    assert "does not exist" in result.error


def test_list_shows_files_relative_to_the_workspace(workspace: Path) -> None:
    file_ops("write", "a.md", "one")
    file_ops("write", "sub/b.md", "two")
    result = file_ops("list", ".")
    assert result.ok
    assert "a.md" in result.content
    assert "sub/b.md" in result.content


def test_unknown_operation_is_reported(workspace: Path) -> None:
    result = file_ops("delete", "a.md")
    assert not result.ok
    assert "unknown operation" in result.error


def test_resolve_accepts_a_path_inside_the_workspace(workspace: Path) -> None:
    assert _resolve("fine.txt").name == "fine.txt"
    with pytest.raises(PathEscape):
        _resolve("../nope.txt")


# --- fetch_page extraction (pure, no network) -----------------------------------


def test_html_to_text_drops_scripts_and_keeps_prose() -> None:
    title, text = html_to_text(
        "<html><head><title>Llama 3.1</title>"
        "<style>body{color:red}</style></head><body>"
        "<script>var tracking = 1;</script>"
        "<nav>Home About</nav>"
        "<p>Llama 3.1 405B has 405 billion parameters.</p>"
        "<p>Released in 2024.</p>"
        "</body></html>"
    )
    assert title == "Llama 3.1"
    assert "405 billion parameters" in text
    assert "tracking" not in text
    assert "color:red" not in text
    assert "Home About" not in text


def test_html_to_text_separates_blocks_instead_of_running_them_together() -> None:
    _, text = html_to_text("<p>First.</p><p>Second.</p>")
    assert "First." in text and "Second." in text
    assert "First.Second." not in text


def test_html_to_text_decodes_common_entities() -> None:
    _, text = html_to_text("<p>Tom &amp; Jerry &mdash; 5 &lt; 10</p>")
    assert "Tom & Jerry — 5 < 10" in text


def test_fetch_page_rejects_non_http_schemes() -> None:
    for url in ("file:///etc/passwd", "ftp://example.com", "javascript:alert(1)", "not a url"):
        result = fetch_page(url)
        assert not result.ok, url


# --- web_search routing ---------------------------------------------------------


def fake_payload(title: str = "Result") -> dict[str, Any]:
    return {
        "answer": "A short summary.",
        "results": [{"title": title, "url": "https://example.com/a", "content": "Some content."}],
    }


def test_search_uses_tavily_and_mints_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    web_search._reset_circuit()
    monkeypatch.setattr(web_search, "_search_tavily", lambda *a, **k: fake_payload("Tavily hit"))

    result = web_search.web_search("llama 3.1 parameters")

    assert result.ok
    assert result.meta["backend"] == "tavily"
    assert result.meta["credits"] == 1
    assert "A short summary." in result.content
    assert [s.url for s in result.sources] == ["https://example.com/a"]


def test_search_falls_back_when_tavily_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    web_search._reset_circuit()

    def boom(*a: object, **k: object) -> dict[str, Any]:
        raise httpx.ConnectError("network down")

    monkeypatch.setattr(web_search, "_search_tavily", boom)
    monkeypatch.setattr(web_search, "_search_ddgs", lambda *a, **k: fake_payload("DDG hit"))

    result = web_search.web_search("anything")

    assert result.ok
    assert result.meta["backend"] == "ddgs"
    assert result.meta["credits"] == 0


def test_quota_exhaustion_trips_the_circuit_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    # Once Tavily says "out of credits", every later step in the run should skip it
    # rather than pay the latency of a call that is going to fail.
    web_search._reset_circuit()
    calls = {"tavily": 0}

    def exhausted(*a: object, **k: object) -> dict[str, Any]:
        calls["tavily"] += 1
        raise web_search.QuotaExhausted("out of credits")

    monkeypatch.setattr(web_search, "_search_tavily", exhausted)
    monkeypatch.setattr(web_search, "_search_ddgs", lambda *a, **k: fake_payload())

    web_search.web_search("first")
    web_search.web_search("second")
    web_search.web_search("third")

    assert calls["tavily"] == 1, "circuit breaker should stop retrying Tavily"
    web_search._reset_circuit()


def test_total_search_failure_tells_the_agent_what_to_do(monkeypatch: pytest.MonkeyPatch) -> None:
    web_search._reset_circuit()

    def boom(*a: object, **k: object) -> dict[str, Any]:
        raise RuntimeError("nope")

    monkeypatch.setattr(web_search, "_search_tavily", boom)
    monkeypatch.setattr(web_search, "_search_ddgs", boom)

    result = web_search.web_search("anything")

    assert not result.ok
    assert "Answer from what you already have" in result.content


def test_empty_query_is_refused_without_a_network_call() -> None:
    assert not web_search.web_search("   ").ok


# --- registry -------------------------------------------------------------------


def test_every_tool_exposes_a_strict_openai_schema() -> None:
    schemas = openai_schemas()
    assert len(schemas) == len(REGISTRY)
    for schema in schemas:
        function = schema["function"]
        assert schema["type"] == "function"
        assert function["description"]
        parameters = function["parameters"]
        # strict mode requires additionalProperties:false and every property listed
        # in required; a schema that violates this is rejected at request time.
        assert function["strict"] is True
        assert parameters["additionalProperties"] is False
        assert set(parameters["required"]) == set(parameters["properties"])


def test_an_empty_name_list_selects_no_tools_rather_than_all_of_them() -> None:
    # Regression: `names or list(REGISTRY)` treated an empty list as falsy and
    # returned every tool. That fires exactly when every tool has hit its per-run
    # cap, silently undoing the budget it was meant to enforce.
    assert openai_schemas([]) == []
    assert len(openai_schemas(None)) == len(REGISTRY)
    assert len(openai_schemas()) == len(REGISTRY)
    assert [s["function"]["name"] for s in openai_schemas(["calculator"])] == ["calculator"]


def test_dispatch_reports_an_unknown_tool_by_name() -> None:
    result = dispatch("nonexistent", {})
    assert not result.ok
    assert "unknown tool" in result.error


def test_dispatch_reports_bad_arguments_instead_of_raising() -> None:
    # The model will get these wrong; they need to come back as something it can read.
    missing = dispatch("calculator", {})
    assert not missing.ok and "missing required" in missing.error

    unexpected = dispatch("calculator", {"expression": "1+1", "typo": 1})
    assert not unexpected.ok and "unexpected" in unexpected.error

    wrong_shape = dispatch("calculator", ["1+1"])  # type: ignore[arg-type]
    assert not wrong_shape.ok


def test_dispatch_converts_a_raising_tool_into_a_result(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(expression: str) -> None:
        raise ValueError("kaboom")

    monkeypatch.setitem(
        registry.REGISTRY,
        "calculator",
        REGISTRY["calculator"].__class__(
            name="calculator",
            description="x",
            parameters=REGISTRY["calculator"].parameters,
            fn=explode,
            max_calls=1,
        ),
    )
    result = dispatch("calculator", {"expression": "1+1"})
    assert not result.ok
    assert "ValueError" in result.error


def test_every_tool_has_a_per_run_call_cap() -> None:
    # budget_verdict enforces these; a tool with no cap is an unbounded loop.
    for name, spec in REGISTRY.items():
        assert spec.max_calls > 0, name
