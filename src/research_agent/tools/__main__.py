"""Exercise every tool once, without the agent loop.

`uv run python -m research_agent.tools --demo`

Useful for confirming a fresh clone can actually reach the network and run the
sandbox before spending anything on LLM calls. Only web_search and fetch_page touch
the network; the rest are free and offline.
"""

import sys

from research_agent.tools.registry import REGISTRY, dispatch

DEMOS: list[tuple[str, dict[str, object]]] = [
    ("calculator", {"expression": "405 - 123"}),
    ("calculator", {"expression": "sqrt(2) * 100"}),
    ("code_execution", {"code": "import statistics\nprint(statistics.mean([405, 123, 70]))"}),
    ("code_execution", {"code": "import os"}),  # must be rejected by the screen
    ("file_ops", {"operation": "write", "path": "demo.md", "content": "hello"}),
    ("file_ops", {"operation": "read", "path": "demo.md", "content": None}),
    ("file_ops", {"operation": "read", "path": "../../../etc/passwd", "content": None}),
    ("web_search", {"query": "2024 Nobel Prize in Physics laureates"}),
    ("fetch_page", {"url": "https://example.com"}),
]


def main() -> int:
    if "--demo" not in sys.argv:
        print(__doc__)
        print(f"Registered tools: {', '.join(sorted(REGISTRY))}")
        return 0

    failures = 0
    for name, arguments in DEMOS:
        preview = {k: (str(v)[:40] if v is not None else None) for k, v in arguments.items()}
        print(f"\n=== {name} {preview}")
        result = dispatch(name, arguments)
        status = "ok" if result.ok else "FAILED"
        print(f"    [{status}] {result.content[:300]}")
        if result.sources:
            print(f"    sources: {len(result.sources)}")
        if result.meta:
            print(f"    meta: {result.meta}")
        # The two deliberately-rejected calls are expected failures, not problems.
        expected_failure = name == "code_execution" and arguments.get("code") == "import os"
        expected_failure |= name == "file_ops" and "etc/passwd" in str(arguments.get("path"))
        if not result.ok and not expected_failure:
            failures += 1

    print(f"\n{len(DEMOS)} calls, {failures} unexpected failure(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
