"""Fetch one URL and return readable text.

Search snippets are often too thin to answer a multi-hop question — the agent needs
to open the page. This does the smallest useful version of that: a size-capped,
time-capped GET, with the HTML reduced to text by a small tag-stripping pass rather
than a readability dependency.

The extraction is deliberately crude and the README says so. Getting main-content
extraction right is its own project, and on the pages a research agent actually
lands on (Wikipedia, news, docs, arXiv) dropping script/style/nav and collapsing
whitespace recovers most of the signal.
"""

import re
from urllib.parse import urlparse

import httpx

from research_agent.config import settings
from research_agent.tools.base import Source, ToolResult, ToolSpec, truncate

MAX_PAGE_BYTES = 2 * 1024 * 1024
MAX_TEXT_CHARS = 6000
USER_AGENT = "research-agent/0.1 (+https://github.com/vireshkoli/research-agent-langgraph)"

# Whole elements whose text content is never page content.
_DROP_ELEMENTS = re.compile(
    r"<(script|style|noscript|svg|nav|footer|header|form|aside)\b.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_COMMENTS = re.compile(r"<!--.*?-->", re.DOTALL)
_BLOCK_END = re.compile(r"</(p|div|h[1-6]|li|tr|section|article|br)\s*>", re.IGNORECASE)
_TAGS = re.compile(r"<[^>]+>")
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_BLANK_LINES = re.compile(r"\n{3,}")
_SPACES = re.compile(r"[ \t]{2,}")

_ENTITIES = {
    "&nbsp;": " ",
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&#39;": "'",
    "&apos;": "'",
    "&mdash;": "—",
    "&ndash;": "–",
}


def html_to_text(html: str) -> tuple[str, str]:
    """Reduce HTML to (title, text). Pure and unit-testable — no network involved."""
    title_match = _TITLE.search(html)
    title = _unescape(_TAGS.sub("", title_match.group(1))).strip() if title_match else ""

    body = _COMMENTS.sub(" ", html)
    body = _DROP_ELEMENTS.sub(" ", body)
    # Turn block boundaries into newlines before stripping tags, so paragraphs do
    # not run together into one unreadable wall of text.
    body = _BLOCK_END.sub("\n", body)
    body = _TAGS.sub(" ", body)
    body = _unescape(body)
    body = _SPACES.sub(" ", body)
    body = "\n".join(line.strip() for line in body.splitlines())
    body = _BLANK_LINES.sub("\n\n", body).strip()
    return title, body


def _unescape(text: str) -> str:
    for entity, char in _ENTITIES.items():
        text = text.replace(entity, char)
    return text


def fetch_page(url: str) -> ToolResult:
    """GET `url` and return its readable text."""
    url = (url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return ToolResult.failure(f"only http and https URLs are supported, got {url!r}")
    if not parsed.netloc:
        return ToolResult.failure(f"not a valid URL: {url!r}")

    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=settings().fetch_timeout_seconds,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            response = client.get(url)
    except httpx.TimeoutException:
        return ToolResult.failure(f"timed out after {settings().fetch_timeout_seconds:g}s")
    except httpx.HTTPError as exc:
        return ToolResult.failure(f"request failed: {type(exc).__name__}: {exc}")

    if response.status_code >= 400:
        return ToolResult.failure(f"HTTP {response.status_code} from {url}")

    content_type = response.headers.get("content-type", "")
    if not any(kind in content_type for kind in ("text/html", "text/plain", "application/xhtml")):
        return ToolResult.failure(f"unsupported content type {content_type!r}")

    raw = response.text[:MAX_PAGE_BYTES]
    title, text = html_to_text(raw) if "html" in content_type else ("", raw)
    if not text.strip():
        return ToolResult.failure(f"no readable text extracted from {url}")

    body, raw_chars, was_truncated = truncate(text, MAX_TEXT_CHARS)
    heading = title or url
    return ToolResult(
        ok=True,
        content=f"{heading}\n{url}\n\n{body}",
        sources=[Source(url=url, title=title or url, snippet=text[:300], tool="fetch_page")],
        raw_chars=raw_chars,
        truncated=was_truncated,
        meta={"status": response.status_code, "final_url": str(response.url)},
    )


SPEC = ToolSpec(
    name="fetch_page",
    description=(
        "Fetch a web page by URL and return its readable text. Use this when a search "
        "snippet is too short to answer the question and you need the full page. Pass "
        "a URL that appeared in an earlier search result."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The absolute http(s) URL to fetch."}
        },
        "required": ["url"],
        "additionalProperties": False,
    },
    fn=fetch_page,
    max_calls=8,
)
