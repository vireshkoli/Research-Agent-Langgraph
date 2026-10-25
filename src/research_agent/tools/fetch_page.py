"""Fetch one URL and return readable text.

Search snippets are often too thin to answer a multi-hop question — the agent needs
to open the page. This does the smallest useful version of that: a size-capped,
time-capped GET, with the HTML reduced to text by a small tag-stripping pass rather
than a readability dependency.

The extraction is deliberately crude and the README says so. Getting main-content
extraction right is its own project, and on the pages a research agent actually
lands on (Wikipedia, news, docs, arXiv) dropping script/style/nav and collapsing
whitespace recovers most of the signal.

**Whose URL is this?** The model chooses what to fetch, and the model reads pages
written by anyone, so a fetched page can steer it toward an address of the page
author's choosing. Before this guard only the scheme was checked, and loopback,
private and link-local addresses were all reachable — including the Space's own
Gradio server on 127.0.0.1:7860 — with the response flowing to the model and into
the public Sources pane. Now:

- the *resolved* address is checked, not the hostname, so `localhost`, decimal
  forms like `2130706433` and IPv4-mapped IPv6 are all caught the same way
- redirects are followed by hand and every hop is checked, since a public URL can
  redirect to a private one
- the body is capped while it streams, not after it has all been buffered
- the citation points at the URL that served the bytes, not the one requested

**Stated limitation:** this does not stop DNS rebinding. httpx resolves the name
again when it connects, so a hostname that answers with a public address for the
check and a private one for the connection gets through. Pinning the connection to
the checked address is the fix, and it is more machinery than this agent warrants.
"""

import ipaddress
import re
import socket
from urllib.parse import urljoin, urlparse

import httpx

from research_agent.config import settings
from research_agent.tools.base import Source, ToolResult, ToolSpec, truncate

MAX_PAGE_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 5
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


def refusal(url: str) -> str | None:
    """Why `url` may not be fetched, or None if it may. Checks the resolved address."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"only http and https URLs are supported, got {url!r}"
    host = parsed.hostname
    if not host:
        return f"not a valid URL: {url!r}"
    if settings().allow_private_fetch:
        return None
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        resolved = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError, ValueError) as exc:
        return f"could not resolve {host!r}: {exc}"
    for *_, sockaddr in resolved:
        address = ipaddress.ip_address(str(sockaddr[0]).split("%")[0])
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        if not address.is_global or address.is_multicast:
            return f"{host!r} resolves to a non-public address ({address}); not fetched"
    return None


def fetch_page(url: str) -> ToolResult:
    """GET `url` and return its readable text."""
    url = (url or "").strip()
    timeout = settings().fetch_timeout_seconds

    try:
        with httpx.Client(
            follow_redirects=False, timeout=timeout, headers={"User-Agent": USER_AGENT}
        ) as client:
            redirects = 0
            for _ in range(MAX_REDIRECTS + 1):
                if reason := refusal(url):
                    return ToolResult.failure(reason, blocked=True)
                with client.stream("GET", url) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            return ToolResult.failure(f"redirect from {url} had no location")
                        url = urljoin(url, location)
                        redirects += 1
                        continue
                    if response.status_code >= 400:
                        return ToolResult.failure(f"HTTP {response.status_code} from {url}")
                    content_type = response.headers.get("content-type", "")
                    if not any(
                        kind in content_type
                        for kind in ("text/html", "text/plain", "application/xhtml")
                    ):
                        return ToolResult.failure(f"unsupported content type {content_type!r}")
                    data = _read_capped(response, MAX_PAGE_BYTES)
                    raw = data.decode(response.encoding or "utf-8", errors="replace")
                    final_url, status = str(response.url), response.status_code
                    break
            else:
                return ToolResult.failure(f"more than {MAX_REDIRECTS} redirects from the URL given")
    except httpx.TimeoutException:
        return ToolResult.failure(f"timed out after {timeout:g}s")
    except httpx.HTTPError as exc:
        return ToolResult.failure(f"request failed: {type(exc).__name__}: {exc}")

    title, text = html_to_text(raw) if "html" in content_type else ("", raw)
    if not text.strip():
        return ToolResult.failure(f"no readable text extracted from {final_url}")

    body, raw_chars, was_truncated = truncate(text, MAX_TEXT_CHARS)
    heading = title or final_url
    return ToolResult(
        ok=True,
        content=f"{heading}\n{final_url}\n\n{body}",
        sources=[
            Source(url=final_url, title=title or final_url, snippet=text[:300], tool="fetch_page")
        ],
        raw_chars=raw_chars,
        truncated=was_truncated,
        meta={"status": status, "final_url": final_url, "redirects": redirects, "bytes": len(data)},
    )


def _read_capped(response: httpx.Response, limit: int) -> bytes:
    """Read at most `limit` bytes, stopping the download rather than slicing after it."""
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_bytes():
        chunks.append(chunk)
        size += len(chunk)
        if size >= limit:
            break
    return b"".join(chunks)[:limit]


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
