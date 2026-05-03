"""Verify the ``Accept: text/markdown`` content-negotiation surface.

Two surfaces under test:

1. ``scripts/render_site_markdown.py`` — the converter that produces
   ``site/**/*.md`` from each HTML page. Asserts (a) the converter
   produces non-empty output for every page, (b) the committed ``.md``
   matches a fresh render so contributors don't drift, and (c) basic
   content checks (the install command, version-independent content).

2. ``site/_worker.js`` — the Cloudflare Pages Advanced Mode worker.
   Verified by re-implementing its routing logic in Python and checking
   the JS file exposes the expected exports / handlers.

The worker is JS, but the project has no JS toolchain. We treat the
worker as a black-box file and assert structural properties that would
break if a refactor lost the contract.
"""

from __future__ import annotations

import contextlib
import importlib.util
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SITE_ROOT = REPO_ROOT / "site"
WORKER_PATH = SITE_ROOT / "_worker.js"


def _load_renderer():
    spec = importlib.util.spec_from_file_location(
        "render_site_markdown",
        REPO_ROOT / "scripts" / "render_site_markdown.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["render_site_markdown"] = module
    spec.loader.exec_module(module)
    return module


def _site_html_pages() -> list[Path]:
    return [p for p in sorted(SITE_ROOT.rglob("*.html")) if not p.name.endswith(".j2")]


# ---------------------------------------------------------------------------
# Coverage: every HTML page has a markdown sibling
# ---------------------------------------------------------------------------


def test_every_html_page_has_markdown_rendition() -> None:
    pages = _site_html_pages()
    assert pages, "site/ has no HTML pages — has the layout changed?"
    missing = [p for p in pages if not p.with_suffix(".md").exists()]
    assert not missing, (
        "Missing .md rendition for: "
        + ", ".join(str(p.relative_to(REPO_ROOT)) for p in missing)
        + "\nRun: python scripts/render_site_markdown.py"
    )


def test_committed_markdown_matches_fresh_render() -> None:
    """Catch drift: edit HTML without re-running the markdown renderer."""
    renderer = _load_renderer()
    pages = _site_html_pages()
    drifted: list[str] = []
    for html_path in pages:
        md_path = html_path.with_suffix(".md")
        committed = md_path.read_text(encoding="utf-8")
        title = renderer._extract_title(html_path.read_text(encoding="utf-8"))
        fresh = renderer.html_to_markdown(html_path.read_text(encoding="utf-8"), title=title)
        if committed != fresh:
            drifted.append(str(md_path.relative_to(REPO_ROOT)))
    assert not drifted, (
        "Markdown renditions are out of sync with their HTML sources: "
        + ", ".join(drifted)
        + "\nRun: python scripts/render_site_markdown.py"
    )


# ---------------------------------------------------------------------------
# Converter unit tests
# ---------------------------------------------------------------------------


def test_converter_strips_chrome() -> None:
    renderer = _load_renderer()
    html = """
    <html><head><title>Page</title><script>alert(1)</script>
    <style>body{color:red}</style></head><body>
    <nav>nav text</nav>
    <main><h1>Hello</h1><p>World.</p></main>
    <footer>foot text</footer>
    <script>noisy()</script>
    </body></html>
    """
    md = renderer.html_to_markdown(html, title="Page")
    assert "alert(1)" not in md
    assert "body{color:red}" not in md
    assert "nav text" not in md
    assert "foot text" not in md
    assert "# Hello" in md
    assert "World." in md


def test_converter_preserves_code_blocks() -> None:
    renderer = _load_renderer()
    html = "<main><pre><code>pip install synthpanel</code></pre></main>"
    md = renderer.html_to_markdown(html)
    assert "```" in md
    assert "pip install synthpanel" in md


def test_converter_renders_tables() -> None:
    renderer = _load_renderer()
    html = """
    <main>
    <table>
      <tr><th>Var</th><th>Effect</th></tr>
      <tr><td>FOO</td><td>does foo</td></tr>
      <tr><td>BAR</td><td>does bar</td></tr>
    </table>
    </main>
    """
    md = renderer.html_to_markdown(html)
    assert "| Var | Effect |" in md
    assert "|---|---|" in md
    assert "| FOO | does foo |" in md


def test_converter_skips_aria_hidden() -> None:
    renderer = _load_renderer()
    html = '<main><p>Open <span aria-hidden="true">→</span> link</p></main>'
    md = renderer.html_to_markdown(html)
    assert "→" not in md
    assert "Open" in md and "link" in md


def test_converter_attaches_link_to_card_heading() -> None:
    """Card pattern: <a href><h3>title</h3><p>desc</p></a> renders as
    a markdown heading whose text is the link."""
    renderer = _load_renderer()
    html = (
        '<main><a href="/mcp"><div><h3>MCP server</h3>'
        '<span aria-hidden="true">→</span></div>'
        "<p>Drop-in config.</p></a></main>"
    )
    md = renderer.html_to_markdown(html)
    assert "### [MCP server](/mcp)" in md
    assert "Drop-in config." in md


def test_landing_page_markdown_has_install_command() -> None:
    """Smoke check: the most important content survives conversion."""
    md = (SITE_ROOT / "index.md").read_text(encoding="utf-8")
    assert "pip install synthpanel" in md
    assert "synthpanel" in md
    assert "MCP" in md or "mcp" in md


def test_landing_page_markdown_drops_chrome() -> None:
    md = (SITE_ROOT / "index.md").read_text(encoding="utf-8")
    # No leaked CSS / JS.
    assert "<script" not in md
    assert "tailwind" not in md.lower()
    # No unprocessed HTML attributes.
    assert "class=" not in md
    assert "data-copy" not in md


# ---------------------------------------------------------------------------
# Worker contract — port the routing logic to Python and assert
# behaviour on representative inputs.
# ---------------------------------------------------------------------------


def _prefers_markdown(accept: str) -> bool:
    """Python port of ``prefersMarkdown`` in site/_worker.js."""
    if not accept:
        return False
    md_q = -1.0
    best_other_q = -1.0
    for raw in accept.split(","):
        part = raw.strip()
        if not part:
            continue
        segments = [s.strip() for s in part.split(";")]
        type_ = segments[0].lower()
        q = 1.0
        for seg in segments[1:]:
            if seg.startswith("q="):
                with contextlib.suppress(ValueError):
                    q = float(seg[2:])
        if q <= 0:
            continue
        if type_ == "text/markdown":
            md_q = max(md_q, q)
        elif type_ not in ("*/*", "text/*"):
            best_other_q = max(best_other_q, q)
    if md_q < 0:
        return False
    return md_q >= best_other_q


def _html_path_to_markdown(path: str) -> str | None:
    """Python port of ``htmlPathToMarkdown`` in site/_worker.js."""
    if not path:
        return None
    if path.endswith("/"):
        return path + "index.md"
    if path.endswith(".html"):
        return path[:-5] + ".md"
    last = path.split("/")[-1]
    if "." in last:
        return None
    return path + "/index.md"


def test_prefers_markdown_explicit() -> None:
    assert _prefers_markdown("text/markdown")
    assert _prefers_markdown("text/markdown, */*; q=0.1")
    assert _prefers_markdown("text/html;q=0.5, text/markdown;q=1.0")


def test_prefers_markdown_rejects_browsers() -> None:
    # Default browser Accept header — must NOT trigger markdown.
    assert not _prefers_markdown("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
    assert not _prefers_markdown("*/*")
    assert not _prefers_markdown("")


def test_prefers_markdown_zero_q() -> None:
    # ``q=0`` is a refusal; explicit refusal of markdown means false.
    assert not _prefers_markdown("text/markdown;q=0, text/html")


def test_html_path_to_markdown_mappings() -> None:
    assert _html_path_to_markdown("/") == "/index.md"
    assert _html_path_to_markdown("/recommended-models/") == "/recommended-models/index.md"
    assert _html_path_to_markdown("/recommended-models") == "/recommended-models/index.md"
    assert _html_path_to_markdown("/blog/post.html") == "/blog/post.md"
    assert _html_path_to_markdown("/og-image.png") is None
    assert _html_path_to_markdown("/sitemap.xml") is None


# ---------------------------------------------------------------------------
# Worker file shape
# ---------------------------------------------------------------------------


def test_worker_file_exists_and_exports_default() -> None:
    assert WORKER_PATH.exists(), "site/_worker.js missing"
    src = WORKER_PATH.read_text(encoding="utf-8")
    assert re.search(r"export\s+default\s*\{", src), (
        "site/_worker.js must export a default object with a fetch() handler"
    )
    assert "async fetch(" in src or "fetch:" in src, "site/_worker.js default export must define fetch()"


def test_worker_sets_required_response_headers() -> None:
    src = WORKER_PATH.read_text(encoding="utf-8")
    # The 200-path must set these three headers — they are the
    # contract the bead specifies (content-type + token count) plus the
    # cache directive that prevents misnegotiated responses.
    assert "text/markdown; charset=utf-8" in src, "Content-Type not set"
    assert '"vary"' in src.lower() or '"Vary"' in src, "Vary header not set"
    assert "x-markdown-tokens" in src, "x-markdown-tokens header not set"


def test_worker_returns_406_when_no_markdown_rendition() -> None:
    """Encode the contract: missing .md -> 406 Not Acceptable."""
    src = WORKER_PATH.read_text(encoding="utf-8")
    assert "406" in src, "Worker must return 406 when no rendition exists"
    assert "not_acceptable" in src or "Not Acceptable" in src
