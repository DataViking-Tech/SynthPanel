"""Render markdown renditions of every ``site/**/*.html`` page.

Used by the ``Accept: text/markdown`` content-negotiation worker
(``site/_worker.js``) to serve agent-friendly markdown alongside the
human-facing HTML. See ``docs/cloudflare-pages-setup.md`` for the
deploy-side picture.

Approach: walk the publish dir, parse each ``.html`` with the stdlib
``html.parser`` HTMLParser, emit a structured markdown rendition next
to the source file (``foo.html`` -> ``foo.md``). Only content inside
``<main>`` is rendered — chrome elements (``<head>``, ``<script>``,
``<style>``, ``<svg>``, ``<button>``, ``<nav>``, ``<footer>``,
``aria-hidden`` decorations) are dropped on the floor.

Run directly (``python scripts/render_site_markdown.py``) or import
``render_all()`` from tests. Idempotent: rerun whenever site HTML
changes.
"""

from __future__ import annotations

import html.parser
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SITE_ROOT = REPO_ROOT / "site"

# Tags whose contents we drop on the floor entirely.
SKIP_TAGS = frozenset(
    {
        "head",
        "script",
        "style",
        "noscript",
        "svg",
        "path",
        "button",
        "form",
        "input",
        "select",
        "textarea",
        "footer",
        "nav",
        "meta",
        "link",
        "title",
    }
)

# Block-level tags that introduce a paragraph break in the output.
BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "header",
        "main",
        "aside",
        "blockquote",
        "details",
        "summary",
        "figure",
        "figcaption",
    }
)

HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

LIST_TAGS = frozenset({"ul", "ol"})

# Self-closing / void tags that ``handle_endtag`` will never see.
VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "keygen",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)

# Tags that, when seen inside an inline-buffered ``<a>``, force the
# anchor into "complex mode" — flush whatever inline label we have and
# stop buffering.
ANCHOR_BREAK_TAGS = HEADING_TAGS | BLOCK_TAGS | LIST_TAGS | {"li", "pre", "table"}


def _attr(attrs: list[tuple[str, str | None]], name: str) -> str | None:
    for k, v in attrs:
        if k == name:
            return v
    return None


class _Anchor:
    """Buffered state for a single ``<a>`` element."""

    __slots__ = ("buffer", "complex", "href", "label_emitted")

    def __init__(self, href: str | None) -> None:
        self.href: str | None = href
        self.buffer: list[str] = []
        # True once we've seen a block-level child and given up on
        # rendering this anchor as a single inline link.
        self.complex: bool = False
        # True once we've emitted ``[text](href)`` somewhere — either
        # via flush at first block child, or at </a> for inline anchors.
        self.label_emitted: bool = False


class _MarkdownEmitter(html.parser.HTMLParser):
    """Convert one HTML document into structured markdown."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        # Stack entries: (tag_name, hides_content_bool). Push on every
        # starttag, pop on every endtag. ``skip_depth`` is the count of
        # entries with ``hides=True`` currently on the stack.
        self._tag_stack: list[tuple[str, bool]] = []
        self.skip_depth = 0
        self.in_main = False
        self.main_depth = 0
        self.in_pre = False
        self.heading_level: int | None = None
        self.heading_buffer: list[str] | None = None
        # If a heading is opened immediately under an anchor whose
        # buffer is empty, we attach the anchor href to the heading
        # text. Tracked via this pending href.
        self._pending_heading_href: str | None = None
        self.anchor_stack: list[_Anchor] = []
        self.table_stack: list[list[list[str]]] = []
        self.row_buffer: list[str] | None = None
        self.cell_buffer: list[str] | None = None
        self._wrote_anything = False
        self._list_depth_value = 0

    # --- helpers ------------------------------------------------------

    def _active_anchor(self) -> _Anchor | None:
        return self.anchor_stack[-1] if self.anchor_stack else None

    def _current_target(self) -> list[str] | None:
        if self.cell_buffer is not None:
            return self.cell_buffer
        if self.heading_buffer is not None:
            return self.heading_buffer
        anchor = self._active_anchor()
        if anchor is not None and not anchor.complex:
            return anchor.buffer
        return self.out

    def _emit(self, text: str) -> None:
        if not text or self.skip_depth:
            return
        target = self._current_target()
        target.append(text)
        if target is self.out:
            self._wrote_anything = True

    def _emit_break(self) -> None:
        if self.skip_depth or not self._wrote_anything:
            return
        if self.cell_buffer is not None or self.heading_buffer is not None:
            return
        anchor = self._active_anchor()
        if anchor is not None and not anchor.complex:
            return
        tail = "".join(self.out[-3:]) if self.out else ""
        if tail.endswith("\n\n"):
            return
        if tail.endswith("\n"):
            self.out.append("\n")
        else:
            self.out.append("\n\n")

    @property
    def _list_depth(self) -> int:
        return self._list_depth_value

    @_list_depth.setter
    def _list_depth(self, value: int) -> None:
        self._list_depth_value = value

    def _list_prefix(self) -> str:
        # Always use ``-`` for list items — the source HTML often
        # renders the numeral itself via a styled span, so re-emitting
        # ``1.`` here would duplicate the numbering.
        return "  " * max(0, self._list_depth - 1) + "- "

    def _flush_anchor_label(self) -> None:
        """Promote the active anchor from inline to complex mode."""
        anchor = self._active_anchor()
        if anchor is None or anchor.complex:
            return
        label = _normalize_inline("".join(anchor.buffer))
        if label:
            if anchor.href:
                self.out.append(f"[{label}]({anchor.href})")
            else:
                self.out.append(label)
            self._wrote_anything = True
            anchor.label_emitted = True
        else:
            # Buffer empty — the next heading absorbs the href instead.
            if anchor.href and self._pending_heading_href is None:
                self._pending_heading_href = anchor.href
        anchor.complex = True

    # --- HTMLParser hooks --------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        is_main = tag == "main"
        hidden = _attr(attrs, "aria-hidden") == "true"
        is_skip_tag = tag in SKIP_TAGS

        # ``<main>`` itself never hides content — it is the gate that
        # enables rendering. Aria-hidden on <main> is treated as a
        # malformed source and ignored.
        if is_main:
            if tag not in VOID_TAGS:
                self._tag_stack.append((tag, False))
            self.in_main = True
            self.main_depth = self.main_depth + 1
            return

        is_void = tag in VOID_TAGS
        hides = hidden or is_skip_tag

        if not is_void:
            self._tag_stack.append((tag, hides))
            if hides:
                self.skip_depth += 1
                return

        if is_void and hides:
            return

        # Outside <main>, drop emission but don't touch skip_depth —
        # the gate is purely on ``in_main``.
        if not self.in_main:
            return
        if self.skip_depth:
            return

        anchor = self._active_anchor()
        if anchor is not None and not anchor.complex and tag in ANCHOR_BREAK_TAGS:
            self._flush_anchor_label()

        if tag == "table":
            self._emit_break()
            self.table_stack.append([])
            return

        if self.table_stack:
            if tag == "tr":
                self.row_buffer = []
                return
            if tag in {"td", "th"}:
                self.cell_buffer = []
                return

        if tag in HEADING_TAGS:
            self._emit_break()
            self.heading_level = int(tag[1])
            self.heading_buffer = []
            return

        if tag in LIST_TAGS:
            self._emit_break()
            self._list_depth = self._list_depth + 1
            return

        if tag == "li":
            self._emit_break()
            self.out.append(self._list_prefix())
            self._wrote_anything = True
            return

        if tag == "a":
            href = _attr(attrs, "href")
            self.anchor_stack.append(_Anchor(href))
            return

        if tag == "code":
            if not self.in_pre:
                self._emit("`")
            return

        if tag == "pre":
            self._emit_break()
            self.in_pre = True
            self.out.append("```\n")
            self._wrote_anything = True
            return

        if tag in {"strong", "b"}:
            self._emit("**")
            return

        if tag in {"em", "i"}:
            self._emit("*")
            return

        if tag == "br":
            self.out.append("\n")
            self._wrote_anything = True
            return

        if tag in BLOCK_TAGS:
            self._emit_break()
            return

    def handle_endtag(self, tag: str) -> None:
        # Pop our parallel stack first — that drives ``skip_depth``.
        last_hides = False
        if self._tag_stack:
            # Balance against most-recent open of this name. HTML can
            # be sloppy; tolerate small mismatches by scanning back.
            for i in range(len(self._tag_stack) - 1, -1, -1):
                if self._tag_stack[i][0] == tag:
                    _, hides = self._tag_stack.pop(i)
                    last_hides = hides
                    break
            else:
                # No match — ignore the close, keep stack as-is.
                pass
        if last_hides:
            self.skip_depth -= 1
            if tag == "main":
                # If <main> itself was hidden somehow, just unwind.
                return
            return

        if tag == "main":
            if self.main_depth > 0:
                self.main_depth -= 1
            if self.main_depth == 0:
                self.in_main = False
            return

        if not self.in_main or self.skip_depth:
            return

        if tag in {"td", "th"} and self.cell_buffer is not None and self.row_buffer is not None:
            cell = _normalize_inline("".join(self.cell_buffer))
            cell = cell.replace("|", "\\|")
            self.row_buffer.append(cell)
            self.cell_buffer = None
            return
        if tag == "tr" and self.row_buffer is not None and self.table_stack:
            if self.row_buffer:
                self.table_stack[-1].append(self.row_buffer)
            self.row_buffer = None
            return
        if tag == "table" and self.table_stack:
            rows = self.table_stack.pop()
            self._emit_table(rows)
            return

        if tag in HEADING_TAGS:
            text = _normalize_inline("".join(self.heading_buffer or []))
            self.heading_buffer = None
            level = self.heading_level or 1
            self.heading_level = None
            href = self._pending_heading_href
            self._pending_heading_href = None
            if not text:
                # Empty heading: drop it but keep the pending-href
                # reset so a later heading doesn't grab a stale link.
                return
            self._emit_break()
            if href:
                self.out.append(f"{'#' * level} [{text}]({href})\n\n")
                anchor = self._active_anchor()
                if anchor is not None:
                    anchor.label_emitted = True
            else:
                self.out.append(f"{'#' * level} {text}\n\n")
            self._wrote_anything = True
            return

        if tag in LIST_TAGS:
            self._list_depth = max(0, self._list_depth - 1)
            self.out.append("\n")
            return

        if tag == "li":
            self.out.append("\n")
            return

        if tag == "a":
            if not self.anchor_stack:
                return
            anchor = self.anchor_stack.pop()
            if not anchor.label_emitted and not anchor.complex:
                # Inline link.
                label = _normalize_inline("".join(anchor.buffer))
                if label:
                    if anchor.href:
                        rendered = f"[{label}]({anchor.href})"
                    else:
                        rendered = label
                    target = self._current_target()
                    target.append(rendered)
                    if target is self.out:
                        self._wrote_anything = True
            return

        if tag == "code":
            if not self.in_pre:
                self._emit("`")
            return

        if tag == "pre":
            if self.out and not self.out[-1].endswith("\n"):
                self.out.append("\n")
            self.out.append("```\n\n")
            self.in_pre = False
            return

        if tag in {"strong", "b"}:
            self._emit("**")
            return

        if tag in {"em", "i"}:
            self._emit("*")
            return

        if tag in BLOCK_TAGS:
            self._emit_break()
            return

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Self-closing tags like ``<br />`` and ``<img />`` don't push
        # the stack — but ``handle_starttag`` is still invoked for some
        # parsers. Guard explicitly to avoid imbalance.
        if tag == "br":
            if self.in_main and not self.skip_depth:
                self.out.append("\n")
                self._wrote_anything = True
            return

    def handle_data(self, data: str) -> None:
        if not self.in_main:
            return
        if self.skip_depth:
            return
        if self.in_pre:
            self._emit(data)
            return
        collapsed = re.sub(r"[ \t\r\n]+", " ", data)
        if not collapsed.strip():
            target = self._current_target()
            if target and not target[-1].endswith((" ", "\n")):
                self._emit(" ")
            return
        target = self._current_target()
        if target and target[-1].endswith("\n"):
            collapsed = collapsed.lstrip()
        self._emit(collapsed)

    def _emit_table(self, rows: list[list[str]]) -> None:
        if not rows:
            return
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        self._emit_break()
        header = rows[0]
        body_rows = rows[1:] if len(rows) > 1 else []
        self.out.append("| " + " | ".join(header) + " |\n")
        self.out.append("|" + "|".join(["---"] * width) + "|\n")
        for r in body_rows:
            self.out.append("| " + " | ".join(r) + " |\n")
        self.out.append("\n")
        self._wrote_anything = True


_INLINE_WS = re.compile(r"\s+")


def _normalize_inline(text: str) -> str:
    return _INLINE_WS.sub(" ", text).strip()


_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")


def _normalize(text: str) -> str:
    text = _BLANK_LINES.sub("\n\n", text)
    text = _TRAILING_WS.sub("\n", text)
    return text.rstrip() + "\n"


def html_to_markdown(html_text: str, *, title: str | None = None) -> str:
    parser = _MarkdownEmitter()
    parser.feed(html_text)
    parser.close()
    body = "".join(parser.out)
    body = _normalize(body)
    if title and not body.lstrip().startswith("# "):
        body = f"# {title.strip()}\n\n{body}"
    return body


_TITLE_RE = re.compile(r"<title>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def _extract_title(html_text: str) -> str | None:
    match = _TITLE_RE.search(html_text)
    if not match:
        return None
    raw = re.sub(r"\s+", " ", match.group(1)).strip()
    return raw or None


def render_file(html_path: Path) -> Path:
    """Render ``html_path`` to ``html_path.with_suffix('.md')``."""
    html_text = html_path.read_text(encoding="utf-8")
    title = _extract_title(html_text)
    md = html_to_markdown(html_text, title=title)
    out_path = html_path.with_suffix(".md")
    out_path.write_text(md, encoding="utf-8")
    return out_path


def render_all(*, root: Path = SITE_ROOT) -> list[Path]:
    """Render every ``*.html`` under ``root`` (recursive, skips ``*.j2``)."""
    outputs: list[Path] = []
    for html_path in sorted(root.rglob("*.html")):
        if html_path.name.endswith(".j2"):
            continue
        outputs.append(render_file(html_path))
    return outputs


def main() -> int:
    outputs = render_all()
    for path in outputs:
        rel = path.relative_to(REPO_ROOT)
        size = path.stat().st_size
        print(f"  {rel}  ({size} bytes)")
    print(f"Rendered {len(outputs)} markdown file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
