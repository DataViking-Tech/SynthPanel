"""Guard the precompiled Tailwind stylesheet for synthpanel.dev (gh-563).

The site used to load the Tailwind *Play* CDN (``cdn.tailwindcss.com``) on
every page — a render-blocking ~300KB third-party script that Tailwind
documents as not-for-production, left pages unstyled without JS, and forced
a CSP allowlist entry. The stylesheet is now precompiled to
``site/assets/tailwind.css`` by the pinned node toolchain in
``scripts/site_tailwind/`` (``npm ci && npm run build``) and committed,
because Cloudflare Pages deploys ``site/`` raw with no build step.

These tests keep that arrangement honest:

* no page (or the CSP in ``site/_headers``) may reference the Play CDN;
* the committed stylesheet must exist and be non-trivially sized;
* every page must link the stylesheet;
* every class used in the HTML must have a matching rule in the committed
  CSS — the pure-Python drift guard that runs in the test matrix. CI's
  ``site-tailwind-drift`` job additionally rebuilds with the pinned
  toolchain and diffs byte-for-byte.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SITE_ROOT = REPO_ROOT / "site"
CSS_PATH = SITE_ROOT / "assets" / "tailwind.css"
STYLESHEET_LINK = '<link rel="stylesheet" href="/assets/tailwind.css" />'


def _site_pages() -> list[Path]:
    pages = [*sorted(SITE_ROOT.rglob("*.html")), SITE_ROOT / "index.html.j2"]
    assert len(pages) >= 7, f"expected the full site page set, found only {len(pages)}"
    return pages


def test_no_page_references_the_play_cdn() -> None:
    offenders = [p for p in _site_pages() if "cdn.tailwindcss.com" in p.read_text()]
    assert not offenders, (
        f"Pages still reference the Tailwind Play CDN (not for production): "
        f"{[str(p.relative_to(REPO_ROOT)) for p in offenders]}"
    )


def test_csp_does_not_allowlist_the_play_cdn() -> None:
    headers = (SITE_ROOT / "_headers").read_text()
    assert "cdn.tailwindcss.com" not in headers, (
        "site/_headers CSP still allowlists cdn.tailwindcss.com — nothing loads "
        "from it anymore, drop it to shrink the script-src attack surface"
    )


def test_precompiled_stylesheet_exists_and_is_nontrivial() -> None:
    assert CSS_PATH.exists(), (
        "site/assets/tailwind.css is missing. Regenerate it: cd scripts/site_tailwind && npm ci && npm run build"
    )
    size = CSS_PATH.stat().st_size
    assert size > 10_000, (
        f"site/assets/tailwind.css is suspiciously small ({size} bytes) — a full "
        "build of the site's class usage minifies to well over 10KB"
    )


def test_every_page_links_the_precompiled_stylesheet() -> None:
    missing = [p for p in _site_pages() if STYLESHEET_LINK not in p.read_text()]
    assert not missing, (
        f"Pages missing the precompiled stylesheet link {STYLESHEET_LINK!r}: "
        f"{[str(p.relative_to(REPO_ROOT)) for p in missing]}"
    )


def _classes_defined_in(css: str) -> set[str]:
    """Class names with a selector in ``css``, unescaped (``sm\\:px-6`` -> ``sm:px-6``)."""
    return {match.group(1).replace("\\", "") for match in re.finditer(r"\.((?:\\.|[a-zA-Z0-9_-])+)", css)}


@pytest.mark.parametrize("page", _site_pages(), ids=lambda p: str(p.relative_to(SITE_ROOT)))
def test_all_classes_used_by_page_are_compiled(page: Path) -> None:
    """Every class in the page resolves to a rule in the committed stylesheet.

    Non-Tailwind classes styled by the page's own inline ``<style>`` block
    (e.g. ``copy-btn``) are exempt. Anything else missing means the committed
    CSS has drifted from the markup — rebuild via scripts/site_tailwind/.
    """
    compiled = _classes_defined_in(CSS_PATH.read_text())
    html = page.read_text()
    inline_css = " ".join(re.findall(r"<style[^>]*>(.*?)</style>", html, re.S))

    missing: set[str] = set()
    for match in re.finditer(r'class="([^"]+)"', html):
        for token in match.group(1).split():
            if token in compiled or "." + token in inline_css:
                continue
            missing.add(token)

    assert not missing, (
        f"{page.relative_to(REPO_ROOT)} uses classes with no rule in "
        f"site/assets/tailwind.css: {sorted(missing)}. Rebuild the stylesheet: "
        "cd scripts/site_tailwind && npm ci && npm run build"
    )
