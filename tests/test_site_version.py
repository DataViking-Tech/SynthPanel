"""Verify the site template renders the canonical version correctly.

Replaces the sp-lwy drift guard. Drift between ``pyproject.toml`` / package
metadata and ``site/index.html`` is now structurally impossible: both derive
from ``src/synth_panel/__version__.py``. This test's job is ensuring the
template actually renders and still surfaces the version in every place the
landing page relies on it.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_render_site():
    spec = importlib.util.spec_from_file_location("render_site", REPO_ROOT / "scripts" / "render_site.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["render_site"] = module
    spec.loader.exec_module(module)
    return module


def test_template_renders_with_current_version() -> None:
    from synth_panel import __version__

    rendered = _load_render_site().render()

    assert f"v{__version__} — public beta" in rendered, "hero badge did not render with the current version"
    assert f"MIT-licensed · v{__version__} ·" in rendered, "footer did not render with the current version"

    jsonld = re.search(r'"softwareVersion":\s*"([^"]+)"', rendered)
    assert jsonld, "JSON-LD softwareVersion missing from rendered site"
    assert jsonld.group(1) == __version__, (
        f"JSON-LD softwareVersion {jsonld.group(1)} != package __version__ {__version__}"
    )

    # No unrendered placeholders should leak into the output.
    assert "{{" not in rendered and "}}" not in rendered, "unrendered template placeholders found in output"


def test_committed_index_matches_template_render() -> None:
    """Catch drift between site/index.html.j2 and the committed site/index.html.

    If a contributor edits the template without re-running render_site.py, or
    edits the rendered file directly, this test fails — keeping the committed
    artifact honest even when Cloudflare Pages deploys it as-is.
    """
    rendered = _load_render_site().render()
    committed = (REPO_ROOT / "site" / "index.html").read_text()
    assert rendered == committed, (
        "site/index.html is out of sync with site/index.html.j2. Run: python scripts/render_site.py"
    )
