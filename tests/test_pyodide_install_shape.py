"""Tests for sy-v8z: `trafilatura` moved out of core dependencies into the
``althing[full]`` optional extra.

The bead was opened from boardroom bo-2ra after `althing 1.2.0`'s
`pyodide_safe_mode` runtime flag was discovered not to reach the
*import-time* dependency cliff. `althing 1.2.0` listed `trafilatura`
in `[project.dependencies]`, which hard-requires `lxml` (a C extension
absent from the Cloudflare Python Workers curated set). `import althing`
therefore failed at Worker boot regardless of any runtime flag.

These tests pin the install-shape contract so a future PR can't silently
re-introduce the cliff:

1. **`trafilatura` is not in `[project.dependencies]`.** A grep of the
   actual `pyproject.toml` so the test fails loud the moment someone
   re-adds it. Implementation deferred to `tomllib`/`tomli` so the
   assertion runs even on minimum-supported Python (3.10).
2. **`trafilatura` is in `[project.optional-dependencies][full]`.** Pins
   the extras name so consumer documentation and migration guides keep
   working.
3. **`import althing.ensemble` succeeds when `trafilatura` is
   unimportable.** Simulates the pyodide environment by installing a
   meta-path finder that raises `ImportError` for `trafilatura` (and
   `lxml`) and re-imports `althing.ensemble` in a clean subprocess.
4. **`synthesize_panel(..., pyodide_safe_mode=True, judge_enabled=False)`
   works without trafilatura installed.** End-to-end behavioural mirror
   of the boardroom pyodide invocation. Also a subprocess so the assertion
   doesn't depend on the test process having already imported trafilatura.

Items 3 and 4 are subprocess-based because pytest's own dependency tree
imports `trafilatura` (via the `dev` extra) before any test runs — we
can't reliably unimport it from inside the test process.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - py < 3.11
    import tomli as tomllib  # type: ignore[import-not-found, no-redef]


_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"


# ---------------------------------------------------------------------------
# Install-shape contract — pyproject.toml structure.
# ---------------------------------------------------------------------------


class TestInstallShape:
    """Pin the contract that `pip install althing` (no extras) does
    not pull `trafilatura` (and therefore not `lxml`)."""

    @pytest.fixture(scope="class")
    def project_table(self) -> dict:
        with _PYPROJECT.open("rb") as fh:
            return tomllib.load(fh)["project"]

    def test_trafilatura_not_in_core_dependencies(self, project_table: dict) -> None:
        core = project_table["dependencies"]
        offenders = [d for d in core if d.lower().startswith("trafilatura")]
        assert offenders == [], (
            "trafilatura must not be in [project.dependencies] — it pulls "
            f"lxml (C ext) and blocks pyodide. Found: {offenders}"
        )

    def test_lxml_not_in_core_dependencies(self, project_table: dict) -> None:
        """Belt-and-suspenders: even if someone adds `lxml` directly,
        catch it before it ships."""
        core = project_table["dependencies"]
        offenders = [d for d in core if d.lower().startswith("lxml")]
        assert offenders == [], (
            f"lxml must not be in [project.dependencies] — it has no pyodide wheel. Found: {offenders}"
        )

    def test_full_extra_exists_and_contains_trafilatura(self, project_table: dict) -> None:
        extras = project_table["optional-dependencies"]
        assert "full" in extras, (
            "Expected `full` optional-dependencies extra (the documented "
            "install path for the fetch ladder). Got extras: "
            f"{sorted(extras.keys())}"
        )
        full = extras["full"]
        in_full = [d for d in full if d.lower().startswith("trafilatura")]
        assert in_full, f"Expected trafilatura in [project.optional-dependencies].full. Got: {full}"


# ---------------------------------------------------------------------------
# Import-time pyodide simulation — subprocess-based.
# ---------------------------------------------------------------------------


_BLOCK_TRAFILATURA_HEADER = """
import sys
import importlib.abc
import importlib.machinery

_BLOCKED = {"trafilatura", "lxml"}


class _BlockFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):  # noqa: ARG002
        head = fullname.split(".", 1)[0]
        if head in _BLOCKED:
            raise ImportError(
                f"simulated pyodide: '{fullname}' is not available"
            )
        return None


# Drop any pre-imported state so the block is honoured even on re-import.
for _mod in list(sys.modules):
    _head = _mod.split(".", 1)[0]
    if _head in _BLOCKED:
        del sys.modules[_mod]

sys.meta_path.insert(0, _BlockFinder())
"""


def _run_under_pyodide_sim(body: str) -> subprocess.CompletedProcess[str]:
    """Run ``body`` in a subprocess that raises ImportError for
    trafilatura / lxml — the canonical pyodide curated-set shape."""
    script = _BLOCK_TRAFILATURA_HEADER + "\n" + body
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=60,
    )


class TestPyodideImportCliff:
    """The canonical Cloudflare Python Workers shape: trafilatura and lxml
    are unimportable. `althing.ensemble` and `synthesize_panel` must
    work anyway."""

    def test_import_althing_ensemble_without_trafilatura(self) -> None:
        result = _run_under_pyodide_sim(
            "import althing.ensemble\n"
            "import sys\n"
            "assert 'trafilatura' not in sys.modules, sys.modules.get('trafilatura')\n"
            "assert 'lxml' not in sys.modules, sys.modules.get('lxml')\n"
            "print('ok')\n"
        )
        assert result.returncode == 0, (
            f"`import althing.ensemble` failed under pyodide-sim.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "ok" in result.stdout

    def test_synthesize_panel_pyodide_safe_mode_without_trafilatura(self) -> None:
        """End-to-end mirror of the boardroom Workers invocation:
        synthesize_panel must produce a SynthesisResult in
        pyodide_safe_mode + judge_enabled=False without trafilatura."""
        body = """
from althing.cost import ZERO_USAGE
from althing.ensemble import SynthesisResult, synthesize_panel
from althing.orchestrator import PanelistResult

questions = [{"text": "What do you think?"}]
panelists = [
    PanelistResult(
        persona_name="Alice",
        responses=[{"question": "What do you think?", "response": "Fine."}],
        usage=ZERO_USAGE,
    )
]

result = synthesize_panel(
    None,
    panelists,
    questions,
    judge_enabled=False,
    pyodide_safe_mode=True,
)
assert isinstance(result, SynthesisResult)

import sys
assert "trafilatura" not in sys.modules
assert "lxml" not in sys.modules
print("ok")
"""
        result = _run_under_pyodide_sim(body)
        assert result.returncode == 0, (
            f"synthesize_panel failed under pyodide-sim.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "ok" in result.stdout


# ---------------------------------------------------------------------------
# Regression — fetch ladder still works when [full] *is* installed.
# ---------------------------------------------------------------------------


class TestFullExtraRegression:
    """When `althing[full]` is installed (the dev environment is one),
    the trafilatura step is reachable and `_step_trafilatura` is wired
    in. We don't exercise an end-to-end fetch here — `test_fetch_lower.py`
    already does — but we pin the symbol and skip cleanly when the extra
    isn't present."""

    def test_step_trafilatura_returns_string_or_none(self) -> None:
        trafilatura = pytest.importorskip("trafilatura")
        assert trafilatura is not None  # extra installed
        from althing.fetch.ladder import _step_trafilatura

        html = b"<html><body><article><p>Hello world.</p></article></body></html>"
        out = _step_trafilatura(html, "https://example.com/")
        # trafilatura may return None for very small fragments; the
        # contract is "string or None", never raise.
        assert out is None or isinstance(out, str)
