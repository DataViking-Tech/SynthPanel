"""sy-6kr: fresh-install smoke tests.

Builds the wheel from this checkout, installs it into a throwaway
virtualenv (no source on the path), and exercises the documented
agent-onboarding commands. Mirrors the ``clean-install-smoke`` CI job
so the contract is also locally reproducible.

These tests are slow (~25 s each — build + venv create + pip install +
import) so they're gated behind ``-m install_smoke`` and skipped by
default. CI runs them as a dedicated job; local devs run them ad hoc
before cutting a release.

To run::

    pytest tests/test_install_smoke.py -m install_smoke

The tests deliberately use a *fake* ``ANTHROPIC_API_KEY`` so ``doctor``
treats the env as "credentials present" without ever calling a
provider. Every smoke command must finish in seconds with exit code 0;
any network call would blow that budget and fail the test.
"""

from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Mark every test in the module so they share the gating flag.
pytestmark = pytest.mark.install_smoke


def _build_wheel(target_dir: Path) -> Path:
    """Build a wheel from the current checkout into ``target_dir``.

    Returns the wheel's Path. We deliberately reuse the parent's Python
    so the wheel's interpreter-tag matches what we'll install into.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    # ``--outdir`` keeps the build out of the repo's dist/ to avoid
    # racing with a local ``python -m build`` invocation.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(target_dir),
        ],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
    )
    wheels = sorted(target_dir.glob("synthpanel-*.whl"))
    assert wheels, f"no wheel produced in {target_dir}"
    return wheels[-1]


def _make_venv(venv_dir: Path) -> Path:
    """Create a fresh venv and return the path to its python executable."""
    venv.EnvBuilder(with_pip=True, clear=True).create(venv_dir)
    py = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python"
    # Ensure we're not accidentally inheriting the parent's site-packages.
    # The .pth files that editable installs drop into the parent env are
    # the most likely contaminant.
    assert py.exists(), f"venv python missing: {py}"
    return py


def _smoke_env() -> dict[str, str]:
    """Build the env passed to smoke subprocesses.

    Strips the parent's PYTHONPATH so the wheel install is the *only*
    way ``synthpanel`` can resolve — that's the whole point of the test.
    Adds a fake credential so ``doctor`` exits 0.
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["ANTHROPIC_API_KEY"] = "fake-key-for-smoke-test"
    return env


def _ensure_build() -> None:
    """Skip rather than fail when ``build`` is unavailable.

    Outside of CI, devs may run pytest before installing the ``dev``
    extra. We'd rather skip with a clear hint than burn through a
    confusing ModuleNotFoundError.
    """
    try:
        import build  # noqa: F401
    except ImportError:
        pytest.skip("install `build` first: pip install build")


@pytest.fixture(scope="module")
def installed_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Module-scoped fixture: build the wheel once, share across tests."""
    _ensure_build()
    return _build_wheel(tmp_path_factory.mktemp("wheel"))


def _install_and_smoke(
    tmp_path: Path,
    wheel: Path,
    extras: str = "",
    *,
    expect_mcp: bool = False,
) -> None:
    """Install ``wheel{extras}`` into a fresh venv and run the smoke commands.

    Raises ``subprocess.CalledProcessError`` (via ``check=True``) on any
    failure, which pytest renders with full stdout/stderr.
    """
    venv_dir = tmp_path / "smoke-venv"
    py = _make_venv(venv_dir)
    spec = f"{wheel}{extras}"

    # --upgrade pip so we know we're not testing against pip < 23.x's
    # quirky resolver. Quiet to keep CI logs scannable; on failure pytest
    # captures the subprocess output via check=True.
    subprocess.run(
        [str(py), "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [str(py), "-m", "pip", "install", "--quiet", spec],
        check=True,
        capture_output=True,
    )

    synthpanel = py.parent / ("synthpanel.exe" if os.name == "nt" else "synthpanel")
    assert synthpanel.exists(), f"entry-point binary missing after install: {synthpanel}"

    env = _smoke_env()

    # `--version` must work without any provider config. This is the
    # smallest possible CLI smoke — entry-point dispatch + package
    # metadata load.
    out = subprocess.run(
        [str(synthpanel), "--version"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert "synthpanel" in out.stdout.lower(), f"--version missing 'synthpanel': {out.stdout!r}"

    subprocess.run(
        [str(synthpanel), "--help"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    # whoami always exits 0 (informational) — it must not raise on a
    # fresh install with no credential store.
    subprocess.run(
        [str(synthpanel), "whoami"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    # doctor exits 0 iff: python OK, deps OK, credentials present,
    # checkpoint root writable, packs loaded. Our fake env satisfies
    # the credential check; the rest must come from the wheel.
    subprocess.run(
        [str(synthpanel), "doctor"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    if expect_mcp:
        # The [mcp] extra must let us import the server's top-level
        # FastMCP object. We don't actually serve over stdio — that
        # would block forever. Just prove the import surface lands.
        subprocess.run(
            [
                str(py),
                "-c",
                "from synth_panel.mcp.server import mcp; assert mcp is not None",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )


def test_pip_install_bare_wheel_yields_working_cli(
    installed_wheel: Path,
    tmp_path: Path,
) -> None:
    """``pip install synthpanel`` (no extras) must give us a CLI that
    survives --version / --help / whoami / doctor on a fresh venv."""
    _install_and_smoke(tmp_path, installed_wheel, extras="")


def test_pip_install_with_mcp_extra_yields_working_mcp_surface(
    installed_wheel: Path,
    tmp_path: Path,
) -> None:
    """``pip install 'synthpanel[mcp]'`` must additionally let us import
    the MCP server (FastMCP top-level object). This is the agent-onboarding
    happy path."""
    _install_and_smoke(tmp_path, installed_wheel, extras="[mcp]", expect_mcp=True)
