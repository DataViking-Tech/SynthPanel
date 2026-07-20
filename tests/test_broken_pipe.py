"""sy-4bs: piping CLI output into a short-circuiting consumer is silent.

Before this fix, ``althing … --help | head`` printed
``Exception ignored while flushing sys.stdout: BrokenPipeError`` to
stderr because Python's interpreter shutdown flush hit the closed pipe.
Agents that introspect Althing by grepping ``--help`` (the documented
pattern in README's Human Operator Quick Start) saw the traceback in
their tool-output streams and treated it as a tool failure.

Two layers under test:

1. The ``_quiet_broken_pipe`` helper directly — a fast unit test, no
   subprocess, validates the stdout-redirect-to-devnull behaviour.
2. The actual CLI under a head-truncated pipe — an integration test
   via subprocess that confirms zero stderr noise on a real pipe close.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- unit tests


def test_quiet_broken_pipe_restores_sigpipe_default() -> None:
    """The helper resets SIGPIPE to SIG_DFL so kernel-level pipe close kills
    the process silently instead of Python raising BrokenPipeError."""
    if not hasattr(signal, "SIGPIPE"):
        pytest.skip("SIGPIPE not available on this platform (Windows)")

    from althing.main import _quiet_broken_pipe

    prior = signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    try:
        _quiet_broken_pipe()
        assert signal.getsignal(signal.SIGPIPE) is signal.SIG_DFL
    finally:
        signal.signal(signal.SIGPIPE, prior)


def test_quiet_broken_pipe_redirects_stdout_when_flush_fails() -> None:
    """If flushing stdout raises BrokenPipeError, the helper must dup2
    /dev/null onto stdout's fd. Verified at the syscall layer via mocks so
    the test doesn't fight pytest's stdout capture."""
    from althing.main import _quiet_broken_pipe

    fake_stdout = mock.Mock()
    fake_stdout.flush.side_effect = BrokenPipeError(32, "Broken pipe")
    fake_stdout.fileno.return_value = 1

    devnull_fd = 999  # arbitrary sentinel; never actually used
    with (
        mock.patch.object(sys, "stdout", fake_stdout),
        mock.patch("os.open", return_value=devnull_fd) as fake_open,
        mock.patch("os.dup2") as fake_dup2,
    ):
        _quiet_broken_pipe()

    fake_open.assert_called_once_with(os.devnull, os.O_WRONLY)
    fake_dup2.assert_called_once_with(devnull_fd, 1)


def test_quiet_broken_pipe_swallows_secondary_oserror() -> None:
    """If the recovery path itself fails (eg detached stdout), the helper
    must not raise — that would leak a traceback back to the caller and
    re-create the original problem."""
    from althing.main import _quiet_broken_pipe

    fake_stdout = mock.Mock()
    fake_stdout.flush.side_effect = BrokenPipeError(32, "Broken pipe")
    fake_stdout.fileno.side_effect = ValueError("I/O operation on closed file")

    with mock.patch.object(sys, "stdout", fake_stdout):
        # Must not raise.
        _quiet_broken_pipe()


# ----------------------------------------------------- integration via pipe


def _run_piped(args: list[str], head_lines: int = 3) -> tuple[int, bytes, bytes]:
    """Run ``python -m althing <args> | head -N`` and return the pipeline
    exit code, head's stdout, and althing's stderr."""
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    sp = subprocess.Popen(
        [sys.executable, "-m", "althing", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=REPO_ROOT,
    )
    head = subprocess.Popen(
        ["head", "-n", str(head_lines)],
        stdin=sp.stdout,
        stdout=subprocess.PIPE,
    )
    assert sp.stdout is not None
    sp.stdout.close()
    head_out, _ = head.communicate(timeout=30)
    assert sp.stderr is not None
    sp_err = sp.stderr.read()
    sp.wait(timeout=10)
    return sp.returncode, head_out, sp_err


def _althing_importable() -> bool:
    """Skip integration tests when althing can't be imported (missing
    optional deps in the test environment). The unit tests above still
    cover the fix's contract."""
    try:
        import althing  # noqa: F401
        from althing.cli import parser  # noqa: F401

        return True
    except Exception:
        return False


needs_cli = pytest.mark.skipif(not _althing_importable(), reason="althing CLI not importable in this environment")


@needs_cli
def test_help_piped_to_head_produces_no_stderr_noise() -> None:
    """The exact failure mode from GH#499: argparse help through a
    short-circuiting consumer must not surface BrokenPipeError on stderr."""
    rc, _, err = _run_piped(["panel", "run", "--help"])
    assert b"BrokenPipeError" not in err, f"BrokenPipeError leaked to stderr: {err!r}"
    assert b"Exception ignored" not in err, f"shutdown-flush warning leaked to stderr: {err!r}"
    assert rc == 0, f"unexpected non-zero exit {rc}; stderr={err!r}"


@needs_cli
def test_top_level_help_piped_is_clean() -> None:
    rc, _, err = _run_piped(["--help"])
    assert b"BrokenPipeError" not in err
    assert b"Exception ignored" not in err
    assert rc == 0


@needs_cli
def test_subcommand_help_piped_is_clean() -> None:
    rc, _, err = _run_piped(["prompt", "--help"], head_lines=1)
    assert b"BrokenPipeError" not in err
    assert b"Exception ignored" not in err
    assert rc == 0


@needs_cli
def test_normal_listing_piped_to_head_is_clean() -> None:
    """Not help-only — any stdout-heavy subcommand should behave the same.
    `instruments list` enumerates the bundled v3 packs and easily overflows
    a `head -2` consumer."""
    rc, _, err = _run_piped(["instruments", "list"], head_lines=2)
    assert b"BrokenPipeError" not in err
    assert b"Exception ignored" not in err
    assert rc == 0
