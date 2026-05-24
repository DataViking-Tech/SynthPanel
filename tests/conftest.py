"""Global test fixtures — network access guard.

Blocks all outbound socket connections by default so that no unit test can
accidentally make live API calls. Tests marked ``@pytest.mark.acceptance``
are exempt (they run with real network access).

This makes the "test_alias_is_resolved_in_send hits live Anthropic API"
class of bugs structurally impossible in CI.
"""

from __future__ import annotations

import signal
import socket
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Original socket.socket.connect — saved once at import time
# ---------------------------------------------------------------------------
_real_connect = socket.socket.connect


def _guarded_connect(self: socket.socket, address: object) -> None:
    """Block outbound connections unless the test is marked ``acceptance``."""
    # Allow localhost / Unix-domain connections (test servers, databases, etc.)
    if isinstance(address, tuple) and len(address) >= 2:
        host = str(address[0])
        if host in ("127.0.0.1", "::1", "localhost"):
            return _real_connect(self, address)

    raise RuntimeError(
        f"Network access blocked in tests (attempted connection to {address!r}). "
        "If this test requires a live API call, mark it with @pytest.mark.acceptance."
    )


@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-applied fixture that blocks network for non-acceptance tests."""
    markers = {m.name for m in request.node.iter_markers()}
    if "acceptance" in markers:
        return  # Let acceptance tests use real network
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)


@pytest.fixture(autouse=True)
def _isolate_credentials_store(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the SynthPanel credential store at a unique tmp path per test.

    Prevents a developer's real ``~/.config/synthpanel/credentials.json``
    from bleeding into tests that assume ``MISSING_CREDENTIALS`` (sp-lve).
    Individual tests can still write to the path by calling
    :func:`synth_panel.credentials.save_credential`.
    """
    sandbox: Path = tmp_path_factory.mktemp("synthpanel-creds")
    monkeypatch.setenv("SYNTHPANEL_CREDENTIALS_PATH", str(sandbox / "credentials.json"))


@pytest.fixture(autouse=True)
def _restore_sigpipe_disposition():
    """Snapshot and restore the process-global SIGPIPE handler around each test.

    ``synth_panel.main.main`` and its ``_quiet_broken_pipe`` helper deliberately
    install ``SIGPIPE=SIG_DFL`` so piped CLI output (``synthpanel … | head``)
    ends silently like a normal Unix tool. The disposition is *process-global*,
    so a test that calls ``main()`` or ``_quiet_broken_pipe()`` can leave
    SIG_DFL installed for the remainder of the pytest session. After that, any
    broken-pipe write during pytest's output capture or coverage teardown is
    delivered as SIGPIPE and silently kills the runner with exit 141 — the
    non-deterministic CI failure tracked under sy-1n1 / sy-6zq (the matrix
    entry that dies varies run to run because it depends on test ordering).

    Restoring the prior handler after every test makes that leak structurally
    impossible regardless of which code paths a test exercises, in the same
    spirit as the network-block fixture above.
    """
    if not hasattr(signal, "SIGPIPE"):
        # Windows has no SIGPIPE; nothing to guard.
        yield
        return
    prev = signal.getsignal(signal.SIGPIPE)
    try:
        yield
    finally:
        # ``getsignal`` returns None when the handler was installed from
        # non-Python code; ``signal.signal`` cannot reinstall that, so skip it.
        if prev is not None:
            signal.signal(signal.SIGPIPE, prev)
