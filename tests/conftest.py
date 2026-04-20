"""Global test fixtures — network access guard.

Blocks all outbound socket connections by default so that no unit test can
accidentally make live API calls. Tests marked ``@pytest.mark.acceptance``
are exempt (they run with real network access).

This makes the "test_alias_is_resolved_in_send hits live Anthropic API"
class of bugs structurally impossible in CI.
"""

from __future__ import annotations

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
