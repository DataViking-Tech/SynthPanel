"""Tests for ``synth_panel.integrations.composio``.

The real ``composio`` package is intentionally not a hard dependency, so
these tests exercise the adapter against a minimal stub that imitates the
surface we rely on: ``composio_client.experimental.Toolkit`` and the
``@toolkit.tool()`` decorator. That's enough to verify the module builds
a well-formed toolkit, registers the right number of actions with the
right slugs, wires each tool's docstring and input schema, and delegates
to the SynthPanel SDK with the arguments the LLM passed in.

Two adversarial paths are also covered:

* Import-guard: when ``composio`` is absent, importing the adapter
  succeeds (cheap) but calling :func:`synthpanel_toolkit` raises a
  targeted :class:`ComposioNotInstalledError` with install instructions.
* Shape-guard: when the client lacks ``experimental.Toolkit`` (older
  Composio), the adapter raises :class:`RuntimeError` rather than
  crashing deep in the Pydantic layer.
"""

from __future__ import annotations

import builtins
import sys
from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Minimal Composio stub
# ---------------------------------------------------------------------------


class _StubTool:
    """Registered tool — mimics a Composio tool object well enough."""

    def __init__(self, func: Any, name: str, description: str | None) -> None:
        self.func = func
        self.name = name
        self.description = description


class _StubToolkit:
    def __init__(self, slug: str, name: str, description: str) -> None:
        self.slug = slug
        self.name = name
        self.description = description
        self.tools: list[_StubTool] = []

    def tool(self):
        def _decorator(func: Any) -> Any:
            self.tools.append(
                _StubTool(
                    func=func,
                    name=func.__name__,
                    description=(func.__doc__ or "").strip().splitlines()[0] if func.__doc__ else None,
                )
            )
            return func

        return _decorator


class _StubExperimental:
    def __init__(self) -> None:
        self.last_toolkit: _StubToolkit | None = None

    def Toolkit(self, *, slug: str, name: str, description: str) -> _StubToolkit:
        tk = _StubToolkit(slug=slug, name=name, description=description)
        self.last_toolkit = tk
        return tk


class _StubComposio:
    def __init__(self) -> None:
        self.experimental = _StubExperimental()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_client() -> _StubComposio:
    return _StubComposio()


@pytest.fixture(autouse=True)
def _fake_composio_module(monkeypatch):
    """Pretend ``composio`` is importable so the adapter proceeds.

    The adapter gates its factory behind ``import composio``; since the
    real package is not in CI, we inject a placeholder module. Individual
    tests that exercise the *missing-import* path remove it again.
    """
    if "composio" not in sys.modules:
        import types

        monkeypatch.setitem(sys.modules, "composio", types.ModuleType("composio"))
    yield


# ---------------------------------------------------------------------------
# Surface / shape
# ---------------------------------------------------------------------------


class TestToolkitShape:
    def test_returns_toolkit_with_expected_slug_and_name(self, stub_client):
        from synth_panel.integrations.composio import (
            TOOLKIT_DESCRIPTION,
            TOOLKIT_NAME,
            TOOLKIT_SLUG,
            synthpanel_toolkit,
        )

        tk = synthpanel_toolkit(stub_client)
        assert tk.slug == TOOLKIT_SLUG == "SYNTHPANEL"
        assert tk.name == TOOLKIT_NAME == "SynthPanel"
        assert tk.description == TOOLKIT_DESCRIPTION
        assert "synthetic focus groups" in tk.description.lower()

    def test_registers_exactly_five_actions(self, stub_client):
        from synth_panel.integrations.composio import synthpanel_toolkit

        tk = synthpanel_toolkit(stub_client)
        names = sorted(t.name for t in tk.tools)
        assert names == [
            "get_panel_result",
            "list_instruments",
            "list_personas",
            "quick_poll",
            "run_panel",
        ]

    def test_every_action_has_a_docstring(self, stub_client):
        """Composio surfaces docstrings to the LLM — none may be blank."""
        from synth_panel.integrations.composio import synthpanel_toolkit

        tk = synthpanel_toolkit(stub_client)
        for tool in tk.tools:
            assert tool.func.__doc__ and tool.func.__doc__.strip(), f"{tool.name} must have a non-empty docstring"


# ---------------------------------------------------------------------------
# Delegation to the SDK
# ---------------------------------------------------------------------------


class _FakePoll:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def to_dict(self) -> dict[str, Any]:
        return self._payload


class TestDelegation:
    def test_quick_poll_forwards_to_sdk(self, stub_client):
        from synth_panel.integrations.composio import synthpanel_toolkit

        tk = synthpanel_toolkit(stub_client)
        quick_poll_tool = next(t for t in tk.tools if t.name == "quick_poll")

        # Pydantic input instance — the decorator passes it positionally.
        QuickPollInput = quick_poll_tool.func.__annotations__["request"]
        req = QuickPollInput(
            question="Is $49/mo fair for an AI bookkeeper?",
            pack_id="general-consumer",
            model="haiku",
        )

        fake_result = _FakePoll({"result_id": "r1", "question": req.question})
        with patch("synth_panel.sdk.quick_poll", return_value=fake_result) as mocked:
            out = quick_poll_tool.func(req, ctx=None)

        mocked.assert_called_once_with(
            question="Is $49/mo fair for an AI bookkeeper?",
            personas=None,
            pack_id="general-consumer",
            model="haiku",
            synthesis=True,
        )
        assert out == {"result_id": "r1", "question": req.question}

    def test_run_panel_forwards_instrument_pack(self, stub_client):
        from synth_panel.integrations.composio import synthpanel_toolkit

        tk = synthpanel_toolkit(stub_client)
        run_panel_tool = next(t for t in tk.tools if t.name == "run_panel")
        RunPanelInput = run_panel_tool.func.__annotations__["request"]

        req = RunPanelInput(instrument_pack="pricing-discovery", pack_id="smb-owners")
        fake_result = _FakePoll({"result_id": "r2", "rounds": []})
        with patch("synth_panel.sdk.run_panel", return_value=fake_result) as mocked:
            out = run_panel_tool.func(req, ctx=None)

        kwargs = mocked.call_args.kwargs
        assert kwargs["instrument_pack"] == "pricing-discovery"
        assert kwargs["pack_id"] == "smb-owners"
        assert kwargs["personas"] is None
        assert out["result_id"] == "r2"

    def test_list_personas_returns_dict_with_packs_key(self, stub_client):
        from synth_panel.integrations.composio import synthpanel_toolkit

        tk = synthpanel_toolkit(stub_client)
        list_personas_tool = next(t for t in tk.tools if t.name == "list_personas")
        EmptyInput = list_personas_tool.func.__annotations__["request"]

        with patch(
            "synth_panel.sdk.list_personas",
            return_value=[{"id": "general-consumer", "name": "General", "persona_count": 5, "builtin": True}],
        ):
            out = list_personas_tool.func(EmptyInput(), ctx=None)

        assert "packs" in out
        assert out["packs"][0]["id"] == "general-consumer"

    def test_list_instruments_returns_dict_with_packs_key(self, stub_client):
        from synth_panel.integrations.composio import synthpanel_toolkit

        tk = synthpanel_toolkit(stub_client)
        tool = next(t for t in tk.tools if t.name == "list_instruments")
        EmptyInput = tool.func.__annotations__["request"]

        with patch(
            "synth_panel.sdk.list_instruments",
            return_value=[{"id": "pricing-discovery", "version": "3"}],
        ):
            out = tool.func(EmptyInput(), ctx=None)

        assert out == {"packs": [{"id": "pricing-discovery", "version": "3"}]}

    def test_get_panel_result_forwards_result_id(self, stub_client):
        from synth_panel.integrations.composio import synthpanel_toolkit

        tk = synthpanel_toolkit(stub_client)
        tool = next(t for t in tk.tools if t.name == "get_panel_result")
        GetInput = tool.func.__annotations__["request"]

        fake_result = _FakePoll({"result_id": "r3", "synthesis": {"recommendation": "ok"}})
        with patch("synth_panel.sdk.get_panel_result", return_value=fake_result) as mocked:
            out = tool.func(GetInput(result_id="r3"), ctx=None)

        mocked.assert_called_once_with("r3")
        assert out["result_id"] == "r3"


# ---------------------------------------------------------------------------
# Guard paths
# ---------------------------------------------------------------------------


class TestGuards:
    def test_missing_composio_raises_targeted_error(self, monkeypatch, stub_client):
        """If composio cannot be imported, factory raises with install hint."""
        monkeypatch.delitem(sys.modules, "composio", raising=False)

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "composio":
                raise ImportError("No module named 'composio'")
            return real_import(name, *args, **kwargs)

        from synth_panel.integrations.composio import (
            ComposioNotInstalledError,
            synthpanel_toolkit,
        )

        with (
            patch("builtins.__import__", side_effect=_fake_import),
            pytest.raises(ComposioNotInstalledError) as excinfo,
        ):
            synthpanel_toolkit(stub_client)

        msg = str(excinfo.value)
        assert "pip install" in msg
        assert "composio" in msg

    def test_client_without_experimental_toolkit_raises(self):
        """Older Composio releases lack `experimental.Toolkit`."""
        from synth_panel.integrations.composio import synthpanel_toolkit

        class _Old:
            experimental = None

        with pytest.raises(RuntimeError, match="experimental.Toolkit"):
            synthpanel_toolkit(_Old())

    def test_module_imports_without_composio(self, monkeypatch):
        """Importing the adapter itself must not require composio.

        Users of SynthPanel who never touch Composio should not pay an
        import-time price. We don't reload — we just assert the module
        is in sys.modules after a plain import below (via other tests)
        and re-verify the import is cheap here.
        """
        monkeypatch.delitem(sys.modules, "composio", raising=False)
        monkeypatch.delitem(sys.modules, "synth_panel.integrations.composio", raising=False)

        import synth_panel.integrations.composio as mod

        assert hasattr(mod, "synthpanel_toolkit")
        assert hasattr(mod, "ComposioNotInstalledError")
        assert mod.TOOLKIT_SLUG == "SYNTHPANEL"
