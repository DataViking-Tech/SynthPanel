"""Third-party framework integrations for SynthPanel.

Each submodule here wires the SynthPanel SDK into an external tool-calling
framework so agents written in that framework can invoke SynthPanel actions
natively. The integrations are lazily importable — installing SynthPanel
alone does not pull in any framework — so use the documented
``synthpanel[<framework>]`` extra (see ``pyproject.toml``) when you want
them.

Available submodules:

* :mod:`synth_panel.integrations.composio` — Composio experimental
  :class:`Toolkit` exposing five SynthPanel actions (``quick_poll``,
  ``run_panel``, ``list_personas``, ``list_instruments``,
  ``get_panel_result``) to any Composio-compatible framework
  (LangChain, CrewAI, Semantic Kernel, AutoGen).
"""

from __future__ import annotations

__all__: list[str] = []
