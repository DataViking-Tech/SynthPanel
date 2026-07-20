"""Third-party framework integrations for Althing.

Each submodule here wires the Althing SDK into an external tool-calling
framework so agents written in that framework can invoke Althing actions
natively. The integrations are lazily importable — installing Althing
alone does not pull in any framework — so use the documented
``althing[<framework>]`` extra (see ``pyproject.toml``) when you want
them.

Available submodules:

* :mod:`althing.integrations.composio` — Composio experimental
  :class:`Toolkit` exposing five Althing actions (``quick_poll``,
  ``run_panel``, ``list_personas``, ``list_instruments``,
  ``get_panel_result``) to any Composio-compatible framework
  (LangChain, CrewAI, Semantic Kernel, AutoGen).
"""

from __future__ import annotations

__all__: list[str] = []
