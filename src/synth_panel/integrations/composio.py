"""Composio connector for SynthPanel.

Exposes the SynthPanel SDK as a Composio experimental :class:`Toolkit` so
agents built on LangChain, CrewAI, Semantic Kernel, or AutoGen can invoke
SynthPanel actions through Composio's native tool-calling path (no MCP
bridge, no subprocess).

Usage::

    from composio import Composio
    from composio_langchain import LangchainProvider
    from synth_panel.integrations.composio import synthpanel_toolkit

    composio = Composio(provider=LangchainProvider())
    toolkit = synthpanel_toolkit(composio)
    session = composio.create(
        user_id="researcher_1",
        experimental={"custom_toolkits": [toolkit]},
    )
    tools = session.tools()  # includes SynthPanel actions
    # ...hand `tools` to LangChain / CrewAI / etc.

Five actions are registered under the ``SYNTHPANEL`` toolkit slug:

* ``quick_poll`` — one question, a panel, synthesized findings.
* ``run_panel`` — full panel run against an installed instrument pack.
* ``list_personas`` — metadata for every installed persona pack.
* ``list_instruments`` — metadata for every installed instrument pack.
* ``get_panel_result`` — load a previously saved panel result by id.

The toolkit is a factory, not a module-level singleton, because Composio
tags each toolkit to a specific ``Composio`` client instance. Call
:func:`synthpanel_toolkit` once per client. The import itself is cheap
and has no hard dependency on Composio — the actual Composio objects are
resolved lazily inside the factory so an import of this module succeeds
on any install, and users get a targeted error message only if they try
to *build* the toolkit without Composio installed.

Provider coverage: Composio's per-framework providers
(``composio_langchain``, ``composio_crewai``, ``composio_openai``,
``composio_autogen``, ``composio_semantickernel``, ``composio_google``)
all consume custom toolkits uniformly — the same toolkit works across
every framework.
"""

from typing import Any

from synth_panel import sdk

# NOTE: we deliberately do NOT use ``from __future__ import annotations`` in
# this module. Composio (and Pydantic) read the live class objects off the
# tool function's annotations to build the JSON Schema that the LLM sees.
# Stringified annotations (PEP 563) would force a runtime
# ``get_type_hints`` resolution, and because our input classes live inside
# :func:`synthpanel_toolkit` (local scope), that resolution cannot find them.

# Pydantic is an optional dependency (ships with Composio). We bind
# ``BaseModel``/``Field`` at module load if available so the class bodies
# inside :func:`synthpanel_toolkit` — and mypy — resolve to real types.
# If the user reaches the factory without pydantic installed,
# ``_require_pydantic`` raises first, so the dummy fallbacks below are
# never actually exercised to build a class.
try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - pydantic ships with composio
    BaseModel = object  # type: ignore[assignment,misc]

    def Field(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        return None


# The toolkit slug is what agents (and Composio's catalog) see. Actions are
# namespaced automatically as LOCAL_SYNTHPANEL_<action> by Composio.
TOOLKIT_SLUG = "SYNTHPANEL"
TOOLKIT_NAME = "SynthPanel"
TOOLKIT_DESCRIPTION = (
    "Run synthetic focus groups against AI persona panels. Ask a single "
    "question (quick_poll), execute a multi-round instrument (run_panel), "
    "list installed persona and instrument packs, or load a saved panel "
    "result. Runs locally via the SynthPanel SDK; requires an LLM API key "
    "(ANTHROPIC_API_KEY by default) in the host environment."
)


class ComposioNotInstalledError(ImportError):
    """Raised when Composio is not installed but required.

    The integration keeps Composio as an optional dependency — installing
    SynthPanel alone is cheap. This error carries the exact ``pip install``
    command in its message so downstream agents can surface it to users.
    """


def _require_composio() -> Any:
    """Import Composio or raise a helpful error.

    Wrapping the import keeps the module importable everywhere (docs
    builds, type checkers, test collectors) while still failing loudly
    at the exact moment a caller tries to wire the toolkit.
    """
    try:
        import composio
    except ImportError as exc:  # pragma: no cover - exercised via patched import
        raise ComposioNotInstalledError(
            "Composio is not installed. Install it with "
            "`pip install synthpanel[composio]` (or `pip install composio`) "
            "to register SynthPanel as a Composio toolkit."
        ) from exc
    return composio


def _require_pydantic() -> Any:
    try:
        import pydantic
    except ImportError as exc:  # pragma: no cover - composio ships with pydantic
        raise ComposioNotInstalledError(
            "Pydantic is required by the Composio toolkit but is not "
            "installed. Install it with `pip install pydantic` (normally "
            "pulled in by Composio)."
        ) from exc
    return pydantic


def synthpanel_toolkit(composio_client: Any) -> Any:
    """Build a Composio :class:`Toolkit` exposing five SynthPanel actions.

    Args:
        composio_client: A ``composio.Composio`` instance. The toolkit is
            constructed via ``composio_client.experimental.Toolkit`` and
            each action is registered with the standard ``@toolkit.tool()``
            decorator, so the returned object is a native Composio
            toolkit — not a wrapper — and participates fully in
            ``session.tools()``, catalog introspection, and per-provider
            tool transformations.

    Returns:
        A Composio toolkit. Pass it via
        ``composio_client.create(user_id=..., experimental={"custom_toolkits": [toolkit]})``.

    Raises:
        ComposioNotInstalledError: if the ``composio`` package cannot be
            imported.

    Why this is a factory: Composio binds each toolkit to the client
    used to build it, and sharing a toolkit across clients is not
    supported. Callers routinely have multiple clients (dev vs. prod,
    different providers) so returning a fresh toolkit per call is the
    safe default.
    """
    _require_composio()
    _require_pydantic()

    experimental = getattr(composio_client, "experimental", None)
    if experimental is None or not hasattr(experimental, "Toolkit"):
        raise RuntimeError(
            "The Composio client does not expose `experimental.Toolkit`. "
            "Update to a Composio release that includes the experimental "
            "custom-tools API, or open an issue against SynthPanel if "
            "this was unexpected."
        )

    toolkit = experimental.Toolkit(
        slug=TOOLKIT_SLUG,
        name=TOOLKIT_NAME,
        description=TOOLKIT_DESCRIPTION,
    )

    # --- Input schemas --------------------------------------------------
    # Keep field descriptions tight — agents read these verbatim when
    # deciding whether to call the tool and with what arguments.

    class QuickPollInput(BaseModel):
        question: str = Field(description="The single question to put to every persona in the panel.")
        pack_id: str | None = Field(
            default=None,
            description=(
                "Name of an installed persona pack (see list_personas). "
                "Provide `pack_id` and/or inline `personas`; at least one "
                "is required."
            ),
        )
        personas: list[dict[str, Any]] | None = Field(
            default=None,
            description=(
                "Inline persona list. Each entry needs at least a `name` key. Merged with any `pack_id` personas."
            ),
        )
        model: str | None = Field(
            default=None,
            description=(
                "Model alias (sonnet, haiku, gemini, grok-3) or canonical id. "
                "Defaults to the first provider whose API key is in the environment."
            ),
        )
        synthesis: bool = Field(
            default=True,
            description="Run the synthesis step after collecting responses.",
        )

    class RunPanelInput(BaseModel):
        instrument_pack: str | None = Field(
            default=None,
            description=(
                "Name of an installed instrument pack (see list_instruments). "
                "Takes precedence over inline `instrument` and `questions`."
            ),
        )
        instrument: dict[str, Any] | None = Field(
            default=None,
            description=(
                "Inline v1/v2/v3 instrument body. For v3 with route_when "
                "clauses, the panel runs as a branching multi-round session."
            ),
        )
        questions: list[str] | None = Field(
            default=None,
            description=(
                "Flat list of question strings for the simplest case "
                "(v1 shape). Ignored when `instrument` or `instrument_pack` "
                "is set."
            ),
        )
        pack_id: str | None = Field(
            default=None,
            description="Name of an installed persona pack.",
        )
        personas: list[dict[str, Any]] | None = Field(
            default=None,
            description="Inline persona list. Merged with pack_id personas.",
        )
        model: str | None = Field(
            default=None,
            description="Model alias for panelist responses.",
        )
        synthesis: bool = Field(
            default=True,
            description="Run the synthesis step.",
        )

    class EmptyInput(BaseModel):
        """No parameters."""

    class GetPanelResultInput(BaseModel):
        result_id: str = Field(
            description=(
                "Panel result id returned by a previous quick_poll or "
                "run_panel call (see list_panel_results for saved ids)."
            )
        )

    # --- Tool implementations -------------------------------------------
    # Each tool is a thin wrapper around an SDK function. We serialise
    # results to dicts because Composio passes tool outputs back to the
    # LLM as JSON; dataclass instances don't round-trip cleanly.

    @toolkit.tool()
    def quick_poll(request: QuickPollInput, ctx: Any) -> dict[str, Any]:
        """Ask one question of a persona panel and return synthesized findings.

        Use this when you want fast qualitative signal from multiple
        AI personas on a single question. Each persona answers
        independently in parallel; the synthesis step aggregates themes,
        agreements, and disagreements.

        Returns a dict with `result_id`, `question`, `responses` (per
        panelist), `synthesis` (themes + recommendation), `model`,
        `total_usage`, `total_cost`, and `metadata`.
        """
        del ctx  # unused but required by Composio's tool signature
        result = sdk.quick_poll(
            question=request.question,
            personas=request.personas,
            pack_id=request.pack_id,
            model=request.model,
            synthesis=request.synthesis,
        )
        return result.to_dict()

    @toolkit.tool()
    def run_panel(request: RunPanelInput, ctx: Any) -> dict[str, Any]:
        """Run a full synthetic focus group panel.

        Provide exactly one question source — an `instrument_pack`
        (installed pack name), an inline `instrument` dict, or a flat
        `questions` list. v3 instruments with `route_when` clauses
        execute as a branching multi-round panel.

        Returns a dict with `result_id`, `rounds`, routing `path`,
        final `synthesis`, `total_usage`, `total_cost`, and `metadata`.
        """
        del ctx
        result = sdk.run_panel(
            personas=request.personas,
            instrument=request.instrument,
            questions=request.questions,
            instrument_pack=request.instrument_pack,
            pack_id=request.pack_id,
            model=request.model,
            synthesis=request.synthesis,
        )
        return result.to_dict()

    @toolkit.tool()
    def list_personas(request: EmptyInput, ctx: Any) -> dict[str, Any]:
        """List installed persona packs (bundled + user-saved).

        Returns a dict with `packs`: a list of `{id, name,
        persona_count, builtin}` entries. Use a pack's `id` as the
        `pack_id` argument to `quick_poll` or `run_panel`.
        """
        del request, ctx
        return {"packs": sdk.list_personas()}

    @toolkit.tool()
    def list_instruments(request: EmptyInput, ctx: Any) -> dict[str, Any]:
        """List installed instrument packs (bundled + user-saved).

        Returns a dict with `packs`: manifest entries (`id`, `name`,
        `version`, `description`, `author`, `source`). Use a pack's
        `id` as the `instrument_pack` argument to `run_panel`.
        """
        del request, ctx
        return {"packs": sdk.list_instruments()}

    @toolkit.tool()
    def get_panel_result(request: GetPanelResultInput, ctx: Any) -> dict[str, Any]:
        """Load a saved panel result by id.

        Use the `result_id` returned by a previous `quick_poll` or
        `run_panel` call. Returns the full :class:`PanelResult` as a
        dict, including all rounds, synthesis, usage, and cost.
        """
        del ctx
        return sdk.get_panel_result(request.result_id).to_dict()

    return toolkit


__all__ = [
    "TOOLKIT_DESCRIPTION",
    "TOOLKIT_NAME",
    "TOOLKIT_SLUG",
    "ComposioNotInstalledError",
    "synthpanel_toolkit",
]
