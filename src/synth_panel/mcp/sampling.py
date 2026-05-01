"""MCP sampling bridge.

MCP's *sampling* feature lets an MCP server ask the invoking client
(Claude Desktop, Claude Code, Cursor, Windsurf, ...) to run an LLM
completion on the server's behalf, using the client's own subscription
or credentials. This lets synthpanel give first-time users a
zero-configuration experience — no ``ANTHROPIC_API_KEY`` setup needed
to fire their first prompt or quick poll.

Design
======

Two routing decisions live here:

1. **Can we sample?** — the client must advertise ``sampling`` capability
   in its ``initialize`` handshake and a :class:`Context` must actually
   be threaded through from the tool call. Exposed via
   :func:`client_supports_sampling`.

2. **Should we sample?** — we honour an explicit ``use_sampling`` flag
   on the calling tool; otherwise we fall back to sampling only when no
   BYOK credentials are present in the environment. Exposed via
   :func:`decide_mode`.

Tool handlers call :func:`sample_text` once the decision is made.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Sampling mode guardrails — keep invocations light so we don't blast
# the host agent's context window. Heavy research panels still require
# BYOK credentials.
SAMPLING_MAX_PERSONAS = 3
SAMPLING_MAX_QUESTIONS = 5
SAMPLING_MAX_TOKENS_DEFAULT = 2048

# sp-k2ed4a: canonical MCP stop_reason value indicating the host's token
# ceiling cut the response off mid-stream. Hosts (Claude Desktop, Cursor,
# Windsurf...) commonly cap output more aggressively than the request,
# silently truncating the JSON a structured-output engine then fails to
# parse. We surface this as a warning so callers can distinguish "host
# clipped me" from generic schema-fail.
SAMPLING_STOP_REASON_TRUNCATED = "maxTokens"

__all__ = [
    "SAMPLING_CRED_ENV_VARS",
    "SAMPLING_FIRST_RUN_HINT",
    "SAMPLING_MAX_PERSONAS",
    "SAMPLING_MAX_QUESTIONS",
    "SAMPLING_MAX_TOKENS_DEFAULT",
    "SAMPLING_STOP_REASON_TRUNCATED",
    "SamplingDecision",
    "build_truncation_warning",
    "client_supports_sampling",
    "decide_mode",
    "has_byok_credentials",
    "sample_text",
]

# Environment variables we treat as "BYOK present". Matches the provider
# set in synth_panel.llm.providers — must stay in sync with the CLI's
# _DEFAULT_MODEL_PREFERENCE and sdk._DEFAULT_MODEL_PREFERENCE, otherwise
# a user who set (e.g.) OPENROUTER_API_KEY gets routed to sampling or a
# "missing credentials" error despite the CLI recognising the same key.
SAMPLING_CRED_ENV_VARS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "XAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
)

# Importing here (module scope) would create a cycle if credentials ever
# grew an MCP dependency; the function-local import in
# :func:`has_byok_credentials` keeps the boundary one-directional.

# First-run hint surfaced on successful sampling runs so users know they
# can graduate to BYOK for larger panels / ensembles.
SAMPLING_FIRST_RUN_HINT = (
    "Running in host-agent sampling mode — synthpanel is borrowing your "
    "MCP client's LLM access instead of calling a provider directly. For "
    "cross-provider ensembles and larger panels, set ANTHROPIC_API_KEY "
    "(or another provider key) in your environment and re-run. "
    "See https://synthpanel.dev/mcp#credentials."
)


@dataclass
class SamplingDecision:
    """Outcome of :func:`decide_mode`.

    Attributes:
        mode: One of ``"sampling"``, ``"byok"``, or ``"error"``.
        hint: When ``mode == "sampling"``, a one-line user hint to
            surface in the response. ``None`` otherwise.
        error: When ``mode == "error"``, a friendly error message
            explaining how to unblock. ``None`` otherwise.
    """

    mode: str
    hint: str | None = None
    error: str | None = None


def has_byok_credentials(env: dict[str, str] | None = None) -> bool:
    """Return True when any provider credential is available to BYOK.

    Checks both the process environment and the on-disk credential store
    written by ``synthpanel login`` (sp-1ez / v0.9.4), so an MCP-launched
    subprocess — which often runs without the invoking shell's env —
    still recognises keys the CLI can see. When ``env`` is provided
    (test harness), only that mapping is consulted; the disk store is
    intentionally skipped so tests stay hermetic.
    """
    if env is not None:
        return any(env.get(var, "").strip() for var in SAMPLING_CRED_ENV_VARS)
    from synth_panel.credentials import has_credential

    return any(has_credential(var) for var in SAMPLING_CRED_ENV_VARS)


def client_supports_sampling(ctx: Any) -> bool:
    """Return True when the invoking MCP client advertised sampling.

    ``ctx`` is a FastMCP :class:`mcp.server.fastmcp.Context`. We go
    through the underlying :class:`ServerSession` because
    ``check_client_capability`` is the supported way to interrogate the
    client's declared capabilities. Any failure (no context, no session,
    capability object not importable) is treated as "no sampling" — we
    never want capability detection to raise into the tool handler.
    """
    if ctx is None:
        return False

    # ``ctx.session`` is a property that raises ValueError when the
    # Context is constructed outside a live request (FastMCP test
    # harness, synthetic invocations). We swallow any such failure —
    # capability detection must never raise into a tool handler.
    try:
        session = ctx.session
    except Exception:
        return False
    if session is None:
        return False

    try:
        from mcp.types import ClientCapabilities, SamplingCapability

        check = session.check_client_capability(ClientCapabilities(sampling=SamplingCapability()))
    except Exception:
        return False
    return bool(check)


def decide_mode(
    ctx: Any,
    *,
    use_sampling: bool | None = None,
    env: dict[str, str] | None = None,
) -> SamplingDecision:
    """Choose between sampling and BYOK for this tool call.

    Args:
        ctx: The FastMCP :class:`Context` passed into the tool.
        use_sampling: Explicit override from the tool caller. ``True``
            forces sampling (error if unsupported), ``False`` forces
            BYOK, ``None`` picks automatically.
        env: Optional env dict for testing. When ``None`` both the
            process environment and the on-disk credential store are
            consulted via :func:`has_byok_credentials`.

    Rules (in order):
        * ``use_sampling=True`` + client supports sampling → sampling.
        * ``use_sampling=True`` + client does NOT support sampling →
          error (explains the client must advertise ``sampling``).
        * ``use_sampling=False`` → BYOK unconditionally.
        * Auto + no creds + client supports sampling → sampling.
        * Auto + no creds + no sampling → error (set an API key OR use
          a sampling-capable client).
        * Auto + creds present → BYOK (preserves existing behaviour
          and keeps ensemble / multi-provider features available).
    """
    supports = client_supports_sampling(ctx)
    has_creds = has_byok_credentials(env)

    if use_sampling is True:
        if supports:
            return SamplingDecision(mode="sampling", hint=SAMPLING_FIRST_RUN_HINT)
        return SamplingDecision(
            mode="error",
            error=(
                "use_sampling=True was requested, but the invoking MCP "
                "client did not advertise the 'sampling' capability. "
                "Either run synthpanel from a sampling-capable client "
                "(Claude Desktop, Claude Code, Cursor, Windsurf) or set "
                "a provider API key (e.g. ANTHROPIC_API_KEY) to use BYOK."
            ),
        )

    if use_sampling is False:
        return SamplingDecision(mode="byok")

    # Auto mode.
    if has_creds:
        return SamplingDecision(mode="byok")
    if supports:
        return SamplingDecision(mode="sampling", hint=SAMPLING_FIRST_RUN_HINT)
    return SamplingDecision(
        mode="error",
        error=(
            "No provider credentials found (ANTHROPIC_API_KEY / "
            "OPENAI_API_KEY / XAI_API_KEY / GOOGLE_API_KEY / "
            "GEMINI_API_KEY / OPENROUTER_API_KEY) and the invoking "
            "MCP client did not advertise 'sampling' capability. "
            "Set a provider key in your environment, or run synthpanel "
            "from a sampling-capable client such as Claude Desktop. "
            "See https://synthpanel.dev/mcp#credentials."
        ),
    )


def build_truncation_warning(*, max_tokens: int, model: str | None) -> str:
    """Build a user-facing message describing host-side token-cap truncation.

    Surfaced in panel/quick-poll ``warnings`` lists so MCP/CLI consumers
    can distinguish a host max_tokens cap from a generic schema-fail when
    a structured-output post-parse fallback fires. ``model`` is whatever
    the host reported running (e.g. ``"claude-opus-4-6"``); ``None`` when
    the host did not name a model.
    """
    model_part = f" (host model: {model})" if model else ""
    return (
        f"MCP host truncated sampling output at the {max_tokens}-token "
        f"ceiling{model_part} — the response may be incomplete and any "
        "structured-output parse failure on this turn is likely caused by "
        "truncation rather than the model ignoring the schema. Hosts may "
        "cap output more aggressively than requested; for longer schemas, "
        "set a provider key (e.g. ANTHROPIC_API_KEY) to use BYOK with a "
        "higher token budget."
    )


async def sample_text(
    ctx: Any,
    *,
    prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = SAMPLING_MAX_TOKENS_DEFAULT,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Run one sampling round via ``ctx.session.create_message``.

    Returns a dict with keys ``text``, ``model``, ``stop_reason``,
    ``role``, ``truncated``, ``requested_max_tokens``, and ``warning``.
    The model string is whatever the host agent chose to run (e.g.
    ``"claude-opus-4-6"`` when invoked from Claude Desktop). We
    normalise content blocks to a single joined string so downstream
    consumers don't have to special-case multi-block responses.

    Truncation detection (sp-k2ed4a): when the host reports
    ``stopReason == "maxTokens"`` the response was cut short by the
    host's output cap. This commonly happens because hosts (Claude
    Desktop, Cursor, Windsurf...) impose their own ceiling that ignores
    or undershoots ``max_tokens``. We log a warning, set ``truncated``,
    and surface a ready-to-display ``warning`` string so callers can
    propagate the signal into their ``warnings`` payload instead of
    chalking the partial JSON up to a generic schema-fail.
    """
    from mcp.types import SamplingMessage, TextContent

    messages = [
        SamplingMessage(
            role="user",
            content=TextContent(type="text", text=prompt),
        )
    ]
    result = await ctx.session.create_message(
        messages=messages,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        temperature=temperature,
    )

    text = _extract_text(result.content)
    stop_reason = getattr(result, "stopReason", None)
    truncated = stop_reason == SAMPLING_STOP_REASON_TRUNCATED
    warning: str | None = None
    if truncated:
        warning = build_truncation_warning(max_tokens=max_tokens, model=result.model)
        logger.warning(
            "MCP sampling truncated: stopReason=%s requested_max_tokens=%d model=%s output_chars=%d",
            stop_reason,
            max_tokens,
            result.model,
            len(text),
        )
    return {
        "text": text,
        "model": result.model,
        "stop_reason": stop_reason,
        "role": result.role,
        "truncated": truncated,
        "requested_max_tokens": max_tokens,
        "warning": warning,
    }


def _extract_text(content: Any) -> str:
    """Flatten sampling result content into a single text string."""
    # ``content`` may be a single TextContent/ImageContent or a list of
    # them. We only surface text — image/audio content from the host
    # agent isn't useful to the panel simulation, so it's silently
    # dropped with a newline join between blocks.
    if content is None:
        return ""
    blocks = content if isinstance(content, list) else [content]
    parts: list[str] = []
    for block in blocks:
        btype = getattr(block, "type", None)
        if btype == "text":
            parts.append(getattr(block, "text", ""))
    return "\n".join(p for p in parts if p)
