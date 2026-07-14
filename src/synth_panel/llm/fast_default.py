"""Fast-model swap for auto-resolved defaults on large panels.

Shared by all three entry points (CLI ``panel run``, SDK
``run_panel``/``quick_poll``, MCP ``run_panel``/``run_quick_poll``) so
the ≥10-persona fast-default policy lives in exactly one place
(sy-2ag / GH#462 / synthbench#261).

Background: with only ``OPENROUTER_API_KEY`` set, the auto-resolved
default model is ``openrouter/auto``, which OpenRouter may route to a
slow reasoning model (~40 s per response; a 20-persona panel hung
>15 min). Pinning a fast model cut the same run to 25-40 s. Most other
provider defaults are already fast (haiku, gpt-4o-mini,
gemini-2.5-flash), so only ``openrouter/auto`` needs the swap today.

The swap applies **only** when the caller did not explicitly choose a
model — explicit choices (including an explicit ``openrouter/auto``)
are always honored verbatim. Call sites must guard on "model was not
supplied" before consulting this module.
"""

from __future__ import annotations

# Persona-count threshold at or above which the auto-resolved default
# model is swapped for a known-fast equivalent.
LARGE_PANEL_PERSONA_THRESHOLD = 10

# Slow auto-resolved default → fast equivalent for large panels. Keyed
# on the resolved default alias; aliases that are already fast (or whose
# routing the user controls) are absent.
FAST_MODEL_SWAP: dict[str, str] = {
    "openrouter/auto": "openrouter/anthropic/claude-haiku-4.5",
}


def fast_default_for_panel(alias: str, persona_count: int) -> tuple[str, str | None]:
    """Return ``(model, swapped_from)`` for an auto-resolved default.

    When *persona_count* is at or above
    :data:`LARGE_PANEL_PERSONA_THRESHOLD` and *alias* has a known-fast
    equivalent in :data:`FAST_MODEL_SWAP`, returns the fast model and
    the original alias it replaced. Otherwise returns ``(alias, None)``.

    Only call this for defaults the user did **not** explicitly choose.
    """
    if persona_count >= LARGE_PANEL_PERSONA_THRESHOLD:
        swapped = FAST_MODEL_SWAP.get(alias)
        if swapped is not None and swapped != alias:
            return swapped, alias
    return alias, None


def format_fast_default_note(
    model: str,
    swapped_from: str,
    persona_count: int,
    override_hint: str = "--model",
) -> str:
    """One-line user-facing note explaining the fast-default swap."""
    return (
        f"auto-selected {model} for a {persona_count}-persona run "
        f"(default {swapped_from} can be very slow at this size); "
        f"pass {override_hint} to override."
    )


__all__ = [
    "FAST_MODEL_SWAP",
    "LARGE_PANEL_PERSONA_THRESHOLD",
    "fast_default_for_panel",
    "format_fast_default_note",
]
