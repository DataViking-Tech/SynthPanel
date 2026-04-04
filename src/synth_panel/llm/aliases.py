"""Model alias resolution (SPEC.md §2 — Provider Resolution)."""

from __future__ import annotations

# Short alias → canonical model identifier.
MODEL_ALIASES: dict[str, str] = {
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
    "grok": "grok-3",
    "gemini": "gemini-2.5-flash",
    "gemini-pro": "gemini-2.5-pro",
}


def resolve_alias(model: str) -> str:
    """Return the canonical model name, resolving short aliases."""
    return MODEL_ALIASES.get(model, model)
