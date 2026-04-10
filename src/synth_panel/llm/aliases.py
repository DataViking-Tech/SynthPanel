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

# Local model prefixes: prefix → default base URL (without /v1 suffix).
_LOCAL_PREFIXES: dict[str, str] = {
    "ollama:": "http://localhost:11434",
    "local:": "http://localhost:1234",
}


def resolve_alias(model: str) -> str:
    """Return the canonical model name, resolving short aliases and local prefixes."""
    for prefix in _LOCAL_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix) :]
    return MODEL_ALIASES.get(model, model)


def get_base_url_override(model: str) -> str | None:
    """Return the local base URL if *model* uses a local prefix, else None.

    ``ollama:llama3`` → ``"http://localhost:11434"``
    ``local:phi3``    → ``"http://localhost:1234"``
    """
    for prefix, url in _LOCAL_PREFIXES.items():
        if model.startswith(prefix):
            return url
    return None
