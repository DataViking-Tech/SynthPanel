"""On-disk credential store for SynthPanel (sp-lve).

Users who don't want to export an ``ANTHROPIC_API_KEY`` (etc.) in every
shell can run ``synthpanel login`` to persist a key to
``~/.config/synthpanel/credentials.json`` (mode ``0600``). Provider code
then resolves credentials in two steps: environment variable first,
config file second.

This avoids the broken zero-config experience where a developer with
Claude Code installed types ``synthpanel prompt`` and hits "Missing API
key" — Claude Code's OAuth tokens live in the macOS keychain under a
different auth scheme and aren't reusable as Anthropic API keys, so we
offer a dedicated ``synthpanel login`` instead of auto-bridging.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

# Recognised provider env var names. ``synthpanel login`` validates
# against this list so a typo doesn't silently persist a key nothing
# will ever read.
KNOWN_CREDENTIAL_ENV_VARS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "XAI_API_KEY",
    "OPENROUTER_API_KEY",
)

# Friendly provider label for each env var (used by ``whoami`` output).
PROVIDER_LABELS: dict[str, str] = {
    "ANTHROPIC_API_KEY": "Anthropic",
    "OPENAI_API_KEY": "OpenAI",
    "GEMINI_API_KEY": "Google (Gemini)",
    "GOOGLE_API_KEY": "Google (Gemini)",
    "XAI_API_KEY": "xAI",
    "OPENROUTER_API_KEY": "OpenRouter",
}


def credentials_path() -> Path:
    """Return the path to the credential store.

    Honors ``SYNTHPANEL_CREDENTIALS_PATH`` for tests and XDG-style
    overrides, then falls back to ``~/.config/synthpanel/credentials.json``.
    """
    override = os.environ.get("SYNTHPANEL_CREDENTIALS_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "synthpanel" / "credentials.json"


def load_credentials() -> dict[str, str]:
    """Return stored credentials as ``{env_var: value}``.

    Returns an empty dict if the file is missing or unreadable. Only
    string values are kept so malformed entries can't crash the caller.
    """
    path = credentials_path()
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else {}
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(k, str) and isinstance(v, str) and v}


def save_credential(env_var: str, value: str) -> Path:
    """Persist ``value`` under ``env_var`` in the credential store.

    Creates the parent directory with mode ``0700`` and writes the file
    atomically (write temp + rename) with mode ``0600`` so other local
    users cannot read the API key.

    Returns the path that was written.
    """
    env_var = env_var.strip().upper()
    value = value.strip()
    if not env_var:
        raise ValueError("env_var must be non-empty")
    if not value:
        raise ValueError("value must be non-empty")

    existing = load_credentials()
    existing[env_var] = value

    path = credentials_path()
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(parent, 0o700)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with contextlib.suppress(OSError):
        os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return path


def delete_credential(env_var: str) -> bool:
    """Remove ``env_var`` from the credential store.

    Returns True if an entry was removed, False if it was absent.
    """
    env_var = env_var.strip().upper()
    existing = load_credentials()
    if env_var not in existing:
        return False
    del existing[env_var]
    path = credentials_path()
    if existing:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with contextlib.suppress(OSError):
            os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    else:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
    return True


def get_credential(env_var: str) -> str | None:
    """Return a credential value for ``env_var``.

    Precedence: environment variable first, then the on-disk store.
    Returns ``None`` when nothing is set so callers can decide whether
    that's an error or a pass-through (e.g. provider picking between
    GEMINI_API_KEY and GOOGLE_API_KEY).
    """
    value = os.environ.get(env_var, "").strip()
    if value:
        return value
    stored = load_credentials().get(env_var, "").strip()
    return stored or None


def has_credential(env_var: str) -> bool:
    """Return True if the env var is set or stored on disk."""
    return get_credential(env_var) is not None
