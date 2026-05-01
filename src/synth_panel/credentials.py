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
import hashlib
import json
import os
from pathlib import Path


class CredentialIntegrityError(Exception):
    """Raised when credentials.json sha256 sidecar does not match the file content."""


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

# Required prefix(es) for each provider's API keys. ``synthpanel login``
# uses this to reject visibly-broken inputs at write time rather than
# letting them persist and surface as opaque 401s on later calls. An
# empty tuple means "no convention worth enforcing" — Gemini/Google keys
# don't have a stable distinctive prefix.
PROVIDER_KEY_PREFIXES: dict[str, tuple[str, ...]] = {
    "ANTHROPIC_API_KEY": ("sk-ant-",),
    "OPENAI_API_KEY": ("sk-",),
    "OPENROUTER_API_KEY": ("sk-or-",),
    "XAI_API_KEY": ("xai-",),
    "GEMINI_API_KEY": (),
    "GOOGLE_API_KEY": (),
}

# CLI ``--provider`` name for each known env var. Inverse of
# ``cli/commands.py:_PROVIDER_ENV_VAR``; kept here so non-CLI call sites
# (provider error messages, etc.) can render the suggested
# ``synthpanel login --provider <name>`` hint without importing CLI code.
PROVIDER_CLI_NAMES: dict[str, str] = {
    "ANTHROPIC_API_KEY": "anthropic",
    "OPENAI_API_KEY": "openai",
    "GEMINI_API_KEY": "gemini",
    "GOOGLE_API_KEY": "google",
    "XAI_API_KEY": "xai",
    "OPENROUTER_API_KEY": "openrouter",
}


def missing_api_key_message(env_var: str, *, alt_env_vars: tuple[str, ...] = ()) -> str:
    """Return a 'Missing API key' error message for ``env_var``.

    Tells the caller which env var was checked, suggests both the
    persistent (``synthpanel login --provider <name>``) and one-shot
    (``export <ENV_VAR>=...``) options, and — for Anthropic specifically
    — calls out the Claude Code OAuth footgun documented in this
    module's top-level docstring (sp-stkj2w).

    ``alt_env_vars`` lists alternate env vars the same provider also
    accepts (e.g. Gemini reads both ``GEMINI_API_KEY`` and
    ``GOOGLE_API_KEY``); they are joined with the primary in the
    "set X or Y" portion of the message.
    """
    label = PROVIDER_LABELS.get(env_var, env_var)
    cli_name = PROVIDER_CLI_NAMES.get(env_var)
    env_vars = (env_var,) + tuple(alt_env_vars)
    env_var_list = " or ".join(env_vars)

    if cli_name:
        msg = (
            f"Missing API key for {label}: set {env_var_list}, "
            f"or run `synthpanel login --provider {cli_name}` to persist a key on disk."
        )
    else:
        msg = (
            f"Missing API key: set {env_var_list}, "
            f"or run `synthpanel login` to persist a key on disk."
        )

    if env_var == "ANTHROPIC_API_KEY":
        msg += (
            " Note: Claude Code's OAuth tokens are NOT reusable as Anthropic "
            "API keys (different auth scheme); get a key from "
            "https://console.anthropic.com/."
        )

    return msg


def detect_provider_from_key(value: str) -> str | None:
    """Return the env var whose distinctive prefix matches ``value``.

    Used by ``synthpanel login`` to suggest the correct ``--provider``
    when a user passes another provider's key by mistake (e.g. an
    Anthropic key given to ``--provider openai``).

    The bare ``sk-`` prefix is intentionally checked last: it identifies
    OpenAI but is also a substring of ``sk-ant-`` and ``sk-or-``, so we
    only fall back to ``OPENAI_API_KEY`` after the more-specific prefixes
    have been ruled out. Returns ``None`` for keys without a recognised
    prefix (e.g. Gemini keys).
    """
    distinctive: tuple[tuple[str, str], ...] = (
        ("sk-ant-", "ANTHROPIC_API_KEY"),
        ("sk-or-", "OPENROUTER_API_KEY"),
        ("sk-proj-", "OPENAI_API_KEY"),
        ("xai-", "XAI_API_KEY"),
    )
    for prefix, env_var in distinctive:
        if value.startswith(prefix):
            return env_var
    if value.startswith("sk-"):
        return "OPENAI_API_KEY"
    return None


def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _compute_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


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

    Raises ``CredentialIntegrityError`` if a sha256 sidecar exists and does
    not match the file content. When no sidecar exists the sidecar is
    generated (migration) and the credentials are returned normally.
    """
    path = credentials_path()
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}

    sidecar = _sidecar_path(path)
    actual_hash = _compute_hash(raw)
    if sidecar.exists():
        try:
            stored_hash = sidecar.read_text(encoding="utf-8").strip()
        except OSError:
            stored_hash = ""
        if stored_hash != actual_hash:
            raise CredentialIntegrityError(
                f"Credential file {path} failed integrity check — "
                "re-run `synthpanel login` to restore your credentials."
            )
    else:
        # Migration: generate sidecar on first successful read
        try:
            sidecar_tmp = path.with_name(path.name + ".sha256.tmp")
            sidecar_tmp.write_text(actual_hash + "\n", encoding="utf-8")
            with contextlib.suppress(OSError):
                os.chmod(sidecar_tmp, 0o600)
            os.replace(sidecar_tmp, sidecar)
        except OSError:
            pass

    try:
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
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

    try:
        existing = load_credentials()
    except CredentialIntegrityError:
        existing = {}  # login supersedes tampered credentials
    existing[env_var] = value

    content = json.dumps(existing, indent=2, sort_keys=True) + "\n"
    new_hash = _compute_hash(content)

    path = credentials_path()
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(parent, 0o700)

    sidecar = _sidecar_path(path)
    sidecar_tmp = path.with_name(path.name + ".sha256.tmp")
    sidecar_tmp.write_text(new_hash + "\n", encoding="utf-8")
    with contextlib.suppress(OSError):
        os.chmod(sidecar_tmp, 0o600)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    with contextlib.suppress(OSError):
        os.chmod(tmp, 0o600)

    os.replace(sidecar_tmp, sidecar)
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
    sidecar = _sidecar_path(path)
    if existing:
        content = json.dumps(existing, indent=2, sort_keys=True) + "\n"
        new_hash = _compute_hash(content)
        sidecar_tmp = path.with_name(path.name + ".sha256.tmp")
        sidecar_tmp.write_text(new_hash + "\n", encoding="utf-8")
        with contextlib.suppress(OSError):
            os.chmod(sidecar_tmp, 0o600)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        with contextlib.suppress(OSError):
            os.chmod(tmp, 0o600)
        os.replace(sidecar_tmp, sidecar)
        os.replace(tmp, path)
    else:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        with contextlib.suppress(FileNotFoundError):
            sidecar.unlink()
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
