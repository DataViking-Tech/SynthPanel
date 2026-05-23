"""Helpers for `synthpanel mcp install` (sy-skf).

Edits a host's MCP config JSON file in-place (or creates it) so that
synthpanel is registered as a stdio MCP server. The default target is
Claude Code's user config (~/.claude.json), but the helpers are
host-agnostic — they only touch the ``mcpServers`` mapping.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

DEFAULT_SERVER_NAME = "synth_panel"

# sy-xyn: actionable copy used by every hard-error surface that detects
# a missing `mcp` extra (install refusal + `mcp-serve` startup guard).
# Distinct from MCP_EXTRA_WARNING (#512) which is the SOFT warning
# embedded in `InstallResult.warnings`. The two stay separate because
# refusal copy + advisory copy need different framing: the refusal
# stops the user and tells them how to proceed; the warning is the
# nudge embedded in a result they're already consuming. Both carry
# "synthpanel[mcp]" verbatim so the install command surfaces in
# either context.
MISSING_MCP_EXTRA_MESSAGE = (
    "the 'mcp' optional dependency is not installed in this Python env. "
    "Install it with:  pip install 'synthpanel[mcp]'  "
    "(see docs/mcp.md for the full setup walkthrough)"
)


@dataclass
class InstallResult:
    """Outcome of an install/uninstall call."""

    target: Path
    name: str
    action: str  # "installed", "updated", "removed", "noop", "would-install", "would-update", "would-remove"
    entry: dict[str, Any] | None  # the server entry that was written (None for remove/noop)
    # sy-0k2 / gh-495: the full config payload that would (or did) result
    # from this call. Populated for every dry-run so `--dry-run` can print
    # exactly what would be written. Also populated on the actual-write
    # path so callers in JSON mode see a uniform shape.
    resulting_config: dict[str, Any] | None = None
    warnings: list[str] | None = None


MCP_EXTRA_WARNING = (
    "Optional MCP dependency is not installed; run `pip install 'synthpanel[mcp]'` before using `synthpanel mcp-serve`."
)


def mcp_extra_available() -> bool:
    """Return whether the optional MCP dependency is importable."""
    return find_spec("mcp") is not None


def resolve_target(scope: str, target: str | None) -> Path:
    """Resolve which config file to edit."""
    if target:
        return Path(target).expanduser()
    if scope == "project":
        return Path.cwd() / ".mcp.json"
    return Path.home() / ".claude.json"


def resolve_command(explicit: str | None) -> str:
    """Resolve the command string the host should invoke."""
    if explicit:
        return explicit
    found = shutil.which("synthpanel")
    return found or "synthpanel"


def parse_env_pairs(pairs: list[str] | None) -> dict[str, str]:
    """Parse ``KEY=VALUE`` items into a dict; raise ValueError on bad input."""
    out: dict[str, str] = {}
    if not pairs:
        return out
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"--env expects KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"--env has empty key in {item!r}")
        out[key] = value
    return out


def build_entry(command: str, env: dict[str, str]) -> dict[str, Any]:
    """Build the JSON object the host expects under ``mcpServers[<name>]``."""
    entry: dict[str, Any] = {
        "command": command,
        "args": ["mcp-serve"],
    }
    if env:
        entry["env"] = env
    return entry


def _load_config(path: Path) -> dict[str, Any]:
    """Load a JSON config file; return ``{}`` if missing or empty."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON must be an object, got {type(data).__name__}")
    return data


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` as pretty-printed JSON, atomically, with mode 0600 on user configs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, indent=2, sort_keys=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(serialized, encoding="utf-8")
    # Match restrictive perms when the file lives under $HOME — the entry
    # may carry an API key in env, mirroring credentials.json hygiene.
    try:
        if str(path).startswith(str(Path.home())):
            os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(path)


def install(
    *,
    target: Path,
    name: str,
    command: str,
    env: dict[str, str],
    force: bool,
    dry_run: bool,
) -> InstallResult:
    """Insert or update the named server entry."""
    config = _load_config(target)
    servers = config.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}

    existing = servers.get(name)
    new_entry = build_entry(command, env)
    warnings = None if mcp_extra_available() else [MCP_EXTRA_WARNING]

    if existing == new_entry:
        # No-op: report the current on-disk config as the resulting config
        # so dry-run consumers can still inspect the unchanged state.
        config["mcpServers"] = servers
        return InstallResult(
            target=target,
            name=name,
            action="noop",
            entry=new_entry,
            resulting_config=config,
            warnings=warnings,
        )

    is_update = name in servers
    if is_update and not force:
        raise FileExistsError(
            f"MCP server {name!r} already exists in {target}. Pass --force to overwrite, or choose a different --name."
        )

    servers[name] = new_entry
    config["mcpServers"] = servers
    if dry_run:
        action = "would-update" if is_update else "would-install"
        return InstallResult(
            target=target,
            name=name,
            action=action,
            entry=new_entry,
            resulting_config=config,
            warnings=warnings,
        )

    _atomic_write(target, config)
    action = "updated" if is_update else "installed"
    return InstallResult(
        target=target,
        name=name,
        action=action,
        entry=new_entry,
        resulting_config=config,
        warnings=warnings,
    )


def uninstall(*, target: Path, name: str, dry_run: bool) -> InstallResult:
    """Remove the named server entry. No-op if it isn't present."""
    if not target.exists():
        return InstallResult(
            target=target,
            name=name,
            action="noop",
            entry=None,
            resulting_config={},
        )

    config = _load_config(target)
    servers = config.get("mcpServers")
    if not isinstance(servers, dict) or name not in servers:
        # Surface the unchanged config so dry-run consumers see the
        # current state even on no-op.
        if isinstance(servers, dict):
            config["mcpServers"] = servers
        return InstallResult(
            target=target,
            name=name,
            action="noop",
            entry=None,
            resulting_config=config,
        )

    if dry_run:
        # Materialize the post-removal config without writing.
        preview_servers = {k: v for k, v in servers.items() if k != name}
        preview = dict(config)
        preview["mcpServers"] = preview_servers
        return InstallResult(
            target=target,
            name=name,
            action="would-remove",
            entry=None,
            resulting_config=preview,
        )

    del servers[name]
    config["mcpServers"] = servers
    _atomic_write(target, config)
    return InstallResult(
        target=target,
        name=name,
        action="removed",
        entry=None,
        resulting_config=config,
    )


def format_text(result: InstallResult) -> str:
    """Render an InstallResult as a single human-readable line."""
    if result.action in ("noop",):
        return f"No changes needed for {result.name!r} in {result.target}."
    if result.action in ("removed", "would-remove"):
        verb = "Would remove" if result.action.startswith("would-") else "Removed"
        return f"{verb} MCP server {result.name!r} from {result.target}."
    verb = {
        "installed": "Installed",
        "updated": "Updated",
        "would-install": "Would install",
        "would-update": "Would update",
    }[result.action]
    return f"{verb} MCP server {result.name!r} in {result.target}."


def format_json(result: InstallResult) -> str:
    """Render an InstallResult as a one-line JSON payload."""
    payload: dict[str, Any] = {
        "target": str(result.target),
        "name": result.name,
        "action": result.action,
    }
    if result.entry is not None:
        payload["entry"] = result.entry
    if result.warnings:
        payload["warnings"] = result.warnings
    # sy-0k2 / gh-495: expose the full resulting config when available so
    # agents can see exactly what would (or did) land on disk.
    if result.resulting_config is not None:
        payload["resulting_config"] = result.resulting_config
    return json.dumps(payload)


def format_resulting_config(result: InstallResult) -> str | None:
    """Render the full resulting MCP config as pretty-printed JSON.

    Returns ``None`` when no resulting-config snapshot is available
    (callers should fall back to ``format_text`` in that case).
    """
    if result.resulting_config is None:
        return None
    return json.dumps(result.resulting_config, indent=2, sort_keys=False)


__all__ = [
    "DEFAULT_SERVER_NAME",
    "InstallResult",
    "build_entry",
    "format_json",
    "format_resulting_config",
    "format_text",
    "install",
    "mcp_extra_available",
    "parse_env_pairs",
    "resolve_command",
    "resolve_target",
    "uninstall",
]
