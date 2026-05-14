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
from pathlib import Path
from typing import Any

DEFAULT_SERVER_NAME = "synth_panel"


@dataclass
class InstallResult:
    """Outcome of an install/uninstall call."""

    target: Path
    name: str
    action: str  # "installed", "updated", "removed", "noop", "would-install", "would-update", "would-remove"
    entry: dict[str, Any] | None  # the server entry that was written (None for remove/noop)


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

    if existing == new_entry:
        return InstallResult(target=target, name=name, action="noop", entry=new_entry)

    is_update = name in servers
    if is_update and not force:
        raise FileExistsError(
            f"MCP server {name!r} already exists in {target}. Pass --force to overwrite, or choose a different --name."
        )

    if dry_run:
        servers[name] = new_entry
        config["mcpServers"] = servers
        action = "would-update" if is_update else "would-install"
        return InstallResult(target=target, name=name, action=action, entry=new_entry)

    servers[name] = new_entry
    config["mcpServers"] = servers
    _atomic_write(target, config)
    action = "updated" if is_update else "installed"
    return InstallResult(target=target, name=name, action=action, entry=new_entry)


def uninstall(*, target: Path, name: str, dry_run: bool) -> InstallResult:
    """Remove the named server entry. No-op if it isn't present."""
    if not target.exists():
        return InstallResult(target=target, name=name, action="noop", entry=None)

    config = _load_config(target)
    servers = config.get("mcpServers")
    if not isinstance(servers, dict) or name not in servers:
        return InstallResult(target=target, name=name, action="noop", entry=None)

    if dry_run:
        return InstallResult(target=target, name=name, action="would-remove", entry=None)

    del servers[name]
    config["mcpServers"] = servers
    _atomic_write(target, config)
    return InstallResult(target=target, name=name, action="removed", entry=None)


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
    return json.dumps(payload)


__all__ = [
    "DEFAULT_SERVER_NAME",
    "InstallResult",
    "build_entry",
    "format_json",
    "format_text",
    "install",
    "parse_env_pairs",
    "resolve_command",
    "resolve_target",
    "uninstall",
]
