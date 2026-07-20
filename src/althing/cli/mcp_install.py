"""Helpers for `althing mcp install` (sy-skf).

Edits a host's MCP config JSON file in-place (or creates it) so that
althing is registered as a stdio MCP server. The default target is
Claude Code's user config (~/.claude.json), but the helpers are
host-agnostic — they only touch the ``mcpServers`` mapping.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

DEFAULT_SERVER_NAME = "althing"


# ---------------------------------------------------------------------------
# Host registry (synthbench#262): named MCP hosts the installer knows how to
# target. Each host maps to a concrete config file, the JSON key its schema
# nests servers under (Zed uses ``context_servers``, everyone else
# ``mcpServers``), and any extra fields the entry needs (Zed requires
# ``"source": "custom"``). The JSON written per host mirrors the README's
# "Use with Claude Code / Cursor / Windsurf / Zed" section exactly.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HostSpec:
    """A named MCP host the installer can write a config for."""

    key: str  # CLI value, e.g. "claude-code"
    label: str  # human name for the restart hint, e.g. "Claude Code"
    config_key: str = "mcpServers"  # JSON key the servers mapping lives under
    # Extra fields prepended to the server entry (Zed: {"source": "custom"}).
    entry_extra: tuple[tuple[str, str], ...] = ()
    supports_project_scope: bool = False


def _claude_desktop_path() -> Path:
    """Platform-specific Claude Desktop config path."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        return base / "Claude" / "claude_desktop_config.json"
    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


HOSTS: dict[str, HostSpec] = {
    "claude-code": HostSpec(
        key="claude-code",
        label="Claude Code",
        supports_project_scope=True,
    ),
    "claude-desktop": HostSpec(
        key="claude-desktop",
        label="Claude Desktop",
    ),
    "cursor": HostSpec(
        key="cursor",
        label="Cursor",
        supports_project_scope=True,
    ),
    "windsurf": HostSpec(
        key="windsurf",
        label="Windsurf",
    ),
    "zed": HostSpec(
        key="zed",
        label="Zed",
        config_key="context_servers",
        entry_extra=(("source", "custom"),),
    ),
}


def host_config_path(host: HostSpec, scope: str = "user") -> Path:
    """Resolve the config file a named host reads, honoring ``scope``.

    Raises ValueError when ``scope == "project"`` for a host that has no
    project-level config file.
    """
    if scope == "project" and not host.supports_project_scope:
        raise ValueError(f"host {host.key!r} has no project-scope config; use --scope user (default).")
    if host.key == "claude-code":
        return Path.cwd() / ".mcp.json" if scope == "project" else Path.home() / ".claude.json"
    if host.key == "claude-desktop":
        return _claude_desktop_path()
    if host.key == "cursor":
        if scope == "project":
            return Path.cwd() / ".cursor" / "mcp.json"
        return Path.home() / ".cursor" / "mcp.json"
    if host.key == "windsurf":
        return Path.home() / ".codeium" / "windsurf" / "mcp_config.json"
    if host.key == "zed":
        return Path.home() / ".config" / "zed" / "settings.json"
    raise ValueError(f"unknown host {host.key!r}")


def detect_hosts() -> list[tuple[HostSpec, Path]]:
    """Return (host, existing-config-path) pairs for hosts present on this machine.

    Only user-scope configs that already exist on disk are reported —
    auto-detection never invents a config file for a host that isn't
    installed. Project-scope files (./.mcp.json, ./.cursor/mcp.json) are
    deliberate opt-ins via ``--scope project`` and are not detected.
    """
    found: list[tuple[HostSpec, Path]] = []
    for host in HOSTS.values():
        path = host_config_path(host, "user")
        if path.is_file():
            found.append((host, path))
    return found


# sy-xyn: actionable copy used by every hard-error surface that detects
# a missing `mcp` extra (install refusal + `mcp-serve` startup guard).
# Distinct from MCP_EXTRA_WARNING (#512) which is the SOFT warning
# embedded in `InstallResult.warnings`. The two stay separate because
# refusal copy + advisory copy need different framing: the refusal
# stops the user and tells them how to proceed; the warning is the
# nudge embedded in a result they're already consuming. Both carry
# "althing[mcp]" verbatim so the install command surfaces in
# either context.
MISSING_MCP_EXTRA_MESSAGE = (
    "the 'mcp' optional dependency is not installed in this Python env. "
    "Install it with:  pip install 'althing[mcp]'  "
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
    "Optional MCP dependency is not installed; run `pip install 'althing[mcp]'` before using `althing mcp-serve`."
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
    """Resolve the command string the host should invoke (#539).

    The MCP host launches this command in its own environment, which may
    not have the installing venv's ``bin`` on ``PATH``. A bare
    ``"althing"`` then fails to launch. Resolve a robust absolute path,
    in order of preference:

    1. an explicit ``--command`` override (used verbatim);
    2. ``shutil.which("althing")`` — the launcher on the current PATH;
    3. the running entry point's real path (``os.path.realpath(sys.argv[0])``)
       when it names a ``althing`` launcher — this is the venv-installed
       console script even when its ``bin`` is not on PATH;
    4. a ``althing`` executable sitting next to the running interpreter
       (``<sys.executable dir>/althing``) — the canonical venv layout;
    5. the literal ``"althing"`` as a last resort.
    """
    if explicit:
        return explicit

    found = shutil.which("althing")
    if found:
        return os.path.realpath(found)

    # The console script we're (likely) running under. In a venv this is an
    # absolute path like <venv>/bin/althing even when that bin dir is not
    # on PATH, so it's a reliable launcher for the host to invoke.
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0:
        real_argv0 = os.path.realpath(argv0)
        base = os.path.basename(real_argv0)
        if base in ("althing", "althing.exe") and os.path.isfile(real_argv0):
            return real_argv0

    # A althing launcher next to the interpreter (canonical venv layout:
    # <venv>/bin/python + <venv>/bin/althing). Check the literal
    # sys.executable dir FIRST: in a venv, sys.executable is <venv>/bin/python
    # (a symlink), so resolving it would jump to the base interpreter's bin
    # and miss the venv's console script. Fall back to the realpath dir for
    # non-venv layouts.
    if sys.executable:
        seen: set[str] = set()
        for bindir in (
            os.path.dirname(sys.executable),
            os.path.dirname(os.path.realpath(sys.executable)),
        ):
            if not bindir or bindir in seen:
                continue
            seen.add(bindir)
            for name in ("althing", "althing.exe"):
                candidate = os.path.join(bindir, name)
                if os.path.isfile(candidate):
                    return candidate

    return "althing"


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


def build_entry(
    command: str,
    env: dict[str, str],
    extra: tuple[tuple[str, str], ...] = (),
) -> dict[str, Any]:
    """Build the JSON object the host expects under its servers mapping.

    ``extra`` fields (e.g. Zed's ``"source": "custom"``) are placed first
    so the written JSON matches the README's per-host snippets verbatim.
    """
    entry: dict[str, Any] = dict(extra)
    entry["command"] = command
    entry["args"] = ["mcp-serve"]
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
    config_key: str = "mcpServers",
    entry_extra: tuple[tuple[str, str], ...] = (),
) -> InstallResult:
    """Insert or update the named server entry.

    ``config_key`` selects the JSON key the servers mapping nests under
    (``"context_servers"`` for Zed, ``"mcpServers"`` everywhere else) and
    ``entry_extra`` carries host-specific entry fields (synthbench#262).
    """
    config = _load_config(target)
    servers = config.get(config_key)
    if not isinstance(servers, dict):
        servers = {}

    existing = servers.get(name)
    new_entry = build_entry(command, env, entry_extra)
    warnings = None if mcp_extra_available() else [MCP_EXTRA_WARNING]

    if existing == new_entry:
        # No-op: report the current on-disk config as the resulting config
        # so dry-run consumers can still inspect the unchanged state.
        config[config_key] = servers
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
    config[config_key] = servers
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


def uninstall(
    *,
    target: Path,
    name: str,
    dry_run: bool,
    config_key: str = "mcpServers",
) -> InstallResult:
    """Remove the named server entry. No-op if it isn't present.

    Removes exactly the one entry under ``config[config_key][name]`` —
    every other server and every unrelated top-level key is preserved.
    """
    if not target.exists():
        return InstallResult(
            target=target,
            name=name,
            action="noop",
            entry=None,
            resulting_config={},
        )

    config = _load_config(target)
    servers = config.get(config_key)
    if not isinstance(servers, dict) or name not in servers:
        # Surface the unchanged config so dry-run consumers see the
        # current state even on no-op.
        if isinstance(servers, dict):
            config[config_key] = servers
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
        preview[config_key] = preview_servers
        return InstallResult(
            target=target,
            name=name,
            action="would-remove",
            entry=None,
            resulting_config=preview,
        )

    del servers[name]
    config[config_key] = servers
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
    "HOSTS",
    "HostSpec",
    "InstallResult",
    "build_entry",
    "detect_hosts",
    "format_json",
    "format_resulting_config",
    "format_text",
    "host_config_path",
    "install",
    "mcp_extra_available",
    "parse_env_pairs",
    "resolve_command",
    "resolve_target",
    "uninstall",
]
