"""Plugin Manager — discovers, installs, enables/disables, and uninstalls plugins (SPEC.md §9)."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from synth_panel.plugins.manifest import (
    MANIFEST_FILENAME,
    PluginKind,
    PluginManifest,
    PluginMetadata,
)
from synth_panel.plugins.registry import PluginRegistry


STATE_FILENAME = "plugins.json"


class PluginError(Exception):
    """Base error for plugin operations."""


class PluginNotFoundError(PluginError):
    """Raised when a plugin is not found."""


class PluginInstallError(PluginError):
    """Raised when plugin installation fails."""


class PluginManager:
    """Manages plugin lifecycle: install, enable, disable, uninstall.

    Plugins are stored under ``config_dir/plugins/<plugin-name>/``.
    Plugin state (enabled/disabled) is tracked in ``config_dir/plugins.json``.
    """

    def __init__(self, config_dir: Path) -> None:
        self._config_dir = config_dir
        self._plugins_dir = config_dir / "plugins"
        self._state_path = config_dir / STATE_FILENAME
        self._plugins_dir.mkdir(parents=True, exist_ok=True)
        self._state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        if self._state_path.exists():
            return json.loads(self._state_path.read_text(encoding="utf-8"))
        return {"plugins": {}}

    def _save_state(self) -> None:
        self._state_path.write_text(
            json.dumps(self._state, indent=2) + "\n", encoding="utf-8"
        )

    def install(self, source_dir: str | Path) -> PluginMetadata:
        """Install a plugin from a source directory.

        Copies the plugin into the config directory and registers it.
        """
        source = Path(source_dir)
        manifest_path = source / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise PluginInstallError(
                f"Plugin manifest not found: {manifest_path}"
            )

        try:
            manifest = PluginManifest.from_file(manifest_path)
        except (ValueError, Exception) as exc:
            raise PluginInstallError(f"Malformed plugin manifest: {exc}") from exc

        plugin_id = manifest.name
        dest = self._plugins_dir / plugin_id

        if dest.exists():
            shutil.rmtree(dest)

        shutil.copytree(source, dest)

        metadata = PluginMetadata(
            id=plugin_id,
            name=manifest.name,
            version=manifest.version,
            description=manifest.description,
            kind=PluginKind.EXTERNAL,
            source_path=str(source),
            default_enabled=manifest.default_enabled,
            root_dir=str(dest),
        )

        self._state["plugins"][plugin_id] = {
            "enabled": manifest.default_enabled,
            "version": manifest.version,
            "source_path": str(source),
        }
        self._save_state()

        return metadata

    def uninstall(self, plugin_id: str) -> None:
        """Uninstall a plugin by removing its directory and state."""
        if plugin_id not in self._state["plugins"]:
            raise PluginNotFoundError(f"Plugin not found: {plugin_id}")

        dest = self._plugins_dir / plugin_id
        if dest.exists():
            shutil.rmtree(dest)

        del self._state["plugins"][plugin_id]
        self._save_state()

    def enable(self, plugin_id: str) -> None:
        """Enable a plugin."""
        if plugin_id not in self._state["plugins"]:
            raise PluginNotFoundError(f"Plugin not found: {plugin_id}")
        self._state["plugins"][plugin_id]["enabled"] = True
        self._save_state()

    def disable(self, plugin_id: str) -> None:
        """Disable a plugin."""
        if plugin_id not in self._state["plugins"]:
            raise PluginNotFoundError(f"Plugin not found: {plugin_id}")
        self._state["plugins"][plugin_id]["enabled"] = False
        self._save_state()

    def list_plugins(self) -> list[PluginMetadata]:
        """List all installed plugins with metadata."""
        result: list[PluginMetadata] = []
        for plugin_id, info in self._state["plugins"].items():
            plugin_dir = self._plugins_dir / plugin_id
            manifest_path = plugin_dir / MANIFEST_FILENAME
            if not manifest_path.exists():
                continue
            manifest = PluginManifest.from_file(manifest_path)
            result.append(PluginMetadata(
                id=plugin_id,
                name=manifest.name,
                version=manifest.version,
                description=manifest.description,
                kind=PluginKind.EXTERNAL,
                source_path=info.get("source_path", ""),
                default_enabled=manifest.default_enabled,
                root_dir=str(plugin_dir),
            ))
        return result

    def is_enabled(self, plugin_id: str) -> bool:
        """Check if a plugin is enabled."""
        info = self._state["plugins"].get(plugin_id)
        if info is None:
            raise PluginNotFoundError(f"Plugin not found: {plugin_id}")
        return info.get("enabled", False)

    def build_registry(self) -> PluginRegistry:
        """Build a read-only registry snapshot of all enabled plugins."""
        enabled_plugins: list[tuple[PluginMetadata, PluginManifest]] = []
        for plugin_id, info in self._state["plugins"].items():
            if not info.get("enabled", False):
                continue
            plugin_dir = self._plugins_dir / plugin_id
            manifest_path = plugin_dir / MANIFEST_FILENAME
            if not manifest_path.exists():
                continue
            manifest = PluginManifest.from_file(manifest_path)
            metadata = PluginMetadata(
                id=plugin_id,
                name=manifest.name,
                version=manifest.version,
                description=manifest.description,
                kind=PluginKind.EXTERNAL,
                source_path=info.get("source_path", ""),
                default_enabled=manifest.default_enabled,
                root_dir=str(plugin_dir),
            )
            enabled_plugins.append((metadata, manifest))
        return PluginRegistry(enabled_plugins)
