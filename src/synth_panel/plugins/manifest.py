"""Plugin manifest and metadata models (SPEC.md §9)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


MANIFEST_FILENAME = "plugin.yaml"


class PluginKind(Enum):
    BUILTIN = "builtin"
    BUNDLED = "bundled"
    EXTERNAL = "external"


@dataclass
class PluginHooks:
    """Hook command declarations for a plugin."""

    pre_tool_use: list[str] = field(default_factory=list)
    post_tool_use: list[str] = field(default_factory=list)
    post_tool_use_failure: list[str] = field(default_factory=list)


@dataclass
class PluginLifecycle:
    """Lifecycle command declarations for a plugin."""

    init: list[str] = field(default_factory=list)
    shutdown: list[str] = field(default_factory=list)


@dataclass
class PluginManifest:
    """Parsed plugin manifest from plugin.yaml."""

    name: str
    version: str
    description: str = ""
    permissions: list[str] = field(default_factory=list)
    default_enabled: bool = True
    hooks: PluginHooks = field(default_factory=PluginHooks)
    lifecycle: PluginLifecycle = field(default_factory=PluginLifecycle)
    tools: list[dict[str, Any]] = field(default_factory=list)
    commands: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PluginManifest:
        hooks_data = data.get("hooks", {})
        hooks = PluginHooks(
            pre_tool_use=hooks_data.get("pre_tool_use", []),
            post_tool_use=hooks_data.get("post_tool_use", []),
            post_tool_use_failure=hooks_data.get("post_tool_use_failure", []),
        )
        lifecycle_data = data.get("lifecycle", {})
        lifecycle = PluginLifecycle(
            init=lifecycle_data.get("init", []),
            shutdown=lifecycle_data.get("shutdown", []),
        )
        return cls(
            name=data["name"],
            version=data["version"],
            description=data.get("description", ""),
            permissions=data.get("permissions", []),
            default_enabled=data.get("default_enabled", True),
            hooks=hooks,
            lifecycle=lifecycle,
            tools=data.get("tools", []),
            commands=data.get("commands", []),
        )

    @classmethod
    def from_file(cls, path: Path) -> PluginManifest:
        """Load a manifest from a YAML file."""
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid plugin manifest: expected mapping, got {type(data).__name__}")
        if "name" not in data or "version" not in data:
            raise ValueError("Plugin manifest must contain 'name' and 'version' fields")
        return cls.from_dict(data)


@dataclass
class PluginMetadata:
    """Runtime representation of an installed plugin."""

    id: str
    name: str
    version: str
    description: str
    kind: PluginKind
    source_path: str
    default_enabled: bool
    root_dir: str
