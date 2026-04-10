"""Plugin and Extension System (SPEC.md §9).

Manifest-based plugin discovery with install/enable/disable/uninstall,
hook interception (pre-tool-use, post-tool-use, post-tool-use-failure),
and lifecycle commands (init/shutdown).
"""

from __future__ import annotations

from synth_panel.plugins.hooks import ShellHookRunner
from synth_panel.plugins.manager import PluginManager
from synth_panel.plugins.manifest import (
    PluginHooks,
    PluginKind,
    PluginLifecycle,
    PluginManifest,
    PluginMetadata,
)
from synth_panel.plugins.registry import PluginRegistry

__all__ = [
    "PluginHooks",
    "PluginKind",
    "PluginLifecycle",
    "PluginManager",
    "PluginManifest",
    "PluginMetadata",
    "PluginRegistry",
    "ShellHookRunner",
]
