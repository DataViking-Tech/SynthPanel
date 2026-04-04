"""Plugin and Extension System (SPEC.md §9).

Manifest-based plugin discovery with install/enable/disable/uninstall,
hook interception (pre-tool-use, post-tool-use, post-tool-use-failure),
and lifecycle commands (init/shutdown).
"""

from synth_panel.plugins.manifest import (
    PluginHooks,
    PluginLifecycle,
    PluginManifest,
    PluginMetadata,
    PluginKind,
)
from synth_panel.plugins.manager import PluginManager
from synth_panel.plugins.registry import PluginRegistry
from synth_panel.plugins.hooks import ShellHookRunner

__all__ = [
    "PluginHooks",
    "PluginLifecycle",
    "PluginManifest",
    "PluginMetadata",
    "PluginKind",
    "PluginManager",
    "PluginRegistry",
    "ShellHookRunner",
]
