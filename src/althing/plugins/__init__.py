"""Plugin and Extension System (SPEC.md §9).

Manifest-based plugin discovery with install/enable/disable/uninstall,
hook interception (pre-tool-use, post-tool-use, post-tool-use-failure),
and lifecycle commands (init/shutdown).
"""

from __future__ import annotations

from althing.plugins.hooks import ShellHookRunner
from althing.plugins.lint import LintIssue, LintReport, lint_plugin
from althing.plugins.manager import PluginManager
from althing.plugins.manifest import (
    PluginHooks,
    PluginKind,
    PluginLifecycle,
    PluginManifest,
    PluginMetadata,
)
from althing.plugins.registry import PluginRegistry

__all__ = [
    "LintIssue",
    "LintReport",
    "PluginHooks",
    "PluginKind",
    "PluginLifecycle",
    "PluginManager",
    "PluginManifest",
    "PluginMetadata",
    "PluginRegistry",
    "ShellHookRunner",
    "lint_plugin",
]
