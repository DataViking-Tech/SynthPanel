"""Plugin Registry — read-only snapshot of enabled plugins (SPEC.md §9)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synth_panel.plugins.manifest import PluginManifest, PluginMetadata

from synth_panel.plugins.manifest import PluginHooks


class PluginRegistry:
    """Immutable snapshot of enabled plugins and their aggregated hooks.

    Created by ``PluginManager.build_registry()``.
    """

    def __init__(
        self,
        enabled_plugins: list[tuple[PluginMetadata, PluginManifest]],
    ) -> None:
        from synth_panel.plugins.manifest import PluginMetadata, PluginManifest

        self._plugins: list[tuple[PluginMetadata, PluginManifest]] = list(enabled_plugins)
        self._aggregated_hooks = self._aggregate_hooks()

    def _aggregate_hooks(self) -> PluginHooks:
        """Combine hooks from all enabled plugins into a single hook set."""
        pre: list[str] = []
        post: list[str] = []
        post_failure: list[str] = []
        for _metadata, manifest in self._plugins:
            pre.extend(manifest.hooks.pre_tool_use)
            post.extend(manifest.hooks.post_tool_use)
            post_failure.extend(manifest.hooks.post_tool_use_failure)
        return PluginHooks(
            pre_tool_use=pre,
            post_tool_use=post,
            post_tool_use_failure=post_failure,
        )

    @property
    def plugins(self) -> list[PluginMetadata]:
        """All enabled plugins."""
        from synth_panel.plugins.manifest import PluginMetadata
        return [m for m, _ in self._plugins]

    @property
    def hooks(self) -> PluginHooks:
        """Aggregated hooks from all enabled plugins."""
        return self._aggregated_hooks

    @property
    def lifecycle_init_commands(self) -> list[str]:
        """All init commands from all enabled plugins, in order."""
        cmds: list[str] = []
        for _, manifest in self._plugins:
            cmds.extend(manifest.lifecycle.init)
        return cmds

    @property
    def lifecycle_shutdown_commands(self) -> list[str]:
        """All shutdown commands from all enabled plugins, in order."""
        cmds: list[str] = []
        for _, manifest in self._plugins:
            cmds.extend(manifest.lifecycle.shutdown)
        return cmds
