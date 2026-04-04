"""Tests for the plugin and extension system (SPEC.md §9)."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from synth_panel.plugins.manifest import (
    MANIFEST_FILENAME,
    PluginHooks,
    PluginKind,
    PluginLifecycle,
    PluginManifest,
    PluginMetadata,
)
from synth_panel.plugins.manager import (
    PluginError,
    PluginInstallError,
    PluginManager,
    PluginNotFoundError,
)
from synth_panel.plugins.registry import PluginRegistry
from synth_panel.plugins.hooks import ShellHookRunner, run_lifecycle_commands
from synth_panel.runtime import HookResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_manifest(plugin_dir: Path, manifest: dict[str, Any]) -> Path:
    """Write a plugin manifest YAML to a directory."""
    import yaml

    plugin_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = plugin_dir / MANIFEST_FILENAME
    manifest_path.write_text(yaml.dump(manifest), encoding="utf-8")
    return manifest_path


def _minimal_manifest(**overrides: Any) -> dict[str, Any]:
    base = {"name": "test-plugin", "version": "1.0.0", "description": "A test plugin"}
    base.update(overrides)
    return base


# ===========================================================================
# Manifest tests
# ===========================================================================


class TestPluginManifest:
    def test_from_dict_minimal(self) -> None:
        manifest = PluginManifest.from_dict({"name": "foo", "version": "1.0.0"})
        assert manifest.name == "foo"
        assert manifest.version == "1.0.0"
        assert manifest.default_enabled is True
        assert manifest.hooks == PluginHooks()
        assert manifest.lifecycle == PluginLifecycle()

    def test_from_dict_full(self) -> None:
        data = {
            "name": "full-plugin",
            "version": "2.0.0",
            "description": "Full plugin",
            "permissions": ["read", "write"],
            "default_enabled": False,
            "hooks": {
                "pre_tool_use": ["echo pre"],
                "post_tool_use": ["echo post"],
                "post_tool_use_failure": ["echo fail"],
            },
            "lifecycle": {
                "init": ["echo init"],
                "shutdown": ["echo shutdown"],
            },
            "tools": [{"name": "my-tool"}],
            "commands": [{"name": "my-cmd"}],
        }
        manifest = PluginManifest.from_dict(data)
        assert manifest.name == "full-plugin"
        assert manifest.default_enabled is False
        assert manifest.hooks.pre_tool_use == ["echo pre"]
        assert manifest.hooks.post_tool_use == ["echo post"]
        assert manifest.hooks.post_tool_use_failure == ["echo fail"]
        assert manifest.lifecycle.init == ["echo init"]
        assert manifest.lifecycle.shutdown == ["echo shutdown"]
        assert manifest.permissions == ["read", "write"]

    def test_from_file(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, _minimal_manifest())
        manifest = PluginManifest.from_file(tmp_path / MANIFEST_FILENAME)
        assert manifest.name == "test-plugin"
        assert manifest.version == "1.0.0"

    def test_from_file_missing_name(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, {"version": "1.0.0"})
        with pytest.raises(ValueError, match="name"):
            PluginManifest.from_file(tmp_path / MANIFEST_FILENAME)

    def test_from_file_invalid_content(self, tmp_path: Path) -> None:
        path = tmp_path / MANIFEST_FILENAME
        path.write_text("just a string", encoding="utf-8")
        with pytest.raises(ValueError, match="expected mapping"):
            PluginManifest.from_file(path)


# ===========================================================================
# Plugin Manager tests
# ===========================================================================


class TestPluginManager:
    def test_install_and_list(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        source_dir = tmp_path / "source" / "my-plugin"
        _write_manifest(source_dir, _minimal_manifest(name="my-plugin"))

        mgr = PluginManager(config_dir)
        metadata = mgr.install(source_dir)

        assert metadata.name == "my-plugin"
        assert metadata.id == "my-plugin"
        assert metadata.kind == PluginKind.EXTERNAL

        plugins = mgr.list_plugins()
        assert len(plugins) == 1
        assert plugins[0].name == "my-plugin"

    def test_install_missing_manifest(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        source_dir = tmp_path / "source" / "bad"
        source_dir.mkdir(parents=True)

        mgr = PluginManager(config_dir)
        with pytest.raises(PluginInstallError, match="manifest not found"):
            mgr.install(source_dir)

    def test_install_malformed_manifest(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        source_dir = tmp_path / "source" / "bad"
        source_dir.mkdir(parents=True)
        (source_dir / MANIFEST_FILENAME).write_text("not valid yaml: [", encoding="utf-8")

        mgr = PluginManager(config_dir)
        with pytest.raises(PluginInstallError, match="Malformed"):
            mgr.install(source_dir)

    def test_enable_disable(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        source_dir = tmp_path / "source" / "my-plugin"
        _write_manifest(source_dir, _minimal_manifest(name="my-plugin"))

        mgr = PluginManager(config_dir)
        mgr.install(source_dir)
        assert mgr.is_enabled("my-plugin") is True

        mgr.disable("my-plugin")
        assert mgr.is_enabled("my-plugin") is False

        mgr.enable("my-plugin")
        assert mgr.is_enabled("my-plugin") is True

    def test_enable_nonexistent(self, tmp_path: Path) -> None:
        mgr = PluginManager(tmp_path / "config")
        with pytest.raises(PluginNotFoundError):
            mgr.enable("nope")

    def test_disable_nonexistent(self, tmp_path: Path) -> None:
        mgr = PluginManager(tmp_path / "config")
        with pytest.raises(PluginNotFoundError):
            mgr.disable("nope")

    def test_uninstall(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        source_dir = tmp_path / "source" / "my-plugin"
        _write_manifest(source_dir, _minimal_manifest(name="my-plugin"))

        mgr = PluginManager(config_dir)
        mgr.install(source_dir)
        assert len(mgr.list_plugins()) == 1

        mgr.uninstall("my-plugin")
        assert len(mgr.list_plugins()) == 0

    def test_uninstall_nonexistent(self, tmp_path: Path) -> None:
        mgr = PluginManager(tmp_path / "config")
        with pytest.raises(PluginNotFoundError):
            mgr.uninstall("nope")

    def test_reinstall_overwrites(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        source_dir = tmp_path / "source" / "my-plugin"
        _write_manifest(
            source_dir, _minimal_manifest(name="my-plugin", version="1.0.0")
        )

        mgr = PluginManager(config_dir)
        mgr.install(source_dir)

        # Update version and reinstall
        _write_manifest(
            source_dir, _minimal_manifest(name="my-plugin", version="2.0.0")
        )
        metadata = mgr.install(source_dir)
        assert metadata.version == "2.0.0"
        assert len(mgr.list_plugins()) == 1

    def test_default_enabled_false(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        source_dir = tmp_path / "source" / "my-plugin"
        _write_manifest(
            source_dir,
            _minimal_manifest(name="my-plugin", default_enabled=False),
        )

        mgr = PluginManager(config_dir)
        mgr.install(source_dir)
        assert mgr.is_enabled("my-plugin") is False

    def test_build_registry(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        source1 = tmp_path / "s1" / "p1"
        source2 = tmp_path / "s2" / "p2"
        _write_manifest(source1, _minimal_manifest(
            name="p1",
            hooks={"pre_tool_use": ["echo p1-pre"]},
        ))
        _write_manifest(source2, _minimal_manifest(
            name="p2",
            hooks={"pre_tool_use": ["echo p2-pre"]},
        ))

        mgr = PluginManager(config_dir)
        mgr.install(source1)
        mgr.install(source2)
        mgr.disable("p2")

        registry = mgr.build_registry()
        assert len(registry.plugins) == 1
        assert registry.plugins[0].name == "p1"
        assert registry.hooks.pre_tool_use == ["echo p1-pre"]

    def test_state_persists(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        source_dir = tmp_path / "source" / "my-plugin"
        _write_manifest(source_dir, _minimal_manifest(name="my-plugin"))

        mgr1 = PluginManager(config_dir)
        mgr1.install(source_dir)
        mgr1.disable("my-plugin")

        # New manager instance reads persisted state
        mgr2 = PluginManager(config_dir)
        assert mgr2.is_enabled("my-plugin") is False


# ===========================================================================
# Plugin Registry tests
# ===========================================================================


class TestPluginRegistry:
    def test_aggregates_hooks(self) -> None:
        m1 = PluginMetadata(
            id="p1", name="p1", version="1.0.0", description="",
            kind=PluginKind.EXTERNAL, source_path="", default_enabled=True, root_dir="",
        )
        man1 = PluginManifest(
            name="p1", version="1.0.0",
            hooks=PluginHooks(pre_tool_use=["cmd1"], post_tool_use=["cmd2"]),
        )
        m2 = PluginMetadata(
            id="p2", name="p2", version="1.0.0", description="",
            kind=PluginKind.EXTERNAL, source_path="", default_enabled=True, root_dir="",
        )
        man2 = PluginManifest(
            name="p2", version="1.0.0",
            hooks=PluginHooks(pre_tool_use=["cmd3"], post_tool_use_failure=["cmd4"]),
        )
        registry = PluginRegistry([(m1, man1), (m2, man2)])

        assert registry.hooks.pre_tool_use == ["cmd1", "cmd3"]
        assert registry.hooks.post_tool_use == ["cmd2"]
        assert registry.hooks.post_tool_use_failure == ["cmd4"]

    def test_lifecycle_commands(self) -> None:
        m1 = PluginMetadata(
            id="p1", name="p1", version="1.0.0", description="",
            kind=PluginKind.EXTERNAL, source_path="", default_enabled=True, root_dir="",
        )
        man1 = PluginManifest(
            name="p1", version="1.0.0",
            lifecycle=PluginLifecycle(init=["echo init1"], shutdown=["echo shut1"]),
        )
        registry = PluginRegistry([(m1, man1)])

        assert registry.lifecycle_init_commands == ["echo init1"]
        assert registry.lifecycle_shutdown_commands == ["echo shut1"]

    def test_empty_registry(self) -> None:
        registry = PluginRegistry([])
        assert registry.plugins == []
        assert registry.hooks == PluginHooks()


# ===========================================================================
# Shell Hook Runner tests
# ===========================================================================


class TestShellHookRunner:
    def test_allow_exit_0(self) -> None:
        hooks = PluginHooks(pre_tool_use=["echo allowed"])
        runner = ShellHookRunner(hooks)
        result = runner.run_pre_tool_use("my_tool", {"key": "val"})
        assert not result.denied
        assert not result.failed
        assert result.messages == ["allowed"]

    def test_deny_exit_2(self) -> None:
        hooks = PluginHooks(pre_tool_use=["echo 'not allowed' && exit 2"])
        runner = ShellHookRunner(hooks)
        result = runner.run_pre_tool_use("my_tool", {})
        assert result.denied
        assert not result.failed
        assert "not allowed" in result.messages[0]

    def test_failure_exit_1(self) -> None:
        hooks = PluginHooks(pre_tool_use=["echo 'broken' && exit 1"])
        runner = ShellHookRunner(hooks)
        result = runner.run_pre_tool_use("my_tool", {})
        assert not result.denied
        assert result.failed
        assert "broken" in result.messages[0]

    def test_chain_short_circuits_on_deny(self) -> None:
        hooks = PluginHooks(pre_tool_use=[
            "echo 'denied' && exit 2",
            "echo 'should not run'",
        ])
        runner = ShellHookRunner(hooks)
        result = runner.run_pre_tool_use("my_tool", {})
        assert result.denied
        assert len(result.messages) == 1
        assert "denied" in result.messages[0]

    def test_chain_short_circuits_on_failure(self) -> None:
        hooks = PluginHooks(pre_tool_use=[
            "exit 1",
            "echo 'should not run'",
        ])
        runner = ShellHookRunner(hooks)
        result = runner.run_pre_tool_use("my_tool", {})
        assert result.failed
        assert len(result.messages) == 0

    def test_multiple_allow_commands(self) -> None:
        hooks = PluginHooks(pre_tool_use=["echo msg1", "echo msg2"])
        runner = ShellHookRunner(hooks)
        result = runner.run_pre_tool_use("my_tool", {})
        assert not result.denied
        assert not result.failed
        assert result.messages == ["msg1", "msg2"]

    def test_env_vars_passed(self) -> None:
        hooks = PluginHooks(pre_tool_use=[
            'echo "$HOOK_EVENT:$HOOK_TOOL_NAME:$HOOK_TOOL_IS_ERROR"'
        ])
        runner = ShellHookRunner(hooks)
        result = runner.run_pre_tool_use("search", {"query": "test"})
        assert result.messages == ["pre_tool_use:search:0"]

    def test_stdin_json_payload(self) -> None:
        hooks = PluginHooks(pre_tool_use=["cat"])
        runner = ShellHookRunner(hooks)
        result = runner.run_pre_tool_use("search", {"query": "test"})
        payload = json.loads(result.messages[0])
        assert payload["event"] == "pre_tool_use"
        assert payload["tool_name"] == "search"
        assert payload["tool_input"] == {"query": "test"}
        assert payload["is_error"] is False

    def test_post_tool_use(self) -> None:
        hooks = PluginHooks(post_tool_use=["echo post-ok"])
        runner = ShellHookRunner(hooks)
        result = runner.run_post_tool_use("my_tool", {}, "output", is_error=False)
        assert result.messages == ["post-ok"]

    def test_post_tool_use_failure_event(self) -> None:
        hooks = PluginHooks(post_tool_use_failure=[
            'echo "$HOOK_EVENT:$HOOK_TOOL_IS_ERROR"'
        ])
        runner = ShellHookRunner(hooks)
        result = runner.run_post_tool_use("my_tool", {}, "err", is_error=True)
        assert result.messages == ["post_tool_use_failure:1"]

    def test_empty_hooks(self) -> None:
        hooks = PluginHooks()
        runner = ShellHookRunner(hooks)
        result = runner.run_pre_tool_use("my_tool", {})
        assert not result.denied
        assert not result.failed
        assert result.messages == []

    def test_no_stdout_no_message(self) -> None:
        hooks = PluginHooks(pre_tool_use=["true"])
        runner = ShellHookRunner(hooks)
        result = runner.run_pre_tool_use("my_tool", {})
        assert result.messages == []


# ===========================================================================
# Lifecycle command tests
# ===========================================================================


class TestLifecycleCommands:
    def test_run_init_commands(self, tmp_path: Path) -> None:
        marker = tmp_path / "init_ran"
        run_lifecycle_commands([f"touch {marker}"])
        assert marker.exists()

    def test_run_shutdown_commands(self, tmp_path: Path) -> None:
        marker = tmp_path / "shutdown_ran"
        run_lifecycle_commands([f"touch {marker}"])
        assert marker.exists()

    def test_failure_does_not_stop_chain(self, tmp_path: Path) -> None:
        marker = tmp_path / "second_ran"
        run_lifecycle_commands([
            "exit 1",
            f"touch {marker}",
        ])
        assert marker.exists()

    def test_empty_commands(self) -> None:
        run_lifecycle_commands([])  # Should not raise
