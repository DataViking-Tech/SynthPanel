"""Tests for `synthpanel mcp install` (sy-skf)."""

from __future__ import annotations

import builtins
import json
import os
import stat
from pathlib import Path

import pytest

from synth_panel.cli import mcp_install
from synth_panel.cli.parser import build_parser
from synth_panel.main import main

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestParser:
    def test_install_registered(self):
        args = build_parser().parse_args(["mcp", "install"])
        assert args.command == "mcp"
        assert args.mcp_command == "install"
        assert args.uninstall is False
        assert args.scope == "user"
        assert args.target is None
        assert args.name == "synth_panel"
        assert args.mcp_command_override is None
        assert args.mcp_env is None
        assert args.force is False
        assert args.dry_run is False

    def test_install_accepts_overrides(self, tmp_path):
        args = build_parser().parse_args(
            [
                "mcp",
                "install",
                "--scope",
                "project",
                "--target",
                str(tmp_path / "cfg.json"),
                "--name",
                "custom",
                "--command",
                "/usr/local/bin/synthpanel",
                "--env",
                "ANTHROPIC_API_KEY=sk-test",
                "--env",
                "OPENAI_API_KEY=sk-other",
                "--force",
                "--dry-run",
            ]
        )
        assert args.scope == "project"
        assert args.target == str(tmp_path / "cfg.json")
        assert args.name == "custom"
        assert args.mcp_command_override == "/usr/local/bin/synthpanel"
        assert args.mcp_env == ["ANTHROPIC_API_KEY=sk-test", "OPENAI_API_KEY=sk-other"]
        assert args.force is True
        assert args.dry_run is True

    def test_uninstall_flag(self):
        args = build_parser().parse_args(["mcp", "install", "--uninstall"])
        assert args.uninstall is True


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_resolve_target_user(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        # Path.home() reads HOME on POSIX; on macOS this is reliable in tests.
        target = mcp_install.resolve_target("user", None)
        assert target == Path.home() / ".claude.json"

    def test_resolve_target_project(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target = mcp_install.resolve_target("project", None)
        assert target == tmp_path / ".mcp.json"

    def test_resolve_target_explicit_overrides_scope(self, tmp_path):
        explicit = tmp_path / "weird.json"
        target = mcp_install.resolve_target("user", str(explicit))
        assert target == explicit

    def test_resolve_command_explicit(self):
        assert mcp_install.resolve_command("/opt/bin/synthpanel") == "/opt/bin/synthpanel"

    def test_resolve_command_prefers_which(self, monkeypatch, tmp_path):
        launcher = tmp_path / "synthpanel"
        launcher.write_text("#!/bin/sh\n")
        monkeypatch.setattr("synth_panel.cli.mcp_install.shutil.which", lambda _name: str(launcher))
        assert mcp_install.resolve_command(None) == str(launcher)

    def test_resolve_command_uses_venv_argv0_when_not_on_path(self, monkeypatch, tmp_path):
        # Simulate `synthpanel mcp install` invoked from a venv whose bin is
        # NOT on PATH: shutil.which finds nothing, but argv[0] is the
        # absolute path to the venv console script (#539). The host must get
        # that absolute path, not the unlaunchable literal "synthpanel".
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        launcher = venv_bin / "synthpanel"
        launcher.write_text("#!/bin/sh\n")

        monkeypatch.setattr("synth_panel.cli.mcp_install.shutil.which", lambda _name: None)
        monkeypatch.setattr("synth_panel.cli.mcp_install.sys.argv", [str(launcher)])

        assert mcp_install.resolve_command(None) == str(launcher)

    def test_resolve_command_uses_synthpanel_next_to_interpreter(self, monkeypatch, tmp_path):
        # No PATH hit and argv[0] is not a synthpanel launcher (e.g. invoked
        # as `python -m synth_panel`), but a synthpanel console script lives
        # next to sys.executable — the canonical venv layout.
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        py = venv_bin / "python"
        py.write_text("#!/bin/sh\n")
        launcher = venv_bin / "synthpanel"
        launcher.write_text("#!/bin/sh\n")

        monkeypatch.setattr("synth_panel.cli.mcp_install.shutil.which", lambda _name: None)
        monkeypatch.setattr("synth_panel.cli.mcp_install.sys.argv", ["python"])
        monkeypatch.setattr("synth_panel.cli.mcp_install.sys.executable", str(py))

        assert mcp_install.resolve_command(None) == str(launcher)

    def test_resolve_command_falls_back_to_literal(self, monkeypatch, tmp_path):
        # Nothing resolvable: no PATH hit, argv[0] is not a synthpanel
        # launcher, and no synthpanel sits beside the interpreter.
        empty_bin = tmp_path / "empty"
        empty_bin.mkdir()
        py = empty_bin / "python"
        py.write_text("#!/bin/sh\n")

        monkeypatch.setattr("synth_panel.cli.mcp_install.shutil.which", lambda _name: None)
        monkeypatch.setattr("synth_panel.cli.mcp_install.sys.argv", ["pytest"])
        monkeypatch.setattr("synth_panel.cli.mcp_install.sys.executable", str(py))

        assert mcp_install.resolve_command(None) == "synthpanel"

    def test_parse_env_pairs(self):
        out = mcp_install.parse_env_pairs(["A=1", "B=two=words"])
        assert out == {"A": "1", "B": "two=words"}

    def test_parse_env_pairs_rejects_bad_input(self):
        with pytest.raises(ValueError):
            mcp_install.parse_env_pairs(["NOEQUALS"])
        with pytest.raises(ValueError):
            mcp_install.parse_env_pairs(["=value"])

    def test_parse_env_pairs_empty(self):
        assert mcp_install.parse_env_pairs(None) == {}
        assert mcp_install.parse_env_pairs([]) == {}

    def test_mcp_extra_available_uses_import_probe(self, monkeypatch):
        monkeypatch.setattr(mcp_install, "find_spec", lambda name: object() if name == "mcp" else None)
        assert mcp_install.mcp_extra_available() is True
        monkeypatch.setattr(mcp_install, "find_spec", lambda _name: None)
        assert mcp_install.mcp_extra_available() is False

    def test_install_warns_when_mcp_extra_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mcp_install, "mcp_extra_available", lambda: False)
        result = mcp_install.install(
            target=tmp_path / "claude.json",
            name="synth_panel",
            command="synthpanel",
            env={},
            force=False,
            dry_run=True,
        )
        assert result.warnings == [mcp_install.MCP_EXTRA_WARNING]

    def test_build_entry(self):
        entry = mcp_install.build_entry("synthpanel", {})
        assert entry == {"command": "synthpanel", "args": ["mcp-serve"]}

    def test_build_entry_with_env(self):
        entry = mcp_install.build_entry("synthpanel", {"ANTHROPIC_API_KEY": "sk"})
        assert entry == {
            "command": "synthpanel",
            "args": ["mcp-serve"],
            "env": {"ANTHROPIC_API_KEY": "sk"},
        }


# ---------------------------------------------------------------------------
# install / uninstall
# ---------------------------------------------------------------------------


class TestInstall:
    def test_creates_new_config(self, tmp_path):
        target = tmp_path / "claude.json"
        result = mcp_install.install(
            target=target,
            name="synth_panel",
            command="synthpanel",
            env={},
            force=False,
            dry_run=False,
        )
        assert result.action == "installed"
        assert target.exists()
        data = json.loads(target.read_text())
        assert data == {"mcpServers": {"synth_panel": {"command": "synthpanel", "args": ["mcp-serve"]}}}

    def test_preserves_other_top_level_keys(self, tmp_path):
        target = tmp_path / "claude.json"
        target.write_text(json.dumps({"theme": "dark", "telemetry": False}))
        mcp_install.install(
            target=target,
            name="synth_panel",
            command="synthpanel",
            env={},
            force=False,
            dry_run=False,
        )
        data = json.loads(target.read_text())
        assert data["theme"] == "dark"
        assert data["telemetry"] is False
        assert "synth_panel" in data["mcpServers"]

    def test_preserves_other_servers(self, tmp_path):
        target = tmp_path / "claude.json"
        target.write_text(json.dumps({"mcpServers": {"other": {"command": "other-bin", "args": []}}}))
        mcp_install.install(
            target=target,
            name="synth_panel",
            command="synthpanel",
            env={},
            force=False,
            dry_run=False,
        )
        data = json.loads(target.read_text())
        assert "other" in data["mcpServers"]
        assert "synth_panel" in data["mcpServers"]

    def test_collision_requires_force(self, tmp_path):
        target = tmp_path / "claude.json"
        target.write_text(json.dumps({"mcpServers": {"synth_panel": {"command": "old", "args": ["mcp-serve"]}}}))
        with pytest.raises(FileExistsError):
            mcp_install.install(
                target=target,
                name="synth_panel",
                command="synthpanel",
                env={},
                force=False,
                dry_run=False,
            )
        # File should not have been touched.
        data = json.loads(target.read_text())
        assert data["mcpServers"]["synth_panel"]["command"] == "old"

    def test_force_overwrites(self, tmp_path):
        target = tmp_path / "claude.json"
        target.write_text(json.dumps({"mcpServers": {"synth_panel": {"command": "old", "args": ["mcp-serve"]}}}))
        result = mcp_install.install(
            target=target,
            name="synth_panel",
            command="synthpanel",
            env={},
            force=True,
            dry_run=False,
        )
        assert result.action == "updated"
        data = json.loads(target.read_text())
        assert data["mcpServers"]["synth_panel"]["command"] == "synthpanel"

    def test_idempotent_noop(self, tmp_path):
        target = tmp_path / "claude.json"
        mcp_install.install(
            target=target,
            name="synth_panel",
            command="synthpanel",
            env={},
            force=False,
            dry_run=False,
        )
        # Second run with identical values: no force needed because content matches.
        result = mcp_install.install(
            target=target,
            name="synth_panel",
            command="synthpanel",
            env={},
            force=False,
            dry_run=False,
        )
        assert result.action == "noop"

    def test_dry_run_does_not_write(self, tmp_path):
        target = tmp_path / "claude.json"
        result = mcp_install.install(
            target=target,
            name="synth_panel",
            command="synthpanel",
            env={},
            force=False,
            dry_run=True,
        )
        assert result.action == "would-install"
        assert not target.exists()

    def test_user_config_gets_restrictive_perms(self, tmp_path, monkeypatch):
        # Force HOME so the helper treats this file as user-owned.
        monkeypatch.setenv("HOME", str(tmp_path))
        target = tmp_path / ".claude.json"
        mcp_install.install(
            target=target,
            name="synth_panel",
            command="synthpanel",
            env={"ANTHROPIC_API_KEY": "secret"},
            force=False,
            dry_run=False,
        )
        mode = stat.S_IMODE(os.stat(target).st_mode)
        assert mode == 0o600

    def test_rejects_non_object_top_level(self, tmp_path):
        target = tmp_path / "claude.json"
        target.write_text(json.dumps(["not", "an", "object"]))
        with pytest.raises(ValueError):
            mcp_install.install(
                target=target,
                name="synth_panel",
                command="synthpanel",
                env={},
                force=True,
                dry_run=False,
            )

    def test_handles_empty_file(self, tmp_path):
        target = tmp_path / "claude.json"
        target.write_text("")
        result = mcp_install.install(
            target=target,
            name="synth_panel",
            command="synthpanel",
            env={},
            force=False,
            dry_run=False,
        )
        assert result.action == "installed"


class TestUninstall:
    def test_removes_entry(self, tmp_path):
        target = tmp_path / "claude.json"
        target.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "synth_panel": {"command": "synthpanel", "args": ["mcp-serve"]},
                        "other": {"command": "other", "args": []},
                    }
                }
            )
        )
        result = mcp_install.uninstall(target=target, name="synth_panel", dry_run=False)
        assert result.action == "removed"
        data = json.loads(target.read_text())
        assert "synth_panel" not in data["mcpServers"]
        assert "other" in data["mcpServers"]

    def test_noop_when_missing_file(self, tmp_path):
        target = tmp_path / "nope.json"
        result = mcp_install.uninstall(target=target, name="synth_panel", dry_run=False)
        assert result.action == "noop"
        assert not target.exists()

    def test_noop_when_entry_absent(self, tmp_path):
        target = tmp_path / "claude.json"
        target.write_text(json.dumps({"mcpServers": {"other": {}}}))
        result = mcp_install.uninstall(target=target, name="synth_panel", dry_run=False)
        assert result.action == "noop"

    def test_dry_run(self, tmp_path):
        target = tmp_path / "claude.json"
        target.write_text(json.dumps({"mcpServers": {"synth_panel": {"command": "synthpanel"}}}))
        result = mcp_install.uninstall(target=target, name="synth_panel", dry_run=True)
        assert result.action == "would-remove"
        # Untouched.
        data = json.loads(target.read_text())
        assert "synth_panel" in data["mcpServers"]


# ---------------------------------------------------------------------------
# End-to-end via main()
# ---------------------------------------------------------------------------


class TestCLI:
    def test_install_writes_target(self, tmp_path, capsys):
        target = tmp_path / "claude.json"
        rc = main(["mcp", "install", "--target", str(target)])
        assert rc == 0
        data = json.loads(target.read_text())
        assert "synth_panel" in data["mcpServers"]
        out = capsys.readouterr().out
        assert "Installed MCP server" in out

    def test_install_json_output(self, tmp_path, capsys):
        target = tmp_path / "claude.json"
        rc = main(
            [
                "--output-format",
                "json",
                "mcp",
                "install",
                "--target",
                str(target),
                "--command",
                "synthpanel",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["action"] == "installed"
        assert payload["name"] == "synth_panel"
        assert payload["target"] == str(target)
        assert payload["entry"]["args"] == ["mcp-serve"]

    def test_collision_exits_nonzero_without_force(self, tmp_path, capsys):
        target = tmp_path / "claude.json"
        target.write_text(json.dumps({"mcpServers": {"synth_panel": {"command": "old", "args": ["mcp-serve"]}}}))
        rc = main(["mcp", "install", "--target", str(target)])
        assert rc == 1
        err = capsys.readouterr().err
        assert "already exists" in err
        assert "--force" in err

    def test_uninstall_via_flag(self, tmp_path, capsys):
        target = tmp_path / "claude.json"
        target.write_text(json.dumps({"mcpServers": {"synth_panel": {"command": "synthpanel"}}}))
        rc = main(["mcp", "install", "--uninstall", "--target", str(target)])
        assert rc == 0
        data = json.loads(target.read_text())
        assert "synth_panel" not in data.get("mcpServers", {})

    def test_dry_run_does_not_write(self, tmp_path, capsys):
        # sy-0k2 / gh-495: dry-run human prose goes to stderr; stdout
        # carries the generated config as pretty JSON.
        target = tmp_path / "claude.json"
        rc = main(["mcp", "install", "--target", str(target), "--dry-run"])
        assert rc == 0
        assert not target.exists()
        captured = capsys.readouterr()
        assert "Would install" in captured.err
        # stdout must be valid JSON so callers can `| jq` or write-it-out themselves.
        payload = json.loads(captured.out)
        assert payload["mcpServers"]["synth_panel"]["command"]
        assert payload["mcpServers"]["synth_panel"]["args"] == ["mcp-serve"]

    def test_dry_run_preserves_existing_servers_in_preview(self, tmp_path, capsys):
        # sy-0k2 / gh-495: the dry-run preview must show ALL servers that
        # would be in the file, not just the one being added.
        target = tmp_path / "claude.json"
        target.write_text(
            json.dumps(
                {
                    "theme": "dark",
                    "mcpServers": {"other": {"command": "other-bin", "args": []}},
                }
            )
        )
        rc = main(["mcp", "install", "--target", str(target), "--dry-run"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["theme"] == "dark"
        assert set(payload["mcpServers"].keys()) == {"other", "synth_panel"}

    def test_dry_run_uninstall_preview_drops_entry(self, tmp_path, capsys):
        # sy-0k2 / gh-495: uninstall dry-run preview shows the post-removal
        # state so callers can verify what would disappear.
        target = tmp_path / "claude.json"
        target.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "synth_panel": {"command": "synthpanel", "args": ["mcp-serve"]},
                        "other": {"command": "other-bin", "args": []},
                    }
                }
            )
        )
        rc = main(["mcp", "install", "--uninstall", "--target", str(target), "--dry-run"])
        assert rc == 0
        # Original file untouched.
        on_disk = json.loads(target.read_text())
        assert "synth_panel" in on_disk["mcpServers"]
        # Preview reflects the post-removal state.
        captured = capsys.readouterr()
        assert "Would remove" in captured.err
        preview = json.loads(captured.out)
        assert "synth_panel" not in preview["mcpServers"]
        assert "other" in preview["mcpServers"]

    def test_dry_run_json_mode_includes_resulting_config(self, tmp_path, capsys):
        # sy-0k2 / gh-495: in JSON mode the entire payload (line, entry,
        # resulting_config) lands on stdout as a single object so callers
        # can parse one stream.
        target = tmp_path / "claude.json"
        rc = main(["--output-format", "json", "mcp", "install", "--target", str(target), "--dry-run"])
        assert rc == 0
        captured = capsys.readouterr()
        # Nothing should land on stderr in JSON mode — the contract is one stream.
        assert captured.err == ""
        payload = json.loads(captured.out)
        assert payload["action"] == "would-install"
        assert payload["entry"]["args"] == ["mcp-serve"]
        assert "synth_panel" in payload["resulting_config"]["mcpServers"]

    def test_bad_env_exits_nonzero(self, tmp_path, capsys):
        target = tmp_path / "claude.json"
        rc = main(["mcp", "install", "--target", str(target), "--env", "NOEQUALS"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "--env" in err

    def test_mcp_serve_missing_extra_is_actionable(self, capsys, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "synth_panel.mcp.server":
                raise ModuleNotFoundError("No module named 'mcp'", name="mcp")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        rc = main(["mcp-serve"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "synthpanel[mcp]" in err
        assert "ModuleNotFoundError" not in err

    def test_no_subcommand_prints_help(self, capsys):
        with pytest.raises(SystemExit):
            main(["mcp"])


# ---------------------------------------------------------------------------
# sy-xyn: mcp extra guard
# ---------------------------------------------------------------------------


class TestMcpExtraGuard:
    """Pin the contract that a base install can't silently generate a
    broken MCP config (GH #507).

    The probe ``mcp_install.mcp_extra_available()`` returns True in the
    pytest env (we install with ``pip install -e .[dev,mcp]``). To
    exercise the missing-extra branch we monkeypatch the probe to return
    False — this is the same shape the CI smoke job sees on a fresh
    ``pip install synthpanel`` (no extras).
    """

    def test_install_refuses_when_extra_missing(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(mcp_install, "mcp_extra_available", lambda: False)
        target = tmp_path / "claude.json"

        rc = main(["mcp", "install", "--target", str(target)])

        assert rc == 1, "must refuse rather than silently write a broken config"
        assert not target.exists(), "no file should be written when refusing"
        err = capsys.readouterr().err
        # Actionable copy is the whole point — the user needs a one-line fix.
        assert "synthpanel[mcp]" in err
        # And the corrective opt-out for the cross-machine case must be discoverable.
        assert "--allow-missing-extra" in err

    def test_install_allows_when_flag_set(self, tmp_path, monkeypatch):
        """Cross-machine workflow: config on laptop, server on remote."""
        monkeypatch.setattr(mcp_install, "mcp_extra_available", lambda: False)
        target = tmp_path / "claude.json"

        rc = main(["mcp", "install", "--target", str(target), "--allow-missing-extra"])

        assert rc == 0
        data = json.loads(target.read_text())
        assert "synth_panel" in data["mcpServers"]

    def test_uninstall_always_allowed(self, tmp_path, monkeypatch):
        """Uninstall must work even when the extra is gone — it's the
        canonical way to clean up after a partial install."""
        monkeypatch.setattr(mcp_install, "mcp_extra_available", lambda: False)
        target = tmp_path / "claude.json"
        target.write_text(json.dumps({"mcpServers": {"synth_panel": {"command": "synthpanel"}}}))

        rc = main(["mcp", "install", "--uninstall", "--target", str(target)])

        assert rc == 0
        data = json.loads(target.read_text())
        assert "synth_panel" not in data.get("mcpServers", {})

    def test_dry_run_also_blocked_without_flag(self, tmp_path, capsys, monkeypatch):
        """Dry-run with a missing extra still emits a broken config that
        an agent could naively pipe into the host's config file. Same
        refusal as the live path."""
        monkeypatch.setattr(mcp_install, "mcp_extra_available", lambda: False)
        target = tmp_path / "claude.json"

        rc = main(["mcp", "install", "--target", str(target), "--dry-run"])

        assert rc == 1
        err = capsys.readouterr().err
        assert "synthpanel[mcp]" in err

    def test_serve_emits_actionable_error_when_extra_missing(self, capsys, monkeypatch):
        """`mcp-serve` must NOT produce a Python traceback. Editor hosts
        only surface the launch command's stderr, so the message has to
        carry the install command itself.

        sy-xyn origin: GH #507 user saw a raw ModuleNotFoundError after
        their editor launched ``synthpanel mcp-serve``.
        """
        monkeypatch.setattr(mcp_install, "mcp_extra_available", lambda: False)

        rc = main(["mcp-serve"])

        assert rc == 1
        err = capsys.readouterr().err
        assert "synthpanel[mcp]" in err
        # Sanity: the traceback prefix isn't in our stderr — that's the
        # observable difference from the v1.5.1 behaviour.
        assert "Traceback" not in err
        assert "ModuleNotFoundError" not in err

    def test_mcp_extra_available_is_truthy_in_dev_env(self):
        """Sanity check on the probe itself — the dev env installs the
        extra, so the probe must return True. If this fails, every other
        test in this file would silently be exercising the refuse branch."""
        assert mcp_install.mcp_extra_available() is True

    def test_message_is_centralised(self):
        """One copy, one source of truth — anything that surfaces the
        missing-extra hint reads from the same constant."""
        assert "synthpanel[mcp]" in mcp_install.MISSING_MCP_EXTRA_MESSAGE
        assert "pip install" in mcp_install.MISSING_MCP_EXTRA_MESSAGE


# ---------------------------------------------------------------------------
# Named hosts + `mcp uninstall` subcommand (synthbench#262)
# ---------------------------------------------------------------------------


class TestHostRegistry:
    def test_host_config_paths_user_scope(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        home = Path.home()
        assert mcp_install.host_config_path(mcp_install.HOSTS["claude-code"]) == home / ".claude.json"
        assert mcp_install.host_config_path(mcp_install.HOSTS["cursor"]) == home / ".cursor" / "mcp.json"
        assert (
            mcp_install.host_config_path(mcp_install.HOSTS["windsurf"])
            == home / ".codeium" / "windsurf" / "mcp_config.json"
        )
        assert mcp_install.host_config_path(mcp_install.HOSTS["zed"]) == home / ".config" / "zed" / "settings.json"
        desktop = mcp_install.host_config_path(mcp_install.HOSTS["claude-desktop"])
        assert desktop.name == "claude_desktop_config.json"

    def test_project_scope_paths(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        assert mcp_install.host_config_path(mcp_install.HOSTS["claude-code"], "project") == tmp_path / ".mcp.json"
        assert mcp_install.host_config_path(mcp_install.HOSTS["cursor"], "project") == tmp_path / ".cursor" / "mcp.json"

    def test_project_scope_rejected_for_user_only_hosts(self):
        for key in ("claude-desktop", "windsurf", "zed"):
            with pytest.raises(ValueError):
                mcp_install.host_config_path(mcp_install.HOSTS[key], "project")

    def test_zed_uses_context_servers_schema(self):
        zed = mcp_install.HOSTS["zed"]
        assert zed.config_key == "context_servers"
        assert dict(zed.entry_extra) == {"source": "custom"}

    def test_detect_hosts_reports_only_existing_configs(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("APPDATA", raising=False)
        assert mcp_install.detect_hosts() == []

        cursor_cfg = tmp_path / ".cursor" / "mcp.json"
        cursor_cfg.parent.mkdir(parents=True)
        cursor_cfg.write_text("{}")
        zed_cfg = tmp_path / ".config" / "zed" / "settings.json"
        zed_cfg.parent.mkdir(parents=True)
        zed_cfg.write_text("{}")

        detected = mcp_install.detect_hosts()
        keys = [h.key for h, _ in detected]
        assert keys == ["cursor", "zed"]
        assert [p for _, p in detected] == [cursor_cfg, zed_cfg]


class TestHostCLI:
    def test_install_host_cursor_writes_user_config(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HOME", str(tmp_path))
        rc = main(["mcp", "install", "--host", "cursor", "--command", "synthpanel"])
        assert rc == 0
        data = json.loads((tmp_path / ".cursor" / "mcp.json").read_text())
        assert data["mcpServers"]["synth_panel"] == {
            "command": "synthpanel",
            "args": ["mcp-serve"],
        }
        out = capsys.readouterr().out
        assert "Restart Cursor to pick up the server." in out

    def test_install_host_windsurf_writes_config(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        rc = main(["mcp", "install", "--host", "windsurf", "--command", "synthpanel"])
        assert rc == 0
        data = json.loads((tmp_path / ".codeium" / "windsurf" / "mcp_config.json").read_text())
        assert data["mcpServers"]["synth_panel"]["args"] == ["mcp-serve"]

    def test_install_host_claude_desktop_writes_config(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        rc = main(["mcp", "install", "--host", "claude-desktop", "--command", "synthpanel"])
        assert rc == 0
        target = mcp_install.host_config_path(mcp_install.HOSTS["claude-desktop"])
        data = json.loads(target.read_text())
        assert data["mcpServers"]["synth_panel"]["command"] == "synthpanel"

    def test_install_host_zed_writes_context_servers(self, monkeypatch, tmp_path, capsys):
        """Zed's schema differs: context_servers + source: custom."""
        monkeypatch.setenv("HOME", str(tmp_path))
        settings = tmp_path / ".config" / "zed" / "settings.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps({"theme": "One Dark", "context_servers": {"other": {"command": "x"}}}))

        rc = main(["mcp", "install", "--host", "zed", "--command", "synthpanel"])
        assert rc == 0
        data = json.loads(settings.read_text())
        # Non-destructive merge: unrelated settings and other servers survive.
        assert data["theme"] == "One Dark"
        assert "other" in data["context_servers"]
        assert data["context_servers"]["synth_panel"] == {
            "source": "custom",
            "command": "synthpanel",
            "args": ["mcp-serve"],
        }
        assert "mcpServers" not in data
        assert "Restart Zed to pick up the server." in capsys.readouterr().out

    def test_install_host_project_scope_cursor(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        rc = main(["mcp", "install", "--host", "cursor", "--scope", "project", "--command", "synthpanel"])
        assert rc == 0
        assert (tmp_path / ".cursor" / "mcp.json").is_file()

    def test_install_host_project_scope_rejected_for_zed(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HOME", str(tmp_path))
        rc = main(["mcp", "install", "--host", "zed", "--scope", "project"])
        assert rc == 1
        assert "no project-scope config" in capsys.readouterr().err

    def test_install_without_env_prints_key_note(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HOME", str(tmp_path))
        rc = main(["mcp", "install", "--host", "cursor", "--command", "synthpanel"])
        assert rc == 0
        err = capsys.readouterr().err
        assert "No API key was written" in err
        assert "synthpanel login" in err

    def test_install_with_env_skips_key_note(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HOME", str(tmp_path))
        rc = main(
            [
                "mcp",
                "install",
                "--host",
                "cursor",
                "--command",
                "synthpanel",
                "--env",
                "ANTHROPIC_API_KEY=sk-test",
            ]
        )
        assert rc == 0
        assert "No API key was written" not in capsys.readouterr().err

    def test_install_host_dry_run_writes_nothing(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HOME", str(tmp_path))
        rc = main(["mcp", "install", "--host", "zed", "--dry-run", "--command", "synthpanel"])
        assert rc == 0
        assert not (tmp_path / ".config" / "zed" / "settings.json").exists()
        captured = capsys.readouterr()
        assert "Would install" in captured.err
        payload = json.loads(captured.out)
        assert payload["context_servers"]["synth_panel"]["source"] == "custom"
        # Dry-run never claims a restart is needed.
        assert "Restart" not in captured.out


class TestUninstallSubcommand:
    def test_parser_registers_uninstall(self):
        args = build_parser().parse_args(["mcp", "uninstall", "--host", "zed"])
        assert args.command == "mcp"
        assert args.mcp_command == "uninstall"
        assert args.host == "zed"
        assert args.name == "synth_panel"
        assert args.dry_run is False

    def test_uninstall_removes_only_our_entry(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HOME", str(tmp_path))
        cfg = tmp_path / ".cursor" / "mcp.json"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "other": {"command": "other-server"},
                        "synth_panel": {"command": "synthpanel", "args": ["mcp-serve"]},
                    },
                    "unrelated": True,
                }
            )
        )
        rc = main(["mcp", "uninstall", "--host", "cursor"])
        assert rc == 0
        data = json.loads(cfg.read_text())
        assert "synth_panel" not in data["mcpServers"]
        assert data["mcpServers"]["other"] == {"command": "other-server"}
        assert data["unrelated"] is True
        assert "Restart Cursor to pick up the server." in capsys.readouterr().out

    def test_uninstall_zed_uses_context_servers(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        cfg = tmp_path / ".config" / "zed" / "settings.json"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(
            json.dumps(
                {
                    "context_servers": {
                        "synth_panel": {"source": "custom", "command": "synthpanel", "args": ["mcp-serve"]},
                        "keep": {"source": "custom", "command": "keep"},
                    }
                }
            )
        )
        rc = main(["mcp", "uninstall", "--host", "zed"])
        assert rc == 0
        data = json.loads(cfg.read_text())
        assert "synth_panel" not in data["context_servers"]
        assert "keep" in data["context_servers"]

    def test_uninstall_noop_when_absent(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HOME", str(tmp_path))
        rc = main(["mcp", "uninstall", "--host", "cursor"])
        assert rc == 0
        assert "No changes needed" in capsys.readouterr().out

    def test_uninstall_works_without_mcp_extra(self, monkeypatch, tmp_path):
        """Cleanup must never be blocked by a missing optional dep."""
        monkeypatch.setattr(mcp_install, "mcp_extra_available", lambda: False)
        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps({"mcpServers": {"synth_panel": {"command": "synthpanel"}}}))
        rc = main(["mcp", "uninstall", "--target", str(cfg)])
        assert rc == 0
        assert "synth_panel" not in json.loads(cfg.read_text()).get("mcpServers", {})


class TestAutoDetect:
    def _seed_hosts(self, home: Path) -> tuple[Path, Path]:
        cursor_cfg = home / ".cursor" / "mcp.json"
        cursor_cfg.parent.mkdir(parents=True)
        cursor_cfg.write_text("{}")
        zed_cfg = home / ".config" / "zed" / "settings.json"
        zed_cfg.parent.mkdir(parents=True)
        zed_cfg.write_text("{}")
        return cursor_cfg, zed_cfg

    def test_auto_with_yes_installs_into_all_detected(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("APPDATA", raising=False)
        cursor_cfg, zed_cfg = self._seed_hosts(tmp_path)

        rc = main(["mcp", "install", "--host", "auto", "--yes", "--command", "synthpanel"])
        assert rc == 0
        assert "synth_panel" in json.loads(cursor_cfg.read_text())["mcpServers"]
        assert json.loads(zed_cfg.read_text())["context_servers"]["synth_panel"]["source"] == "custom"
        captured = capsys.readouterr()
        assert "Detected MCP host configs" in captured.err
        assert "Restart Cursor to pick up the server." in captured.out
        assert "Restart Zed to pick up the server." in captured.out

    def test_auto_without_yes_non_tty_errors(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("APPDATA", raising=False)
        self._seed_hosts(tmp_path)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        rc = main(["mcp", "install", "--host", "auto", "--command", "synthpanel"])
        assert rc == 1
        assert "--yes" in capsys.readouterr().err

    def test_auto_prompts_and_honors_answers(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("APPDATA", raising=False)
        cursor_cfg, zed_cfg = self._seed_hosts(tmp_path)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        answers = iter(["y", "n"])
        monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))

        rc = main(["mcp", "install", "--host", "auto", "--command", "synthpanel"])
        assert rc == 0
        assert "synth_panel" in json.loads(cursor_cfg.read_text())["mcpServers"]
        assert "context_servers" not in json.loads(zed_cfg.read_text())

    def test_auto_with_no_hosts_detected_errors(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("APPDATA", raising=False)
        rc = main(["mcp", "install", "--host", "auto", "--yes"])
        assert rc == 1
        assert "no known MCP host configs detected" in capsys.readouterr().err
