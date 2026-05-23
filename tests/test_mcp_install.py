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

    def test_resolve_command_falls_back_to_literal(self, monkeypatch):
        monkeypatch.setattr("synth_panel.cli.mcp_install.shutil.which", lambda _name: None)
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
