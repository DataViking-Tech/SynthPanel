"""Subcommand handlers for synth-panel CLI.

Each handler receives parsed args and output format, returns an exit code.
"""

from __future__ import annotations

import argparse

from synth_panel.cli.output import OutputFormat, emit


def handle_prompt(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Run a single non-interactive prompt and exit."""
    prompt_text = " ".join(args.text)
    # TODO: wire to agent runtime when available
    emit(fmt, message=f"[stub] Would run prompt: {prompt_text}")
    return 0


def handle_login(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Start an authentication flow."""
    # TODO: implement OAuth flow
    emit(fmt, message="[stub] Login not yet implemented.")
    return 0


def handle_logout(args: argparse.Namespace, fmt: OutputFormat) -> int:
    """Clear saved authentication credentials."""
    # TODO: implement credential clearing
    emit(fmt, message="[stub] Logout not yet implemented.")
    return 0
