"""Slash command registry and dispatch for the interactive REPL.

Commands are registered in SLASH_COMMANDS. New commands only need a new entry.
Per SPEC.md §8.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from synth_panel.cli.output import OutputFormat, emit

if TYPE_CHECKING:
    from synth_panel.cli.repl import SessionState


# Type alias for slash command handlers
SlashHandler = Callable[["SessionState", list[str], OutputFormat], None]


def _cmd_help(state: SessionState, argv: list[str], fmt: OutputFormat) -> None:
    """List available slash commands."""
    lines = ["Available commands:"]
    for name, (_, summary) in sorted(SLASH_COMMANDS.items()):
        lines.append(f"  /{name:<14s} {summary}")
    emit(fmt, message="\n".join(lines))


def _cmd_status(state: SessionState, argv: list[str], fmt: OutputFormat) -> None:
    """Show current session state."""
    parts = [
        f"Turn count: {state.turn_count}",
        f"Compacted: {state.compacted_count}",
        f"Model: {state.model or '(default)'}",
    ]
    if state.last_usage:
        parts.append(
            f"Last usage: input={state.last_usage.get('input_tokens', 0)} "
            f"output={state.last_usage.get('output_tokens', 0)}"
        )
    emit(fmt, message="\n".join(parts))


def _cmd_compact(state: SessionState, argv: list[str], fmt: OutputFormat) -> None:
    """Compact session history."""
    if state.runtime is None:
        emit(fmt, message="No active runtime session to compact.")
        return
    session = state.runtime.session
    if len(session.messages) <= 2:
        emit(fmt, message="Not enough messages to compact.")
        return
    # Build summary from older messages, keeping last 2
    older = session.messages[:-2]
    summary_parts: list[str] = []
    for msg in older:
        for block in msg.content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    summary_parts.append(f"[{msg.role}]: {text[:200]}")
    summary_text = "Compacted conversation summary:\n" + "\n".join(summary_parts[:20])
    session.compact(summary_text, keep_last=2)
    state.compacted_count += 1
    emit(fmt, message=f"Session compacted. {len(session.messages)} messages remaining.")


def _cmd_model(state: SessionState, argv: list[str], fmt: OutputFormat) -> None:
    """Show or switch the active model."""
    if argv:
        state.model = argv[0]
        emit(fmt, message=f"Model set to: {state.model}")
    else:
        emit(fmt, message=f"Current model: {state.model or '(default)'}")


def _cmd_permissions(state: SessionState, argv: list[str], fmt: OutputFormat) -> None:
    """Show or switch the permission mode."""
    # TODO: wire to permission state
    if argv:
        emit(fmt, message=f"[stub] Permission mode set to: {argv[0]}")
    else:
        emit(fmt, message="[stub] Current permission mode: full-access")


def _cmd_config(state: SessionState, argv: list[str], fmt: OutputFormat) -> None:
    """Display active profile and available profiles."""
    from synth_panel.profiles import list_available_profiles

    lines: list[str] = []

    # Show active profile if set
    active_profile = getattr(state, "profile", None)
    if active_profile is not None:
        lines.append(f"Active profile: {active_profile.name}")
        lines.append(f"  Source: {active_profile.source_path or 'unknown'}")
        lines.append(f"  Hash:   {active_profile.config_hash()}")
        profile_dict = active_profile.to_dict()
        for key, val in profile_dict.items():
            if key == "name":
                continue
            lines.append(f"  {key}: {val}")
    else:
        lines.append("No active profile (using CLI defaults)")

    # Show overrides
    overrides = getattr(state, "profile_overrides", None)
    if overrides:
        lines.append("\nCLI overrides applied on top of profile:")
        for key, val in overrides.items():
            lines.append(f"  {key}: {val}")

    # List available profiles
    available = list_available_profiles()
    if available:
        lines.append("\nAvailable profiles:")
        for p in available:
            marker = " *" if active_profile and p["name"] == active_profile.name else ""
            lines.append(f"  {p['name']:<16s} ({p['source']}){marker}")

    emit(fmt, message="\n".join(lines))


def _cmd_memory(state: SessionState, argv: list[str], fmt: OutputFormat) -> None:
    """Show loaded instruction/memory files."""
    # TODO: wire to memory system
    emit(fmt, message="[stub] No instruction/memory files loaded.")


def _cmd_clear(state: SessionState, argv: list[str], fmt: OutputFormat) -> None:
    """Start a fresh session (requires --confirm)."""
    if "--confirm" not in argv:
        emit(fmt, message="Use /clear --confirm to start a fresh session.")
        return
    state.turn_count = 0
    state.compacted_count = 0
    state.last_usage = None
    emit(fmt, message="Session cleared.")


# Registry: name → (handler, summary)
SLASH_COMMANDS: dict[str, tuple[SlashHandler, str]] = {
    "help": (_cmd_help, "List available commands"),
    "status": (_cmd_status, "Show current session state"),
    "compact": (_cmd_compact, "Compact session history"),
    "model": (_cmd_model, "Show or switch the active model"),
    "permissions": (_cmd_permissions, "Show or switch permission mode"),
    "config": (_cmd_config, "Inspect configuration"),
    "memory": (_cmd_memory, "Show loaded instruction/memory files"),
    "clear": (_cmd_clear, "Start a fresh session (--confirm required)"),
}


def dispatch_slash(line: str, state: SessionState, fmt: OutputFormat) -> None:
    """Parse and dispatch a slash command line."""
    parts = line.lstrip("/").split()
    if not parts:
        return

    cmd_name = parts[0]
    argv = parts[1:]

    entry = SLASH_COMMANDS.get(cmd_name)
    if entry is None:
        emit(fmt, message=f"Unknown command: /{cmd_name}. Type /help for a list.")
        return

    handler, _ = entry
    handler(state, argv, fmt)
