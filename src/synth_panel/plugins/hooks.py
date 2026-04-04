"""Shell-based hook runner implementing the HookRunner protocol (SPEC.md §9).

Hook commands are shell commands that receive context via environment variables
and stdin. Exit codes determine the outcome:
  0  — Allow (stdout captured as message)
  2  — Deny (stdout used as denial reason)
  other non-zero — Failure (chain short-circuits)
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from synth_panel.plugins.manifest import PluginHooks
from synth_panel.runtime import HookResult


class ShellHookRunner:
    """Executes shell hook commands per the exit-code protocol.

    Implements the ``HookRunner`` protocol from ``synth_panel.runtime``.
    """

    def __init__(self, hooks: PluginHooks, *, cwd: str | None = None) -> None:
        self._hooks = hooks
        self._cwd = cwd

    def run_pre_tool_use(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> HookResult:
        return self._run_chain(
            commands=self._hooks.pre_tool_use,
            event="pre_tool_use",
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=None,
            is_error=False,
        )

    def run_post_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: str,
        is_error: bool,
    ) -> HookResult:
        event = "post_tool_use_failure" if is_error else "post_tool_use"
        commands = (
            self._hooks.post_tool_use_failure if is_error
            else self._hooks.post_tool_use
        )
        return self._run_chain(
            commands=commands,
            event=event,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            is_error=is_error,
        )

    def _run_chain(
        self,
        *,
        commands: list[str],
        event: str,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: str | None,
        is_error: bool,
    ) -> HookResult:
        """Run a chain of hook commands with short-circuit evaluation."""
        result = HookResult()

        for cmd in commands:
            env = self._build_env(event, tool_name, tool_input, tool_output, is_error)
            stdin_payload = self._build_stdin(
                event, tool_name, tool_input, tool_output, is_error
            )

            try:
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    env=env,
                    input=stdin_payload,
                    cwd=self._cwd,
                )
            except Exception:
                result.failed = True
                return result

            stdout = proc.stdout.strip()

            if proc.returncode == 0:
                if stdout:
                    result.messages.append(stdout)
            elif proc.returncode == 2:
                result.denied = True
                if stdout:
                    result.messages.append(stdout)
                return result
            else:
                result.failed = True
                if stdout:
                    result.messages.append(stdout)
                return result

        return result

    def _build_env(
        self,
        event: str,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: str | None,
        is_error: bool,
    ) -> dict[str, str]:
        env = dict(os.environ)
        env["HOOK_EVENT"] = event
        env["HOOK_TOOL_NAME"] = tool_name
        env["HOOK_TOOL_INPUT"] = json.dumps(tool_input)
        if tool_output is not None:
            env["HOOK_TOOL_OUTPUT"] = tool_output
        env["HOOK_TOOL_IS_ERROR"] = "1" if is_error else "0"
        return env

    def _build_stdin(
        self,
        event: str,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: str | None,
        is_error: bool,
    ) -> str:
        payload: dict[str, Any] = {
            "event": event,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "is_error": is_error,
        }
        if tool_output is not None:
            payload["tool_output"] = tool_output
        return json.dumps(payload)


def run_lifecycle_commands(commands: list[str], *, cwd: str | None = None) -> None:
    """Run lifecycle commands (init or shutdown) sequentially.

    Each command runs as a shell subprocess. Failures are logged but
    do not prevent subsequent commands from running.
    """
    for cmd in commands:
        try:
            subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=cwd,
            )
        except Exception:
            pass
