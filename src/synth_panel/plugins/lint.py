"""Plugin manifest linter (sy-0rr).

Author-time validation for synthpanel plugins. Surfaces structural
problems in ``plugin.yaml`` before the plugin is installed and loaded
at runtime, where the same problems would otherwise present as opaque
load failures.

The linter validates the on-disk shape that ``PluginManifest.from_file``
expects, plus the keys (`pre_tool_use`, `post_tool_use`,
`post_tool_use_failure`) and lifecycle commands the rest of the plugin
system actually consumes. Hook *commands* are shell strings, not Python
callables, so the linter only validates their type and emptiness — not
their semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from synth_panel.plugins.manifest import MANIFEST_FILENAME

VALID_HOOK_NAMES = ("pre_tool_use", "post_tool_use", "post_tool_use_failure")
VALID_LIFECYCLE_NAMES = ("init", "shutdown")
REQUIRED_FIELDS = ("name", "version")


@dataclass
class LintIssue:
    """A single lint finding against a plugin manifest."""

    severity: str  # "error" | "warning"
    code: str
    message: str

    def format(self) -> str:
        marker = "✗" if self.severity == "error" else "!"
        return f"{marker} [{self.code}] {self.message}"


@dataclass
class LintReport:
    """Result of linting one plugin directory."""

    plugin_path: str
    plugin_name: str | None
    issues: list[LintIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[LintIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[LintIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def passed(self, *, strict: bool = False) -> bool:
        if self.errors:
            return False
        return not (strict and self.warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin_path": self.plugin_path,
            "plugin_name": self.plugin_name,
            "errors": [{"code": i.code, "message": i.message} for i in self.errors],
            "warnings": [{"code": i.code, "message": i.message} for i in self.warnings],
        }


def lint_plugin(path: str | Path) -> LintReport:
    """Lint a plugin source directory and return a structured report.

    The path may be a directory containing ``plugin.yaml`` or a direct
    path to the manifest file itself. The function never raises for
    malformed manifests — every problem becomes a `LintIssue` so callers
    can render a complete report in one pass.
    """
    raw_path = Path(path)
    report = LintReport(plugin_path=str(raw_path), plugin_name=None)

    manifest_path = _resolve_manifest_path(raw_path, report)
    if manifest_path is None:
        return report

    data = _load_manifest_yaml(manifest_path, report)
    if data is None:
        return report

    if not isinstance(data, dict):
        report.issues.append(
            LintIssue(
                severity="error",
                code="manifest-shape",
                message=(f"plugin.yaml must be a YAML mapping, got {type(data).__name__}"),
            )
        )
        return report

    name = data.get("name")
    if isinstance(name, str) and name:
        report.plugin_name = name

    _check_required_fields(data, report)
    _check_scalar_types(data, report)
    _check_permissions(data, report)
    _check_hooks(data, report)
    _check_lifecycle(data, report)
    _check_tools_commands(data, report)

    return report


def _resolve_manifest_path(path: Path, report: LintReport) -> Path | None:
    if not path.exists():
        report.issues.append(
            LintIssue(
                severity="error",
                code="path-missing",
                message=f"plugin path does not exist: {path}",
            )
        )
        return None

    if path.is_file():
        return path

    manifest = path / MANIFEST_FILENAME
    if not manifest.exists():
        report.issues.append(
            LintIssue(
                severity="error",
                code="manifest-missing",
                message=f"{MANIFEST_FILENAME} not found in {path}",
            )
        )
        return None
    return manifest


def _load_manifest_yaml(manifest_path: Path, report: LintReport) -> Any:
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        report.issues.append(
            LintIssue(
                severity="error",
                code="manifest-unreadable",
                message=f"cannot read {manifest_path}: {exc}",
            )
        )
        return None

    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        report.issues.append(
            LintIssue(
                severity="error",
                code="manifest-yaml",
                message=f"plugin.yaml is not valid YAML: {exc}",
            )
        )
        return None


def _check_required_fields(data: dict[str, Any], report: LintReport) -> None:
    for field_name in REQUIRED_FIELDS:
        value = data.get(field_name)
        if value is None or value == "":
            report.issues.append(
                LintIssue(
                    severity="error",
                    code="missing-field",
                    message=f"required field '{field_name}' is missing or empty",
                )
            )
        elif not isinstance(value, str):
            report.issues.append(
                LintIssue(
                    severity="error",
                    code="field-type",
                    message=(f"field '{field_name}' must be a string, got {type(value).__name__}"),
                )
            )


def _check_scalar_types(data: dict[str, Any], report: LintReport) -> None:
    description = data.get("description", "")
    if not isinstance(description, str):
        report.issues.append(
            LintIssue(
                severity="error",
                code="field-type",
                message=(f"field 'description' must be a string, got {type(description).__name__}"),
            )
        )

    if "default_enabled" in data and not isinstance(data["default_enabled"], bool):
        report.issues.append(
            LintIssue(
                severity="error",
                code="field-type",
                message=(f"field 'default_enabled' must be a boolean, got {type(data['default_enabled']).__name__}"),
            )
        )


def _check_permissions(data: dict[str, Any], report: LintReport) -> None:
    if "permissions" not in data:
        return
    perms = data["permissions"]
    if not isinstance(perms, list):
        report.issues.append(
            LintIssue(
                severity="error",
                code="field-type",
                message=(f"field 'permissions' must be a list of strings, got {type(perms).__name__}"),
            )
        )
        return
    for i, p in enumerate(perms):
        if not isinstance(p, str):
            report.issues.append(
                LintIssue(
                    severity="error",
                    code="field-type",
                    message=(f"permissions[{i}] must be a string, got {type(p).__name__}"),
                )
            )


def _check_hooks(data: dict[str, Any], report: LintReport) -> None:
    if "hooks" not in data:
        return
    hooks = data["hooks"]
    if not isinstance(hooks, dict):
        report.issues.append(
            LintIssue(
                severity="error",
                code="field-type",
                message=(f"field 'hooks' must be a mapping, got {type(hooks).__name__}"),
            )
        )
        return

    for key, value in hooks.items():
        if key not in VALID_HOOK_NAMES:
            report.issues.append(
                LintIssue(
                    severity="error",
                    code="unknown-hook",
                    message=(f"unknown hook name '{key}' (valid: {', '.join(VALID_HOOK_NAMES)})"),
                )
            )
            continue
        _check_command_list(f"hooks.{key}", value, report)


def _check_lifecycle(data: dict[str, Any], report: LintReport) -> None:
    if "lifecycle" not in data:
        return
    lifecycle = data["lifecycle"]
    if not isinstance(lifecycle, dict):
        report.issues.append(
            LintIssue(
                severity="error",
                code="field-type",
                message=(f"field 'lifecycle' must be a mapping, got {type(lifecycle).__name__}"),
            )
        )
        return

    for key, value in lifecycle.items():
        if key not in VALID_LIFECYCLE_NAMES:
            report.issues.append(
                LintIssue(
                    severity="error",
                    code="unknown-lifecycle",
                    message=(f"unknown lifecycle stage '{key}' (valid: {', '.join(VALID_LIFECYCLE_NAMES)})"),
                )
            )
            continue
        _check_command_list(f"lifecycle.{key}", value, report)


def _check_command_list(label: str, value: Any, report: LintReport) -> None:
    if not isinstance(value, list):
        report.issues.append(
            LintIssue(
                severity="error",
                code="field-type",
                message=(f"{label} must be a list of shell command strings, got {type(value).__name__}"),
            )
        )
        return

    if not value:
        report.issues.append(
            LintIssue(
                severity="warning",
                code="empty-hook",
                message=f"{label} is an empty list — declare it only if you have commands",
            )
        )
        return

    for i, cmd in enumerate(value):
        if not isinstance(cmd, str):
            report.issues.append(
                LintIssue(
                    severity="error",
                    code="field-type",
                    message=(f"{label}[{i}] must be a string, got {type(cmd).__name__}"),
                )
            )
        elif not cmd.strip():
            report.issues.append(
                LintIssue(
                    severity="error",
                    code="empty-command",
                    message=f"{label}[{i}] is an empty command",
                )
            )


def _check_tools_commands(data: dict[str, Any], report: LintReport) -> None:
    for key in ("tools", "commands"):
        if key not in data:
            continue
        value = data[key]
        if not isinstance(value, list):
            report.issues.append(
                LintIssue(
                    severity="error",
                    code="field-type",
                    message=(f"field '{key}' must be a list of mappings, got {type(value).__name__}"),
                )
            )
            continue
        for i, entry in enumerate(value):
            if not isinstance(entry, dict):
                report.issues.append(
                    LintIssue(
                        severity="error",
                        code="field-type",
                        message=(f"{key}[{i}] must be a mapping, got {type(entry).__name__}"),
                    )
                )
