# API Stability Policy

synth-panel is currently **pre-1.0** (0.x.y). This document describes what contributors and users can expect about breaking changes.

## Pre-1.0 stance

- Breaking changes are possible.
- **Minor bumps** (0.x → 0.(x+1)) may include breaking changes. The CHANGELOG documents all breakage.
- **Patch bumps** (0.x.y → 0.x.(y+1)) are bug-fix-only. Never breaking.
- Use the CHANGELOG as your source of truth before upgrading.

## 1.0 promise

1.0 will require a separate announcement and commitment to strict semver (breaking changes = major bump only). We're not there yet.

## Public API surface

The following are considered **public** and we try hard not to break them:

- `synth_panel.llm.providers.base.LLMProvider` — the adapter base class
- `ProviderConfig`, `CompletionRequest`, `CompletionResponse`, `StreamEvent` — adapter contract types
- `synth-panel` CLI commands (`prompt`, `panel run`, `pack list`, `instruments list|show`, etc.)
- MCP tool signatures (12 tools — see [MCP docs](./mcp.md))
- Instrument YAML formats (v1 flat, v2 linear, v3 branching)
- Persona pack YAML format

## Internal / unstable

These may change without notice between minor versions:

- `synth_panel.runtime.*`
- `synth_panel.orchestrator.*` internals (the orchestrator public methods are stable; internal state is not)
- `synth_panel.plugins.*` — plugin system is evolving
- `synth_panel.structured.*` — structured output subsystem is evolving
- Anything not listed above as public
