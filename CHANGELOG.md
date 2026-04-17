# Changelog

All notable changes to synthpanel are documented here.

For auto-generated release notes, see [GitHub Releases](https://github.com/DataViking-Tech/synth-panel/releases).

## [Unreleased]

### Added
- (sp-2cw.5) Production Docker image published to `ghcr.io/dataviking-tech/synthpanel` and `synthpanel/synthpanel` on tagged releases. Multi-arch (linux/amd64, linux/arm64), python:3.12-slim base, default CMD is `synthpanel mcp-serve`. Reads provider keys from env (`ANTHROPIC_API_KEY` etc.). New CI workflow `.github/workflows/docker.yml` builds and pushes on `v*` tag push or `workflow_dispatch`. README gains a "Run via Docker" section and a GHCR badge.

## [0.9.0] - 2026-04-15

### Public Launch
- First release post-public-flip. Repo renamed from `synth-panel` to
  `SynthPanel` (PyPI distribution name `synthpanel` unchanged).
- Pre-launch audit verdict: READY-TO-FLIP (see `docs/release-audit-2026-04-15.md`).

### Documentation
- README badges: PyPI version, CI status, MIT license, Python versions.
- README links updated to canonical Pascal-case repo name.
- CHANGELOG backfilled with 0.5/0.6/0.7 entries.
- `docs/stability.md` documents `lookup_pricing_by_provider` as part of public surface.

### Internal
- Removed Gas Town agent-internal config from public repo.
- Reconciled conflicting CODEOWNERS file.

## [0.8.0] - 2026-04-14

### Added
- (sp-027) `lookup_pricing_by_provider(provider_string)` — parses synthbench-format provider strings (synthpanel/*, openrouter/*, raw-anthropic/*, etc.) into pricing tuples; returns `(None, False)` for ollama, baselines, ensembles, and unresolved providers.

### Fixed
- (sp-027) Multi-question CLI cost-drop: `_run_multi_cli` and `_run_multi_batch` now propagate `total_cost` / `panelist_cost` / `total_usage` / `panelist_usage` to per-response metadata, matching `_run_single`.

### Notes
- Version bumped from 0.4.1 → 0.8.0 to reconcile pyproject.toml with PyPI release line (last published was 0.7.4); minor bump reflects new public API.


## [0.4.1] - 2026-04-14

### Added
- `lookup_pricing_by_provider(provider_string)` helper in `synth_panel.cost`: parses synthbench `config.provider` strings (`synthpanel/`, `openrouter/`, `raw-anthropic/`, `raw-openai/`, `raw-gemini/`, `ollama/` plus `t=`/`profile=`/`tpl=` decorators) and resolves to `(ModelPricing, is_estimated)`. Refuses substring fallback to SONNET so callers (notably synthbench publish) decide whether to emit null. Returns `(None, False)` for `ollama/*`, the named baselines, `ensemble/*`, and unknown inner models. (sp-027)
- `pricing snapshot_date: 2026-04-14` comment above the pricing table to anchor downstream snapshot generation. (sp-027)
- `panelist_usage` field on the rounds-shaped CLI JSON output, restoring symmetry with `panelist_cost`/`total_cost`/`total_usage` so multi-question runs no longer drop a usage bucket downstream consumers rely on. (sp-027)
- v3 branching instruments with `route_when` predicates and DAG validation
- Router predicate engine: `contains`, `equals`, `matches` operators
- Multi-round branching orchestrator loop
- 5 bundled v3 instrument packs: `pricing-discovery`, `name-test`, `feature-prioritization`, `landing-page-comprehension`, `churn-diagnosis`
- `instruments` CLI subcommand: `list`, `show`, `install`, `graph`
- Instrument pack loader (single-file YAML with manifest fields)
- MCP `list_instrument_packs`, `get_instrument_pack`, `save_instrument_pack` tools
- Rounds-shaped panel output with `path`, `terminal_round`, and `warnings` fields
- `extend_panel` MCP tool for ad-hoc follow-up rounds
- Text-mode path line above panel run output
- `--var KEY=VALUE` and `--vars-file` for instrument templates (#39)
- `pack show <id>` as an API-parity alias (#41)
- CI guard to block live API calls in non-acceptance tests
- GitHub Release notes + changelog config in auto-tag workflow

### Fixed
- Multi-question CLI runs now emit the full cost shape (`total_cost`, `total_usage`, `panelist_cost`, `panelist_usage`) on the rounds-shaped output. Previously `panelist_usage` was absent, which silently zeroed the synthbench leaderboard's `$/100Q` column for new rows. (sp-027)
- Fail loud when all provider requests error (#37)
- Default `--model` now respects available credentials and announces pick (#38)
- Publish workflow trigger corrected + manual PyPI setup documented (#40)
- `contents: read` permission added to publish job (#42)

## [0.4.0] - 2026-04-10

First published release on [PyPI](https://pypi.org/project/synthpanel/).

### Added
- v2 multi-round linear instruments with session reuse across rounds
- Instrument v2 parser with multi-round support
- Template engine for dynamic question rendering
- Session persistence — save/load per panelist
- `response_sentiment` condition evaluator with LLM-based classification
- Panel synthesis module (`synthesize_panel`) wired into CLI and MCP
- Condition evaluation module for conditional follow-ups
- Persona pack registry with 5 bundled starter packs
- Structured output via tool-use forcing, wired through MCP and CLI
- Semver auto-tag + PyPI publish workflow (trusted publishing)

### Fixed
- MCP import guard + mock alias test to avoid live API calls
- Condition evaluation wired into orchestrator follow-up loop

## [0.3.0]

### Added
- Structured output via tool-use forcing
- Cost tracking with per-turn token accounting (4 buckets: input, output, cache_write, cache_read)
- MCP server with stdio transport (12 tools, 4 resources, 3 prompt templates)
- Persona-pack persistence (`save_persona_pack`, `get_persona_pack`, `list_persona_packs`)
- Panel result persistence and retrieval

## [0.7.4] - 2026-04-14

Patch release in the 0.7.x series. See the [README Versions table](README.md#versions) for the headline 0.7.x features and the GitHub Release notes for per-tag detail.

## [0.7.0] - 2026-04-14

### Added
- Multi-model ensemble blending (`--blend`)
- OpenRouter provider support
- Temperature / top_p controls
- Persona prompt template customization (see `templates/`)

## [0.6.0] - 2026-04-13

### Added
- `--models` weighted model spec (e.g., `haiku:0.33,gemini:0.33,gpt-4o-mini:0.34`)
- `--temperature` / `--top_p` flags
- Persona prompt templates
- Pack generation helpers
- Domain templates
- MCP server improvements

## [0.5.0] - 2026-04-12

### Added
- v3 branching instruments with `route_when` predicates and DAG validation
- Router predicate engine: `contains`, `equals`, `matches` operators
- Multi-round branching orchestrator loop
- 5 bundled v3 instrument packs: `pricing-discovery`, `name-test`, `feature-prioritization`, `landing-page-comprehension`, `churn-diagnosis`
- `instruments` CLI subcommand: `list`, `show`, `install`, `graph`
- Instrument pack loader (single-file YAML with manifest fields)
- MCP `list_instrument_packs`, `get_instrument_pack`, `save_instrument_pack` tools
- Rounds-shaped panel output with `path`, `terminal_round`, and `warnings` fields
- `extend_panel` MCP tool for ad-hoc follow-up rounds

[Unreleased]: https://github.com/DataViking-Tech/synth-panel/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/DataViking-Tech/synth-panel/compare/v0.7.4...v0.8.0
[0.7.4]: https://github.com/DataViking-Tech/synth-panel/compare/v0.7.0...v0.7.4
[0.7.0]: https://github.com/DataViking-Tech/synth-panel/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/DataViking-Tech/synth-panel/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/DataViking-Tech/synth-panel/compare/v0.4.0...v0.5.0
[0.4.1]: https://github.com/DataViking-Tech/synth-panel/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/DataViking-Tech/synth-panel/releases/tag/v0.4.0
[0.3.0]: https://github.com/DataViking-Tech/synth-panel/releases/tag/v0.3.0
