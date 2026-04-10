# Changelog

All notable changes to synth-panel are documented here.

For auto-generated release notes, see [GitHub Releases](https://github.com/DataViking-Tech/synth-panel/releases).

## [Unreleased]

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
- Text-mode path line above panel run output
- `--var KEY=VALUE` and `--vars-file` for instrument templates (#39)
- `pack show <id>` as an API-parity alias (#41)
- CI guard to block live API calls in non-acceptance tests
- GitHub Release notes + changelog config in auto-tag workflow

### Fixed
- Fail loud when all provider requests error (#37)
- Default `--model` now respects available credentials and announces pick (#38)
- Publish workflow trigger corrected + manual PyPI setup documented (#40)
- `contents: read` permission added to publish job (#42)

## [0.4.0] - 2026-04-10

First published release on [PyPI](https://pypi.org/project/synth-panel/).

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

[Unreleased]: https://github.com/DataViking-Tech/synth-panel/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/DataViking-Tech/synth-panel/releases/tag/v0.4.0
[0.3.0]: https://github.com/DataViking-Tech/synth-panel/releases/tag/v0.3.0
