# Changelog

All notable changes to synthpanel are documented here.

For auto-generated release notes, see [GitHub Releases](https://github.com/DataViking-Tech/synth-panel/releases).

## [Unreleased]

### Added
- (sp-viz-layer) `synthpanel report RESULT` — post-hoc Markdown renderer for saved panel results. Accepts a result ID or a path to a result JSON; writes to stdout by default or to a file via `-o PATH`. Every report opens with a mandatory synthetic-panel banner and closes with a matching footer so the output can't be mistaken for human-respondent data. Scope is Markdown v1 (provenance, per-model rollup, persona summary, synthesis, failure stats); `--format` accepts only `markdown` and is reserved as a forward-compat slot for HTML in v2. A `synthpanel[report]` optional-deps extra exists and installs cleanly but is currently empty — forward fence for v2 HTML deps. Ships via T1–T5: scaffold (sp-x8fl), loader (sp-kwhl), renderer (sp-u88v), CLI wiring (sp-awfz), docs (sp-z3uy). Full spec at `specs/sp-viz-layer/`.

### Documentation
- (sp-z3uy) README: document `synthpanel report` usage in the quick-start section with stdout / `-o FILE` examples, banner call-out, and a note that the `[report]` extra is currently empty but installs cleanly.
- (sp-z3uy) synthpanel.dev landing page: add `synthpanel report` to the quick-start code snippet with a synthetic-panel banner call-out.

## [0.9.9] - 2026-04-22

### Fixed
- (sp-exu6) Synthesis: `--synthesis-strategy=auto` now routes to `map-reduce` when the estimated prompt would overflow the synthesis model's single-pass context window, instead of hard-failing on the pre-flight check. Mayor introduced the regression during the sp-avmm × sp-9rzu rebase in 0.9.8 — pre-flight ran *before* strategy-select, so `auto` was effectively `single-only with a hard limit`. Dogfooded n=100 ensemble audit surfaced the bug on all three panels.
- (sp-9gcm) Cost: resolve aliases to their canonical OpenRouter-prefixed model IDs before keying into the pricing table. `--models haiku:0.25,deepseek-v3:0.25,gemini-flash-lite:0.25,qwen3-plus:0.25` previously missed sp-oshf's `deepseek-v3.2` and current `gemini-flash-lite` entries, so those models fell through to DEFAULT_PRICING and produced 40–93% divergence warnings in the n=100 audit. Top-level cost was already authoritative via sp-j3vk; this tightens the local-table sanity-check path.

### Added
- (sp-g270) `panel run --personas-merge` now warns (and optionally errors) when a merged pack contains persona names already present in bundled packs. Pre-run stderr line + new top-level `personas_merge_warnings` array in JSON output lists dropped names and post-dedup panel size. New `--personas-merge-on-collision={dedup,error}` flag controls behavior. Caught the n=100 silent 10% shrink that cost mayor 20 minutes of debugging.

### Changed
- (sp-ssrw) Version is now sourced from a single `src/synth_panel/__version__.py` and `pyproject.toml` reads it via `dynamic = ["version"]`. `site/index.html` renders from `site/index.html.j2` with `{{ version }}` substitution. Retires sp-lwy's drift-guard test as a render-correctness check; release-cut friction is now a one-line edit.

## [0.9.8] - 2026-04-22

### Fixed
- (sp-avmm) Synthesis: fail loud when `synthesize_panel()` raises or when the estimated synthesis prompt overflows the synthesis model's context window. Previously the CLI, SDK, and MCP/SDK sync runner all caught synthesis exceptions and proceeded as if synthesis had been skipped, so panel results shipped with `synthesis: null` and exit code 0 even though the API had returned 400 (observed at n=50 where the haiku call requested ~262k vs haiku's 200k context). Now each call site runs a pre-flight token-count check against a context-window table (haiku/sonnet/opus=200k, gemini-*=1M, qwen3=131k, deepseek-v3=128k, default=128k with warn), surfaces a structured `synthesis_error` payload (`error_type`, `message`, `suggested_fix`) at the top level, flags `run_invalid: true`, and exits with code 2 on the CLI path.
- (sp-kvpx) Cost: route per-model and per-panelist cost through `resolve_cost` so `cost_breakdown.total`, `per_model_results[*].cost`, and per-panelist `cost` honor sp-j3vk's provider-reported precedence. Prior to this, `ensemble_run`, `build_mixed_model_rollup`, the sync MCP/SDK runner, and `format_panelist_result` all called `estimate_cost(usage, lookup_pricing(model))` directly, so every non-top-level cost in the ensemble payload stayed on the local pricing table. Observed divergence in the mayor round-5 audit: `total_cost=$0.27` (authoritative) vs `cost_breakdown.total=$0.71` on the same panel.

### Added
- (sp-kkzz) Per-question map-reduce synthesis for the n=50-500 narrative band. `panel run` now accepts `--synthesis-strategy=<single|map-reduce|auto>` (default `auto`). In map-reduce mode, one synthesis call runs per question in parallel (summarizing just that question's responses, with optional cluster-aware persona metadata) followed by one reduce call that combines the per-question summaries into the final cross-question synthesis. `auto` compares a pre-flight token estimate against the synthesis model's context window and picks `single` when it fits, `map-reduce` otherwise.
- (sp-2hpi) Structured response_schema validation and deterministic distribution analysis for bounded question types (Likert, enum, yes/no). Aggregation pipeline computes per-question distributions, subgroup breakdowns, and correlations without an LLM call — foundation for scaling beyond n=500 where narrative synthesis is inappropriate.
- (sp-i2ub) Rate-limit-aware LLM client with `--max-concurrent N` and `--rate-limit-rps RPS` flags. 429s and provider-specific rate-limit errors back off with jitter and honor `retry-after` headers so large-n panels don't trip upstream rate limits.
- (sp-yaru) Live convergence telemetry for panel runs: `--convergence-check-every N` emits running JSD per bounded question, `--auto-stop` halts when rolling-average JSD stays below `--convergence-eps` for `--convergence-m` checks (min floor via `--convergence-min-n`), and the panel output gains a `convergence` report section with per-question curves and convergence-n. `--convergence-baseline DATASET:QUESTION` (optional `synthpanel[convergence]` extras) overlays a human baseline from SynthBench.
- (sp-6wbm) Four new bundled persona packs raising total shipped personas from 24 → 84: `job-seekers` (15), `recruiters-talent` (5), `product-research` (20), `ai-eval-buyers` (20).
- (sp-ftr) Ship the advertised `/synthpanel-poll` slash command.

## [0.9.7] - 2026-04-21

### Fixed
- (sp-j3vk) Cost: trust provider-reported cost over the local pricing table. When a provider returns `usage.cost` (OpenRouter) or equivalent in its response, that value is now recorded verbatim instead of being recomputed from token counts against our maintained rate table. This is the architectural root-cause fix that supersedes the sp-cxyb / sp-5ggf / sp-nn8k / sp-loil bandaids: local pricing drift can no longer inflate or deflate reported spend, and OpenAI-via-OpenRouter paths stop reporting 40× overages when our table is stale relative to the provider's billing.
- (sp-nn8k) Cost: surface `DEFAULT_PRICING` fallback loudly in panel output. When a model is not found in the pricing table and we fall back to the default rate, the panel result now includes a `pricing_fallback` warning listing the affected model(s), so silent mispricing can no longer hide in `$0` or inflated-cost runs. Bandaid ahead of sp-j3vk.
- (sp-27rz) Ensemble: guarantee every weighted model in `--models` gets at least 1 persona. Prior rounding could drop low-weight models entirely (weight < 1/n_personas produced 0 personas after floor), so the ensemble silently ran without models the user explicitly selected. Now ensures ≥1 persona per listed model, redistributing from higher-weight buckets.
- (sp-5ggf) Cost: add pricing table entries for common OpenRouter-proxied models (gpt-4o-mini, qwen, deepseek, mistral variants) so they stop falling through to `DEFAULT_PRICING` and reporting wrong costs. Bandaid ahead of sp-j3vk.
- (sp-cxyb) Cost: correct `SONNET_PRICING` to Claude Sonnet 4.5 rates ($3/M in, $15/M out, $0.30/M cached, $3.75/M cache-write) instead of the stale Opus-3 rates that were doubling reported Sonnet cost. Bandaid ahead of sp-j3vk.

## [0.9.6] - 2026-04-21

### Fixed
- (sp-atvc) Ensemble cost reporting: `metadata.cost.per_model` now buckets panelist token usage by the model that actually ran each panelist and prices each bucket at its own provider's rate. Previously ensemble, `--blend`, and `--models` weighted runs summed tokens across providers then priced the aggregate at the default model's rate, so multi-model runs held a single bucket for the default model only and `total_cost` undercounted by ~6x in the mayor round 4 audit.
- (sp-0h9x) Panel results: `per_model_results` and `cost_breakdown` are now populated on every non-ensemble `panel run` (CLI + MCP), not just `models=[...]` ensemble runs. Mixed-model panels via `persona_models` surface one rollup entry per distinct model; single-model panels surface a one-entry dict. sp-gl9 only wired these fields in the ensemble path, so mayor's audits and other consumers reading the flat panel shape still saw `None`.
- (sp-loil) Cost: price `openrouter/openai/gpt-5-mini` at the published OpenAI rate ($0.25/M in, $2.00/M out, $0.025/M cached input) instead of falling through to the Sonnet default pricing. Unknown-model fallback was inflating reported cost for gpt-5-mini by ~40x (13k/4.8k tokens reported $0.56 vs actual ~$0.013).

## [0.9.5] - 2026-04-21

### Added
- (sp-6yi) `panel run` fails fast on unsubstituted `{placeholder}` variables in instrument or persona packs, with actionable error output listing the missing `--var` keys. Previously the run would proceed and emit garbled prompts.
- (sp-anje) Landing-page-comprehension regression test locks in the sp-6yi fail-fast guard so future refactors can't silently re-allow unsubstituted placeholders into panel runs.
- (sp-on4) `panel run --personas-merge PATH` (repeatable): layer extra persona files onto the base `--personas` pack without hand-editing YAML. Files merge in order; persona entries sharing a `name` with an earlier one replace it in place.
- (sp-x8g) `panel run --dry-run` previews resolved personas, instrument rounds, model selection, and cost estimate without calling any LLM — useful for config validation in CI or pre-run sanity checks.
- (sp-bjt4) Run-level `run_invalid` flag: when ≥50% of panelists report missing required input at the synthesizer stage, the panel result is marked invalid so downstream tooling can surface the failure instead of silently publishing a bad run.
- (sp-8ap) Landing page: audience clarity section, concrete use cases, and example output to help first-time visitors evaluate the tool without digging into docs.
- (sp-6rm) 1280×640 GitHub social preview card asset.

### Fixed
- (sp-ui40) Metadata: resolved `--var` keys and hashed values now fold into `config_hash`, so runs with identical instruments but different variable substitutions produce distinct hashes and don't collide in result stores.
- (sp-mkpo) MCP: BYOK detection now routes through the credentials store rather than reading environment variables directly, so keys persisted via `synthpanel login` are visible to the MCP server.
- (sp-gl9) Ensemble: `per_model_results` and `cost_breakdown` shapes now match the documented contract — clients relying on these fields will no longer see missing keys or type drift.
- (sp-2xy) OpenRouter: request `usage.include` on chat completions and tolerate null `usage` payloads so we stop emitting $0 cost rows for completed turns.
- (sp-bzb) CLI: `--synthesis-model` help text corrected, and the resolved synthesis model now surfaces in the pre-run cost estimate.
- (sp-rn58) Site: drop `.html` from blog `og:url` and `<link rel="canonical">` to stop the 308 redirect that was breaking preview cards on some social platforms.
- (sp-oxw) Site: sync landing page version badge and Schema.org JSON-LD `softwareVersion` to v0.9.4.
- (sp-869) CI: use `tomli` as a fallback for Python 3.10 compatibility where `tomllib` isn't in stdlib.

### Documentation
- (sp-lb4b) README: bump Docker pin example from 0.9.1 to 0.9.4.
- (sp-da6) MCP: document the persona object schema with concrete examples in both `run_panel` and `run_quick_poll` tool descriptions.

## [0.9.4] - 2026-04-20

### Fixed
- (sp-1ez) P0 release packaging: `synthpanel login`/`logout`/`whoami` subcommands were merged to main via PR #178 (sp-lve) on 2026-04-20 but the PR carried no `semver:*` label, so auto-tag never fired and 0.9.3 shipped without the credential-store CLI. This release re-cuts the wheel so the advertised commands actually appear in `synthpanel --help`.

### Added
- (sp-lve) `synthpanel login` / `logout` / `whoami` — persist a per-provider API key to the on-disk credential store so the CLI works without exporting env vars. Key can also be piped (`echo sk-... | synthpanel login`) for CI/script use.

### Fixed
- (sp-t6r) MCP: recognise `OPENROUTER_API_KEY` as a BYOK credential and pick a sensible default model when OpenRouter is the only configured provider.
- (sp-d86) Site: prevent iOS Safari overscroll white flash on the landing page.
- (sp-v1w) Site: bump copy-button touch target to 44px to meet iOS Human Interface Guidelines.

### Documentation
- (sp-dub) Promote the MCP sampling-fallback story to the README opener and landing page; align framework count at 8 across surfaces.
- (sp-ovl) SEO: Schema.org JSON-LD, tightened meta descriptions, and `og:site_name`.
- (sp-fiv) Smithery registry section + refreshed registry-submissions runbook.
- (sp-f12) Add Anthropic Cookbook notebook as the canonical integration-example source.

## [0.9.2] - 2026-04-19

### Fixed
- (sp-6gd) P0 demo blocker: confirm `from synth_panel import quick_poll` works. The public SDK re-exports landed in `src/synth_panel/__init__.py` via sp-2cw.1 but were never published to PyPI — 0.9.0 shipped an empty `__init__.py`. This release cuts the first PyPI build that actually exposes the advertised surface (`quick_poll`, `run_prompt`, `run_panel`, `extend_panel`, `get_panel_result`, `list_instruments`, `list_panel_results`, `list_personas`, plus `PanelResult`, `PollResult`, `PromptResult`).

### Added
- (sp-2cw.1) Public Python SDK convenience layer: `from synth_panel import quick_poll, run_prompt, run_panel, …` now resolves against `synth_panel.sdk`. See `docs/stability.md` for the supported surface.
- (sp-2cw.2) `docs/examples/` — "Works with X" integration examples for 6 agent frameworks (Claude Agent SDK, OpenAI Agents, LangGraph, AutoGen, CrewAI, LlamaIndex).
- (sp-2cw.3) Composio toolkit registration manifest.
- (sp-2cw.4) Expanded Claude Code skills library under `skills/`.
- (sp-2cw.5) Production Docker image published to `ghcr.io/dataviking-tech/synthpanel` and `synthpanel/synthpanel` on tagged releases. Multi-arch (linux/amd64, linux/arm64), python:3.12-slim base, default CMD is `synthpanel mcp-serve`. Reads provider keys from env (`ANTHROPIC_API_KEY` etc.). New CI workflow `.github/workflows/docker.yml` builds and pushes on `v*` tag push or `workflow_dispatch`. README gains a "Run via Docker" section and a GHCR badge.
- (sp-6at) MCP sampling fallback for `run_prompt` and `run_quick_poll` so tools still function when the host supports MCP sampling but no provider key is configured.

### Documentation
- (sp-2cw.6) README "Works with" section lifted above the fold and expanded to seven frameworks.
- (sp-4rp) Landing-page sync for v0.9.x and demo polish.

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
