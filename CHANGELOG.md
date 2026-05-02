# Changelog

All notable changes to synthpanel are documented here.

For auto-generated release notes, see [GitHub Releases](https://github.com/DataViking-Tech/SynthPanel/releases).

## [Unreleased]

### Added
- (GH-289, sp-b8y47x) New bundled `students` persona pack — 15 personas spanning undergraduate (5), graduate (5), and non-traditional learners (5). Covers in-state public, HBCU, liberal-arts, large-public commuter, and international F-1 cohorts at the undergrad layer; PhD, professional master's (MBA, MPH), and MD-PhD at the graduate layer; and returning adult, online part-time, bootcamp career-changer, GI-bill veteran, and working-clinical-professional online-master's profiles for non-traditional. Demographics, funding sources, and pain points differ per persona to avoid the "rounding-error" failure mode flagged in the n=100 self-audit. Total shipped persona count rises 145 → 160 across 10 bundled packs.
- (GH-297, sp-tzavk0) `synthpanel analyze <result> --output responses-csv` — emit one row per (panelist, question) response as a flat CSV for spreadsheet workflows (Google Sheets / Excel pivots, qualitative coding). Default columns: `persona_id, persona_name, question_id, question_text, response, response_type, cost`; opt-in extras via `--columns` (`model`, `variant_of`, `input_tokens`, `output_tokens`, `error`). Distinct from the existing `--output csv` analytical summary. Cells are CSV-injection-safe (formula triggers `=`, `+`, `-`, `@` and control chars get a `'` prefix per OWASP guidance), embedded newlines and commas round-trip cleanly through `csv.DictReader`, and rows use RFC 4180 CRLF terminators. Structured (dict/list) responses serialize as JSON.
- (GH-308, sp-4y5.1) `synthpanel pack diff <pack-a> <pack-b>` — compare two persona packs side-by-side. Reports added/removed/unchanged/changed personas (matched by name), per-persona field-level diffs (age, occupation, background, traits, gender), and composition deltas (age range, age mean, role distribution, gender split when present). Accepts built-in pack names, user-saved pack IDs, or YAML file paths for either side; supports `--format json` for CI integration.
- (sy-ws76) `synthpanel panel run --resume <run-id>` is now a standalone entry point: pass just the run id and the original `--personas` / `--instrument` paths are recovered from the checkpoint's saved CLI args. Existing flags can still be passed to override. New `--allow-drift` flag downgrades checkpoint config drift from a hard error to a warning ("statistically inconsistent" run), for cases where intentionally mixing configs is acceptable. Pre-`sy-ws76` checkpoints (no `cli_args` field) still load — back-compat preserved.
- (sp-4loufu) Per-persona LLM overrides via a YAML `llm_overrides:` block on each persona, accepting `temperature`, `top_p`, `max_tokens`, and `model`. Lets researchers vary stochasticity within a single panel — e.g. a "deliberate" persona at `temperature: 0.3` and an "exploratory" one at `0.9` — without giving up the run-level `--temperature` default for everyone else. Overrides flow through the structured-output and extraction calls too, are validated up front (out-of-range temperature, unknown keys, etc. fail the run before any LLM call), and naturally show up in per-persona cost tracking because each persona's request carries its own `max_tokens`. The new `llm_overrides.model` is recognised alongside the legacy top-level `model` field; top-level wins on collision so existing YAML is unchanged.

### Fixed
- (GH-340, sp-xehk) Provider clients now share a single, formally-named retry/backoff policy (`synth_panel.llm.retry.RetryPolicy`) instead of relying on retry logic buried inside `LLMClient`. The class encapsulates the budgets, backoff curve, and `Retry-After` handling that previously lived in `_with_retry`, so all five providers (Anthropic, Gemini, xAI, OpenRouter, OpenAI-compatible) get identical behavior and the policy is now reusable / injectable via `LLMClient(retry_policy=...)`. Retry attempts log at `INFO` (was `WARNING`) with `provider`, `attempt`, and `reason` fields so operators can see where backoff is happening per-provider without enabling DEBUG. Provider display names are formalized on `ProviderConfig.name`. Behavior unchanged: 401 still does not retry, 429 retries with budget+jitter, server-supplied `Retry-After` still dominates exponential backoff. Closes #340.
- (GH #287, sp-stkj2w) The "Missing API key" error raised by every provider now names the missing env var (`ANTHROPIC_API_KEY`, etc.), recommends both the persistent option (`synthpanel login --provider <name>`) and a one-shot `export <ENV_VAR>=...`, and — for Anthropic specifically — calls out the Claude Code OAuth footgun (Claude Code's keychain tokens use a different auth scheme and are not reusable as Anthropic API keys). The Gemini path lists both `GEMINI_API_KEY` and `GOOGLE_API_KEY`. New `synth_panel.credentials.missing_api_key_message` helper centralises the wording so future providers stay consistent. Closes #287.
- (GH #298) Terminal output (model-assignment table, `panel inspect` per-persona summary, `analyze` frequency table) now aligns columns correctly when persona names or theme categories contain CJK characters, accented Latin (precomposed or decomposed), or emoji. Previously padding counted code points rather than rendered cells, so `"王芳"` (4 cells, 2 code points) would shift right of an ASCII row and break alignment. New stdlib-only `synth_panel.text_width` helper handles East Asian Wide / Fullwidth, combining marks (zero-width), and common emoji blocks; no new dependencies.
- (sp-4y5.9, GH #311) `synthpanel pack inspect <pack-id>` no longer silently truncates long persona fields. Description, occupation, background, and traits are word-wrapped to terminal width by default with a continuation indent. Pass `--full` to skip wrapping and preserve embedded newlines (paragraph breaks survive). Previously a long `description` or `background` field would appear cut off at terminal width with no indication that truncation had occurred — a copy-paste hazard for users reviewing personas. Closes #311.
- (GH-299, sp-60w2te) `synthpanel panel run --checkpoint-dir` no longer silently overwrites an existing checkpoint when a fresh run id collides with one already on disk. Before this fix, a `new_run_id()` collision (or two concurrent invocations sharing the same checkpoint root) could destroy the first run's progress without warning, and a later `--resume <id>` would resume the second run's state instead. The checkpoint writer now refuses on collision with a clear error pointing at `--resume <id>` to continue or `--force-overwrite` to replace; concurrent fresh starts on the same id are blocked by a per-directory `fcntl.flock`, so the race condition cannot happen even if the existence check would have passed. Closes #299.

### Changed (loudness)
- (sp-g59o) Detection: warn loudly when synthesis output appears unstructured (likely model schema-adherence flake). Triggered when every list field — themes, agreements, disagreements, surprises — is empty while the recommendation slot carries >600 chars of prose. Surfaces as a `synth_panel.synthesis` `logger.warning`, on `SynthesisResult.warnings`, and propagated up to `PanelResult.warnings`. Schema-honoring runs are unchanged. Observed at ~25% on `gemini-flash-lite` synthesis; detection is provider-agnostic.
- (sp-k2ed4a) MCP sampling truncation surfacing: `synth_panel.mcp.sampling.sample_text` now detects host-side `stopReason="maxTokens"` truncation, logs a `logger.warning`, and returns `truncated`/`requested_max_tokens`/`warning` fields. The sampling paths in `run_prompt`, `run_panel`, and `run_quick_poll` propagate truncated turns into the response `warnings` list with persona/synthesis labels, so a failed structured-output parse can be attributed to the host clipping output rather than the model ignoring the schema. No protocol-level startup check is possible — MCP capability negotiation does not expose the host's max_tokens cap — so per-turn detection is the loud surface.

### Fixed
- (sp-4y5.7, GH #309) Cap `anthropic` and `openai` SDK loggers at WARNING by default, completing the third-party DEBUG-leak fix from PR #352. Issue #309 explicitly listed both libraries as noisy at DEBUG; they were missing from the `_NOISY_LOGGERS` set. `--debug-all` (and its help text) now surface them alongside `httpx`/`httpcore`/`urllib3`/MCP/websocket libs.

## [0.12.0] - 2026-04-26

Minor bump shipping two new CLI features (`--best-model-for`,
`--submit-to-synthbench`), the new `synthpanel pack calibrate` subcommand,
six bundled persona packs deepened from 5 → 15 personas (~60 new personas
addressing the 'too generic' finding from the n=100 self-audit), and two
`synthpanel report` rendering improvements.

### Added
- (sp-zq3) `synthpanel panel run --best-model-for TOPIC[:DATASET]` — fetches SynthBench public leaderboard.json, picks the top-SPS model for the requested topic, surfaces the recommendation with SPS, JSD, n, and $/100q context. Uses 24h cache at `~/.synthpanel/synthbench-cache.json`. Falls back gracefully when SynthBench is unreachable. Plus a generated docs page at synthpanel.dev/recommended-models mapping use-cases to SynthBench-validated model picks.
- (sp-ezz) `synthpanel panel run --submit-to-synthbench` — opt-in submission of calibrated panel runs to the SynthBench public leaderboard via Tier-2 API. Hard-fails at parse time without `--calibrate-against` (only calibrated runs produce SynthBench-shaped scores). First-run consent prompt explaining privacy implications; `--yes` to bypass for CI use. Requires `SYNTHPANEL_SYNTHBENCH_API_KEY` env var.
- (sp-sghl) `synthpanel pack calibrate <pack-yaml> --against DATASET:QUESTION` — first-class pack calibration: runs a panel using the pack against a SynthBench baseline, computes JSD, writes the result back into the pack YAML as a top-level `calibration:` list. Supports `--n`, `--models`, `--samples-per-question`, `--output`, `--dry-run`, `--yes`. Round-trip preserves persona definitions exactly via ruamel.yaml. Plus new docs/calibration.md methodology guide explaining JSD interpretation and which packs to calibrate against which questions.
- (sp-edqg, sp-bzgm, sp-xjty, sp-cs0q, sp-z28k, sp-ebrl) Six bundled packs deepened from 5 → 15 personas: developer, enterprise-buyer, general-consumer, healthcare-patient, recruiters-talent, startup-founder. ~60 new personas across role specialty, demographic depth, and career stage. Addresses the 'persona packs too generic' finding from the n=100 self-audit.

### Changed
- (sp-f9jg) Per-model rollup in `synthpanel report` now buckets correctly by canonical model id. Prior reports showed alias rows (haiku, gemini-flash-lite) with tokens but no cost, AND canonical rows (openrouter/anthropic/claude-haiku-4.5) with cost but no tokens — duplicate rows that misled users. Now one row per model with both columns populated.
- (sp-xltd) `synthpanel report` synthesis section now renders the full themes / agreements / disagreements / recommendation, not just a 240-char summary peek. Closes the 'synthesis is a black box' gap that the n=100 self-audit surfaced as a real product weakness.

## [0.11.0] - 2026-04-24

Minor bump shipping the `sp-i2ub` scaled-orchestration epic (panelist-level
checkpointing, mid-run cost gate, valid-partial-JSON abort discipline), a
6-bug loudness sweep that converts silent failures into loud ones, and two
CI hygiene fixes.

### Added
- (sp-hsk3) Panelist-level checkpointing with `--resume <run-id>`: persists run state every K=25 panelists (override via `--checkpoint-dir PATH`, default `~/.synthpanel/checkpoints/<run-id>/`). Auto-checkpoint on SIGINT/SIGTERM. `--resume` picks up without reprocessing completed panelists.
- (sp-4hhk, replaces sp-utnk) `--max-cost <USD>` mid-run projected-total cost gate. Projected = `running_cost / current_n * total_n`; halt gracefully when projected exceeds threshold. Halt produces valid partial JSON with `run_invalid: true`, `cost_exceeded: true`, `halted_at_panelist`.
- (sp-56pb) Valid partial JSON on every abort path: rate-exhaustion, SIGINT, `--max-cost` gate, and individual panelist failure all produce parseable JSON for completed panelists `0..k` with `run_invalid: true` and a specific `abort_reason`. Exit code is non-zero (2) on abort.

### Changed (loudness)
- (sp-s1is) Alias config parse failures (YAML/JSON) now log warnings instead of silently returning an empty dict.
- (sp-qvqx) Synthesis with a partial structured payload now fails loudly instead of yielding empty fields.
- (sp-0ozi) MCP `extend_panel` surfaces synthesis exceptions in the tool response payload (was `synth: null`).
- (sp-t5ok) Condition evaluator warns loudly on unknown condition types and missing sentiment client (was silent default-True).
- (sp-319x) Orchestrator records follow-up exceptions in the response payload (was silently dropped).

### Fixed
- (sp-rmtj) `test_aliases` fixture isolated from the developer's `~/.synthpanel/aliases.yaml` (was flaky on machines with non-default aliases).

### CI
- (sp-42i) Auto-tag workflow now fails loudly on release PRs without a semver label, defaulting to `semver:patch` when the title starts with `chore(release):`.
- (sp-kdya) `pip-audit` ignores CVE-2026-3219 in pip 26.0.1 (no patched pip released yet; remove ignore once fix lands).

## [0.10.0] - 2026-04-23

Minor bump shipping three completed QRSPI epics — `sp-viz-layer` (post-hoc
Markdown reporting), `sp-inline-calibration` (inline calibration against
published human baselines), and `sp-pack-registry` (decentralized HACS-style
persona-pack registry) — plus supporting work merged since v0.9.9.

### Added
- (sp-viz-layer) `synthpanel report RESULT` — post-hoc Markdown renderer for saved panel results. Accepts a result ID or a path to a result JSON; writes to stdout by default or to a file via `-o PATH`. Every report opens with a mandatory synthetic-panel banner and closes with a matching footer so the output can't be mistaken for human-respondent data. Scope is Markdown v1 (provenance, per-model rollup, persona summary, synthesis, failure stats); `--format` accepts only `markdown` and is reserved as a forward-compat slot for HTML in v2. A `synthpanel[report]` optional-deps extra exists and installs cleanly but is currently empty — forward fence for v2 HTML deps. Ships via T1–T5: scaffold (sp-x8fl), loader (sp-kwhl), renderer (sp-u88v), CLI wiring (sp-awfz), docs (sp-z3uy). Full spec at `specs/sp-viz-layer/`.
- (sp-5r88 / sp-a6jc / sp-ttwy / sp-bldz) Inline SynthBench calibration via `panel run --calibrate-against DATASET:QUESTION`. Force-enables convergence tracking against a published human baseline (v1 allowlist: `gss`, `ntia`), auto-derives a `pick_one` extractor schema from the baseline when option count ≤ 5 (override with `--extract-schema`), and attaches a `calibration` sub-object to every tracked question in the output. The sub-object carries `jsd`, `baseline_spec`, `extractor`, `auto_derived`, and — on disjoint supports — `alignment_error`. Requires `pip install 'synthpanel[convergence]'`. Cadence is NOT implicit — pair with `--convergence-check-every` to control sampling.
- (sp-udsv) `gh:` URL resolver — parses `gh:owner/repo[@ref][/path]` into raw-content URLs with tight allowlist validation. Foundation piece for the pack registry (pack import from GitHub).
- (sp-7we4) Decentralized registry module at `synth_panel.registry` with HTTP fetch + on-disk cache layers. 24h TTL, offline fallback when the remote is unreachable, and deterministic cache keys so cold/warm runs produce identical lookups.
- (sp-w9a5) `synthpanel pack import gh:<user>/<repo>[@ref][/path]` — import persona packs directly from GitHub via the `gh:` resolver. `--unverified` affordance required for packs outside the curated registry; collision UX surfaces existing local packs and offers `--force` to overwrite.
- (sp-vzhl) `synthpanel pack search <term>` substring search over cached registry entries, and `synthpanel pack list --registry` to enumerate available packs from the registry (falls back to last good cache offline).
- (sp-lk3w) Optional `version:` field on persona packs. MCP surfaces a non-fatal shadow warning when a user-installed pack shadows a bundled pack with an older version string, so silently-stale packs can't sit on top of a newer bundled definition.

### Changed
- (sp-bldz) Convergence: inline calibration now attaches a `per_question[key].calibration` sub-object as the shipped wire format (`jsd`, `baseline_spec`, `extractor`, `auto_derived`, and — on disjoint supports — `alignment_error`). A flat `per_question[key].human_jsd` scalar was considered during D-gate and rejected; any downstream consumer that wrote speculative code against `.human_jsd` should migrate to `.calibration.jsd`.
- (sp-ttwy) `pick_one` extractor schema is auto-derived from the baseline when the baseline option count is ≤ 5; hard-fails otherwise so callers are forced to pass an explicit `--extract-schema`.

### Documentation
- (sp-z3uy) README + synthpanel.dev: document `synthpanel report` usage in the quick-start section with stdout / `-o FILE` examples, synthetic-panel banner call-out, and a note that the `[report]` extra is currently empty but installs cleanly.
- (sp-0g9r / sp-7npy) Convergence docs: document `--calibrate-against` and the shipped `per_question[key].calibration` sub-object wire format. Any downstream consumer that wrote speculative code against `.human_jsd` should migrate to `.calibration.jsd`.
- (sp-ezcq) New `docs/registry.md` reference covering `pack import gh:...`, `pack search`, `pack list --registry`, the 24h cache, and the contribution flow for community packs.
- (sp-o1y0) Landing page: "Who this isn't for" positioning block surfacing non-enterprise scope.
- (#251) Full doc audit and refresh for the v0.9.9 feature set across README, CHANGELOG, and site.

### Tests
- (sp-m1mz) Acceptance: live-registry smoke test covering cache miss → fetch → cache hit paths end-to-end.
- (sp-idqa) Acceptance: end-to-end calibration against live GSS HAPPY baseline validating the full `--calibrate-against` path.

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

[Unreleased]: https://github.com/DataViking-Tech/SynthPanel/compare/v0.12.0...HEAD
[0.12.0]: https://github.com/DataViking-Tech/SynthPanel/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/DataViking-Tech/SynthPanel/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/DataViking-Tech/SynthPanel/compare/v0.9.9...v0.10.0
[0.9.9]: https://github.com/DataViking-Tech/SynthPanel/compare/v0.9.8...v0.9.9
[0.9.8]: https://github.com/DataViking-Tech/SynthPanel/compare/v0.9.7...v0.9.8
[0.9.7]: https://github.com/DataViking-Tech/SynthPanel/compare/v0.9.6...v0.9.7
[0.9.6]: https://github.com/DataViking-Tech/SynthPanel/compare/v0.9.5...v0.9.6
[0.9.5]: https://github.com/DataViking-Tech/SynthPanel/compare/v0.9.4...v0.9.5
[0.9.4]: https://github.com/DataViking-Tech/SynthPanel/compare/v0.9.2...v0.9.4
[0.9.2]: https://github.com/DataViking-Tech/SynthPanel/compare/v0.9.0...v0.9.2
[0.9.0]: https://github.com/DataViking-Tech/SynthPanel/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/DataViking-Tech/SynthPanel/compare/v0.7.4...v0.8.0
[0.7.4]: https://github.com/DataViking-Tech/SynthPanel/compare/v0.7.0...v0.7.4
[0.7.0]: https://github.com/DataViking-Tech/SynthPanel/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/DataViking-Tech/SynthPanel/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/DataViking-Tech/SynthPanel/compare/v0.4.0...v0.5.0
[0.4.1]: https://github.com/DataViking-Tech/SynthPanel/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/DataViking-Tech/SynthPanel/releases/tag/v0.4.0
[0.3.0]: https://github.com/DataViking-Tech/SynthPanel/releases/tag/v0.3.0
