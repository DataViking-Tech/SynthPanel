# synth-panel — Product Plan

> PM: synthpanel/crew/pm | Date: 2026-04-04

## Product Identity

**synth-panel** is a lightweight, LLM-agnostic research harness for running
synthetic focus groups using AI personas. It lets researchers, product managers,
and designers test ideas against diverse simulated perspectives before committing
to real user research.

**Who it's for:** Product teams, UX researchers, indie hackers, and anyone who
wants fast qualitative signal before spending time/money on real focus groups.

**What problem it solves:** Real user research is slow and expensive. synth-panel
gives you a 5-minute synthetic read from configurable personas — not a
replacement for real research, but a fast pre-filter that catches obvious blind
spots.

**Differentiators:**
- Provider-agnostic (Anthropic, OpenAI, xAI, Gemini — swap with a flag)
- MCP server for IDE integration (Claude Code, Cursor, Windsurf)
- Parallel panelist execution with per-persona cost tracking
- Plugin system for extensibility
- Pure CLI — no web UI to maintain, composable with pipes

## Current State (Honest Assessment)

### What works end-to-end
- **Single prompt mode** — `synth-panel prompt "question"` → LLM response + cost
- **Full panel run** — parallel personas, instrument questions, follow-ups, cost rollup
- **MCP server** — 7 tools, 4 resource patterns, 3 prompt templates, stdio transport
- **All 4 LLM providers** — complete send/stream with retry logic
- **Cost tracking** — per-turn, per-panelist, cumulative, with budget enforcement
- **Session persistence** — save/load/fork with JSON and JSONL serialization
- **Plugin system** — manifest-based, lifecycle hooks, state tracking
- **Structured output** — tool-use forcing with JSON validation and fallback

### What's stubbed or broken
- **REPL user input** — non-slash input prints `[stub]` instead of calling runtime
- **REPL slash commands** — `/compact`, `/permissions`, `/config`, `/memory` are stubs
- **Login/logout CLI commands** — print `[stub]` messages
- **Permission mode** — parsed by CLI but not wired to runtime

### Test suite
- **221 passed, 2 failed, 20 skipped** (243 total)
- 2 failures are known issues (test env setup, not code bugs)
- 20 skips are acceptance tests gated on API key (correct behavior)
- MCP server tests require optional dep (correct behavior)

## Go-Public Definition

"Ready to show the world" means someone can:

1. `pip install synth-panel` and it works
2. Run `synth-panel prompt "test"` with their API key and get a response
3. Run a full panel with the example YAML files
4. Read the README and not find lies
5. Use the MCP server from their IDE
6. Trust the LICENSE

That's it. We're not shipping a polished SaaS product — we're shipping a useful
CLI tool that does what it says on the tin. The bar is **honest, functional, and
installable**.

## Priority List

### BLOCKERS (must fix before go-public)

| # | Issue | Bead | Effort | Why |
|---|-------|------|--------|-----|
| 1 | README claims "zero external dependencies" — false (httpx, pyyaml are required) | sp-v1m | 5 min | PyPI page will repeat this lie |
| 2 | README shows working REPL session — it's stubbed | sp-34q | 5 min (caveat) or 2-4h (implement) | First thing users try after install |
| 3 | No LICENSE file (pyproject.toml says MIT) | sp-t4h | 5 min | Legal blocker for adoption |

### SHOULD-FIX (before or immediately after launch)

| # | Issue | Bead | Effort | Why |
|---|-------|------|--------|-----|
| 4 | Gemini missing from README provider table | sp-2h7 | 5 min | Users won't know it's supported |
| 5 | Fix test that hits live API instead of mocking | sp-s6o | 15 min | CI will fail without API key |
| 6 | Fix test_unreachable_base_url (needs dummy API key) | — | 10 min | Same CI concern |
| 7 | Remove unused pytest-httpx dev dependency | — | 2 min | Clean deps signal quality |
| 8 | Add `authors` field to pyproject.toml | — | 2 min | PyPI metadata |
| 9 | Remove or hide login/logout stub commands | — | 10 min | Confusing dead-ends |

### POLISH (nice for launch but not blocking)

| # | Issue | Bead | Effort | Why |
|---|-------|------|--------|-----|
| 10 | Wire REPL user input to runtime | sp-34q | 2-4h | Makes the REPL actually useful |
| 11 | Wire REPL stub slash commands | — | 1-2h | Feature completeness |
| 12 | Add py.typed marker | — | 2 min | Downstream type-checking |
| 13 | Fix patrol molecule test_command (Go → Python) | sp-50q | 5 min | Gas Town infra issue |

## What to Defer

These should **NOT** be done before launch:

- **Web UI / dashboard** — CLI-first is the identity. A web UI is a different product.
- **OAuth / login flow** — No server component needed for v1. API keys work.
- **Plugin marketplace** — Plugin system works; discovery/distribution is premature.
- **Streaming output in panel mode** — Batch results are fine for v1.
- **Advanced persona generation** — Users define their own; auto-generation is a feature, not a fix.
- **Session management UI** — Persistence works; browsing sessions is a luxury.
- **REPL /compact, /permissions, /config, /memory** — Nice-to-have, not launch-critical. Just don't advertise them.

## Open Questions

1. **REPL: fix or document?** The REPL input stub is the biggest UX gap. Options:
   - (A) Add a caveat to README ("REPL input coming soon") — 5 min, ships faster
   - (B) Wire it up properly — 2-4h, ships a complete product
   - **Recommendation:** (A) for speed, with (B) as first post-launch task.
   CPO call needed.

2. **Package name on PyPI** — Is `synth-panel` available? Need to check. If taken,
   alternatives: `synthpanel`, `synth-focus`, `synthetic-panel`.

3. **Who is listed as author?** Need name/email for pyproject.toml `authors` field.

4. **API key management story** — Currently env vars only. Is that sufficient for
   v1, or do we want a config file / `synth-panel config set` command?

5. **Version number** — Currently 0.1.0. Ship as 0.1.0 (beta signal) or bump to
   1.0.0 (confidence signal)?

## Recommended Execution Order

A single developer can get to go-public in one focused session:

1. Fix README dependency claim (sp-v1m) — 5 min
2. Add LICENSE file (sp-t4h) — 5 min
3. Add REPL caveat to README (sp-34q) — 5 min
4. Add Gemini to README table (sp-2h7) — 5 min
5. Fix test mocking issues (sp-s6o + acceptance test) — 25 min
6. Clean up pyproject.toml (authors, remove pytest-httpx) — 5 min
7. Remove or gate login/logout stubs — 10 min
8. Final test run, commit, tag v0.1.0 — 10 min

**Total estimated developer time: ~70 minutes to go-public ready.**

The product is fundamentally sound. The work is documentation hygiene and test
fixes, not architectural problems.
