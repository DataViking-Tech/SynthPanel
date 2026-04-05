# synth-panel Architecture Review

**Author:** synthpanel/crew/architect | **Date:** 2026-04-04  
**Inputs:** CPO-VISION.md, PRODUCT-PLAN.md, AUDIT.md, full source review

---

## Executive Summary

The codebase is structurally sound. The provider abstraction, orchestrator, runtime, and MCP server are well-designed and functional. The CPO's three identity pillars (open-source, provider-agnostic, agent-native) are architecturally supported — but the agent-native surface needs work to earn "first-class" status. The 6-month direction (instrument sophistication, persona packs, agent-native workflows) is achievable without architectural rewrites, but two decisions should be locked now before they become expensive.

---

## 1. Architecture vs CPO Vision: Assessment

### Pillar 1: Open-Source — SUPPORTED
- Clean `src/` layout, MIT license (file now exists), standard pyproject.toml
- No proprietary dependencies, no phone-home, no account requirement
- Plugin system designed for community extension
- **No gaps.**

### Pillar 2: Provider-Agnostic — SUPPORTED (minor inconsistencies)
- Four providers (Anthropic, OpenAI-compat, xAI, Gemini) with unified `LLMClient` interface
- Alias system (`sonnet` → `claude-sonnet-4-6`, `gemini` → `gemini-2.5-flash`) works
- Prefix-based provider detection is simple and correct
- **Minor inconsistencies:**
  - Only Anthropic parses thinking blocks and cache tokens — others silently drop them
  - Tool choice semantics differ: Anthropic `any` vs OpenAI `required` — mapped correctly but worth documenting
  - `hasattr()` duck-typing for content blocks instead of `isinstance()` — fragile but functional
  - No streaming retry (only `send()` retries) — acceptable for v0.1

### Pillar 3: Agent-Native — PARTIALLY SUPPORTED (gaps below)
The MCP server exists and works. But the CPO said: "Every CLI feature should have an agent-callable equivalent." That's not true today.

---

## 2. Agent-Native Surface: Gap Analysis

The CPO specifically said the MCP server and Claude Code skill are "a first-class product surface equal to the CLI." Current state:

### What MCP has that CLI doesn't
- `run_quick_poll` — single-question convenience (CLI requires full YAML)
- Resource URI patterns (`persona-pack://`, `panel-result://`)
- 3 prompt templates (focus_group, name_test, concept_test)

### What CLI has that MCP doesn't
| CLI Feature | MCP Equivalent | Gap |
|---|---|---|
| `synth-panel prompt "text"` | None | **No single-prompt tool.** MCP only does panel runs. An agent can't ask a quick question without constructing personas. |
| YAML file loading | Inline JSON only | MCP requires callers to pass persona/question dicts inline. Can't reference a file or saved pack by ID in `run_panel`. |
| `--output-format` control | Always JSON | Minor — JSON is correct for agents. Not a gap. |
| Structured output schemas | Not exposed | `StructuredOutputEngine` exists in core but neither CLI nor MCP exposes it. The instrument YAML supports `response_schema` but it's not wired through. |

### Critical agent-native gaps

**Gap 1: No `run_prompt` MCP tool.** The simplest use case — "ask a question with a model" — requires constructing a persona and a panel. An agent doing quick research can't just call `run_prompt("What are the top competitors in X?")`. This is the equivalent of missing `curl` and only having `wget --mirror`.

**Gap 2: Can't reference saved persona packs in `run_panel`.** The MCP server has `save_persona_pack` and `get_persona_pack`, but `run_panel` doesn't accept a `pack_id` parameter. An agent must: get pack → extract personas → pass them to run_panel. This should be one call.

**Gap 3: Structured output not exposed.** The `StructuredOutputEngine` is implemented and tested but not wired into either surface. For agent-native workflows, structured output is essential — agents need JSON they can parse, not free-text responses.

**Gap 4: No instrument sophistication in MCP.** MCP questions are flat `{text, follow_ups}` dicts. No `response_schema`, no branching logic, no conditional follow-ups. The instrument YAML format supports `response_schema` but it's not parsed by the MCP server's question builder.

---

## 3. Implementation Tasks — Dependency-Ordered

### Phase 0: Go-Public Blockers (PM's list — endorse as-is)
These are documentation/packaging tasks, not architecture. Defer to PM's execution order.

1. **README rewrite** (CPO wants this first — agree)
2. **Remove REPL demo from README** (CPO override of PM's "caveat" recommendation)
3. **Remove login/logout stubs** (CPO: "Dead commands erode trust")
4. **Fix dependency claim** (already done per CLAUDE.md update)
5. **LICENSE file** (already added per git pull)
6. **Gemini in README provider table**
7. **Fix 2 test failures** (sp-s6o + unreachable_base_url)
8. **pyproject.toml cleanup** (authors, remove pytest-httpx)
9. **Tag v0.1.0**

### Phase 1: Agent-Native Parity (post-launch, pre-0.2.0)
Priority: make the MCP surface a true equal to CLI.

| # | Task | Depends On | Effort | Why |
|---|------|-----------|--------|-----|
| 1.1 | **Add `run_prompt` MCP tool** — single question, no personas required. Wraps `LLMClient.send()` directly. | — | 2h | Closes the biggest agent-native gap. Most common agent use case. |
| 1.2 | **Add `pack_id` parameter to `run_panel`** — resolve pack from storage, merge with inline personas if both provided. | — | 1h | Eliminates 3-step pack→extract→pass workflow for agents. |
| 1.3 | **Wire `response_schema` through MCP** — add optional `response_schema` to `run_panel` and `run_quick_poll`, pass to `StructuredOutputEngine`. | — | 3h | Enables agents to get structured data, not free text. |
| 1.4 | **Wire `response_schema` through CLI** — `--schema` flag on `panel run`, reads JSON schema from file or inline. | 1.3 | 1h | CLI parity with MCP. |

### Phase 2: Instrument Sophistication (0.2.0 → 0.5.0)
The CPO's second bet: "branching logic, conditional follow-ups, multi-round conversations."

| # | Task | Depends On | Effort | Why |
|---|------|-----------|--------|-----|
| 2.1 | **Design instrument schema v2** — YAML format supporting conditionals, branching, response-dependent follow-ups. Document in SPEC. | — | Design: 4h | Must be designed before implementation. Schema change affects everything downstream. |
| 2.2 | **Instrument parser + validator** — load v2 instruments, validate branching logic, reject cycles. | 2.1 | 8h | Core infrastructure for sophisticated instruments. |
| 2.3 | **Multi-round runtime loop** — extend `_run_panelist()` to run multiple rounds, feeding prior responses into next round's question selection. | 2.1, 2.2 | 12h | The actual behavioral change. Requires runtime to track conversation state across rounds. |
| 2.4 | **Expose v2 instruments in MCP** — accept instrument definition (not just question list) in `run_panel`. | 2.2 | 3h | Agent-native access to sophisticated instruments. |

### Phase 3: Persona Pack Ecosystem (0.2.0+, parallel with Phase 2)

| # | Task | Depends On | Effort | Why |
|---|------|-----------|--------|-----|
| 3.1 | **Bundled starter packs** — 3-5 packs shipped with the package (startup founder, enterprise buyer, healthcare patient, general consumer, developer). | — | 4h | Reduces time-to-value. New users get useful personas immediately. |
| 3.2 | **Pack validation** — schema validation on save, required fields enforcement, personality trait normalization. | — | 2h | Quality gate for community-contributed packs. |
| 3.3 | **Pack import/export CLI commands** — `synth-panel pack list`, `synth-panel pack import file.yaml`, `synth-panel pack export pack-id`. | 3.2 | 3h | CLI parity with MCP pack management. |
| 3.4 | **Community pack repository** — GitHub repo or directory structure for discovering/sharing packs. README links + contribution guide. | 3.1 | 2h | The flywheel CPO described. |

### Phase 4: REPL (0.2.0, if prioritized)

| # | Task | Depends On | Effort | Why |
|---|------|-----------|--------|-----|
| 4.1 | **Wire REPL input to runtime** — replace stub with `AgentRuntime.run_turn()` call. | — | 3h | Makes REPL functional. |
| 4.2 | **Wire `/compact` slash command** — call `session.compact()` with summary. | 4.1 | 1h | Session management for long conversations. |
| 4.3 | **Wire `/config` and `/memory`** — load config files, instruction files. | — | 2h | Nice-to-have, not blocking. |

---

## 4. Technical Debt Blocking the 6-Month Direction

### Debt 1: Duplicated persona/question builders
`_persona_system_prompt()` and `_build_question_prompt()` exist identically in both `cli/commands.py` and `mcp/server.py`. When instrument sophistication arrives (Phase 2), these will diverge or need to be updated in two places.

**Fix:** Extract to a shared module (e.g., `synth_panel/prompts.py`). Do this before Phase 2 work begins.  
**Effort:** 1h  
**Risk if deferred:** Divergent behavior between CLI and MCP panels.

### Debt 2: Orchestrator creates new Session per question
In `_run_panelist()` (orchestrator.py:274-280), each question gets a fresh `Session()`. This means conversation history doesn't carry between questions within a single panelist run. For flat question lists this is fine. For multi-round instruments (Phase 2), this is a blocker — persona responses in round 1 must be visible in round 2.

**Fix:** Single session per panelist, accumulating messages across questions/rounds. The `Session` class already supports this — the orchestrator just needs to stop creating new ones per question.  
**Effort:** 2h  
**Risk if deferred:** Phase 2 (multi-round) requires this. Doing it later means changing behavior of existing panel runs.

### Debt 3: No shared output formatting between CLI and MCP
CLI uses `emit()` (text/json/ndjson). MCP always returns `json.dumps()`. Panel result formatting is done differently in each path. When structured output (Phase 1.3) and instrument v2 (Phase 2.4) land, result formatting will need to be consistent.

**Fix:** Extract result formatting to a shared module that both CLI and MCP consume.  
**Effort:** 2h

### Debt 4: Plugin system has no integration tests
The plugin framework (manifest, hooks, manager, registry) is architecturally complete but untested with real plugins. Before the ecosystem grows, the hook contract needs integration test coverage.

**Fix:** Write 3-5 integration tests with a minimal test plugin.  
**Effort:** 3h  
**Risk if deferred:** Hook contract may break silently.

---

## 5. Architectural Decisions to Make NOW

### Decision 1: Single session per panelist (RECOMMEND: YES)

**Current:** New `Session()` per question.  
**Proposed:** Single `Session()` per panelist, accumulating all Q&A.

**Why now:** This changes the behavior of existing panel runs (personas will see prior Q&A as context). If we ship 0.1.0 with per-question sessions and users build around that behavior, changing it later is a breaking change. If we change it now (or at least before anyone depends on it), it's free.

**Trade-off:** Per-question isolation means each question gets a "fresh" persona. Per-panelist accumulation means later questions benefit from conversational context but may be biased by earlier answers. For research, accumulation is usually what you want — it mirrors how real focus groups work.

**Recommendation:** Change to single session per panelist before 0.1.0 if possible, or immediately after. Document the behavior.

### Decision 2: Instrument schema version field (RECOMMEND: ADD NOW)

**Current:** Instrument YAML has no version field.  
**Proposed:** Add `version: 1` to current format, require it.

**Why now:** Phase 2 introduces instrument schema v2 (branching, conditionals). If v1 instruments don't declare their version, the parser can't distinguish them from v2. Adding `version: 1` to the spec now means v2 can be introduced alongside v1 with graceful fallback.

**Cost:** ~15 minutes. Add `version` field to instrument parser, default to 1 if missing, validate.

### Decision 3: MCP tool naming convention (RECOMMEND: CLEAN UP)

**Current:** MCP tools are named inconsistently:
- `run_panel`, `run_quick_poll` (verb_noun)
- `tool_list_persona_packs`, `tool_get_persona_pack` (tool_verb_noun)
- `tool_save_persona_pack`, `tool_list_panel_results`, `tool_get_panel_result` (tool_verb_noun)

The `tool_` prefix on data tools but not on execution tools is inconsistent. MCP tool names are the API surface agents interact with.

**Recommendation:** Remove the `tool_` prefix before go-public. After launch, renaming tools is a breaking change for any agent workflow that references them. Names should be: `list_persona_packs`, `get_persona_pack`, `save_persona_pack`, `list_panel_results`, `get_panel_result`.

**Cost:** 30 minutes. Rename functions, update tests.

---

## 6. MCP Server + Claude Skill: "First-Class Product Surface" Assessment

**Verdict: SOLID FOUNDATION, NOT YET FIRST-CLASS.**

### What's good
- 7 tools cover the core workflow (run panel, manage packs, retrieve results)
- 4 resource URI patterns provide alternative access
- 3 prompt templates reduce friction for common research patterns
- Async execution with progress reporting
- Auto-persistence of results
- Haiku default model (cost-appropriate for agent-initiated runs)

### What's missing for "first-class"
1. **No `run_prompt` tool** — the simplest use case isn't supported
2. **Can't reference packs by ID in `run_panel`** — unnecessary round-trips
3. **No structured output** — agents get free text, not parseable data
4. **No instrument support** — MCP questions are flat, no schema or branching
5. **Inconsistent tool naming** — `tool_` prefix on some tools
6. **No error detail in responses** — errors are strings, not structured objects with category/retryable fields

### What it takes to be first-class
Phase 1 (tasks 1.1-1.3) plus the naming cleanup. That's ~7 hours of work. The architecture supports all of it — no redesign needed, just wiring.

---

## 7. Summary: What to Do, In What Order

```
NOW (before 0.1.0):
├── README rewrite (CPO priority)
├── Remove REPL demo + login/logout stubs
├── Add instrument version field (Decision 2, 15 min)
├── Clean MCP tool names (Decision 3, 30 min)
├── PM's remaining blockers (LICENSE ✓, deps claim ✓, Gemini row, tests)
└── Tag 0.1.0

IMMEDIATELY AFTER (0.1.x):
├── Phase 1: Agent-native parity (7h total)
│   ├── run_prompt MCP tool
│   ├── pack_id in run_panel
│   └── Structured output wiring
├── Extract shared prompt builders (Debt 1)
├── Single session per panelist (Decision 1 / Debt 2)
└── Bundled starter persona packs (Phase 3.1)

0.2.0:
├── REPL wired up (Phase 4)
├── Instrument schema v2 design (Phase 2.1)
├── Plugin integration tests (Debt 4)
└── Pack validation + CLI commands (Phase 3.2-3.3)

0.5.0:
├── Multi-round instruments (Phase 2.2-2.4)
├── Community pack ecosystem (Phase 3.4)
└── Shared result formatting (Debt 3)
```

The product is architecturally ready for the CPO's vision. The work is wiring, not redesign.
