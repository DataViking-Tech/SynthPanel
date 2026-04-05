# synth-panel — Execution Plan

> PM: synthpanel/crew/pm | Date: 2026-04-04  
> Inputs: PRODUCT-PLAN.md, CPO-VISION.md, ARCHITECTURE-REVIEW.md, AUDIT.md

This is the definitive plan that polecats execute against.

---

## Scope Boundaries

| Version | Scope |
|---------|-------|
| **0.1.0** | Honest README, no dead code, clean tests, correct metadata. Ship what works. |
| **0.2.0** | Agent-native parity (MCP = CLI), REPL wired, starter persona packs, instrument v2 design |
| **0.5.0** | Multi-round instruments, community pack ecosystem, shared result formatting |
| **Deferred indefinitely** | Web UI, OAuth, plugin marketplace, streaming panel output, persona auto-generation |

---

## Phase 0: Go-Public (0.1.0)

Ordered by execution dependency. A single polecat can ship this in one session.

### GP-1: README rewrite ← DO FIRST
**Bead:** sp-cs7  
**Effort:** 45 min  
**Dependencies:** None  
**Assignable to:** Any polecat

The CPO and architect agree: rewrite the README, don't patch it line-by-line.
The README is the product's face and every other fix is invisible until it's honest.

**What to change:**
1. **Remove the REPL demo entirely** (lines ~184-193). Don't caveat — delete.
   The CPO is clear: "A caveat next to a demo that doesn't work makes the product
   look apologetic." When the REPL works, add it back.
2. **Remove any "zero dependencies" claim.** Replace with: "Minimal dependencies:
   `httpx` for HTTP, `pyyaml` for YAML parsing. Optional: `mcp` for MCP server."
   (Note: the CLAUDE.md architecture line was already fixed, but check README body.)
3. **Add Gemini to the provider table.** Row:
   `| Google | GOOGLE_API_KEY or GEMINI_API_KEY | gemini-2.5-flash |`
4. **Remove any mention of `login`/`logout` commands.** They're stubs being removed.
5. **Verify every code example in the README actually works.** If it doesn't, remove it.
6. **Update the one-liner** to match CPO positioning:
   "Open-source synthetic focus groups. Any LLM. Your terminal or your agent's tool call."

**Acceptance criteria:**
- Every code example in the README is runnable (given an API key)
- No mention of features that are stubbed
- Provider table has all 4 providers
- Dependency claims are accurate

**Closes:** sp-v1m, sp-34q, sp-2h7

### GP-2: Remove login/logout stub commands
**Bead:** sp-p4z  
**Effort:** 15 min  
**Dependencies:** None  
**Assignable to:** Any polecat

Remove `login` and `logout` subcommands from the CLI entirely. CPO: "Dead commands
erode trust. If we add auth later, we add the commands then."

**Files to change:**
- `src/synth_panel/cli/parser.py` — remove login/logout subparser registration
- `src/synth_panel/cli/commands.py` — remove `handle_login()` and `handle_logout()` functions
  (currently at lines ~247-258)

**Acceptance criteria:**
- `synth-panel login` returns "unknown command" or similar
- `synth-panel --help` does not list login/logout
- No dead code left behind

### GP-3: Fix test that hits live Anthropic API
**Bead:** sp-s6o (exists)  
**Effort:** 15 min  
**Dependencies:** None  
**Assignable to:** Any polecat

`tests/test_client.py::TestAliasResolution::test_alias_is_resolved_in_send` makes
a real API call. Should mock the provider's `send()` method.

**What to do:**
- Mock `LLMClient._resolve_provider()` or the provider's `send()` to return a
  fake `CompletionResponse`
- Assert the resolved model name is correct (the alias was expanded)
- Do NOT add `@pytest.mark.acceptance` — this is a unit test that should always pass

**Acceptance criteria:**
- Test passes without any API key set
- Test still validates that alias resolution flows through to the provider

### GP-4: Fix test_unreachable_base_url
**Bead:** sp-5io  
**Effort:** 10 min  
**Dependencies:** None  
**Assignable to:** Any polecat

`tests/test_acceptance.py::test_unreachable_base_url` fails because no dummy API
key is set, so it gets `MISSING_CREDENTIALS` instead of `TRANSPORT` error.

**What to do:**
- Set a dummy `ANTHROPIC_API_KEY` in the test's setup (e.g., `sk-ant-dummy-for-test`)
- The test should then correctly hit the unreachable URL and get a transport error

**Acceptance criteria:**
- Test passes without a real API key
- Error category is TRANSPORT or RETRIES_EXHAUSTED, not MISSING_CREDENTIALS

### GP-5: Clean MCP tool naming
**Bead:** sp-ybh  
**Effort:** 30 min  
**Dependencies:** None  
**Assignable to:** Any polecat

Architect Decision 3: remove `tool_` prefix from MCP data tools before launch.
After launch, renaming is a breaking change for agent workflows.

**Current → Target:**
- `tool_list_persona_packs` → `list_persona_packs`
- `tool_get_persona_pack` → `get_persona_pack`
- `tool_save_persona_pack` → `save_persona_pack`
- `tool_list_panel_results` → `list_panel_results`
- `tool_get_panel_result` → `get_panel_result`

**Files to change:**
- `src/synth_panel/mcp/server.py` — rename the 5 functions
- `tests/test_mcp_server.py` — update any references to old names
- README.md / CLAUDE.md — update tool list if mentioned

**Acceptance criteria:**
- MCP server starts and all 7 tools are callable
- No `tool_` prefixed function names remain in mcp/server.py
- Tests pass (or are updated to match)

### GP-6: Add instrument version field
**Bead:** sp-9uv  
**Effort:** 15 min  
**Dependencies:** None  
**Assignable to:** Any polecat

Architect Decision 2: add `version: 1` to instrument schema now, before anyone
depends on the unversioned format.

**What to do:**
- In the instrument YAML parser (likely in `cli/commands.py` where instruments are
  loaded), accept and validate a `version` field
- Default to `1` if missing (backward compat with example files)
- Add `version: 1` to `examples/survey.yaml`
- Document the field in CLAUDE.md's instrument YAML section

**Acceptance criteria:**
- `version: 1` instruments parse correctly
- Instruments without `version` default to 1 (no breakage)
- Example files updated

### GP-7: pyproject.toml cleanup
**Bead:** sp-7b5  
**Effort:** 10 min  
**Dependencies:** None  
**Assignable to:** Any polecat

**What to do:**
1. Add `authors` field. Use: `authors = [{name = "DataViking", email = "openclaw@dataviking.tech"}]`
   (confirm with mayor if different)
2. Remove `pytest-httpx` from `[project.optional-dependencies.dev]` — it's declared
   but never imported in any test file
3. Verify `license = "MIT"` matches the LICENSE file content

**Acceptance criteria:**
- `pip install -e ".[dev]"` doesn't pull pytest-httpx
- PyPI metadata preview shows author info
- No unused dev dependencies

### GP-8: Tag v0.1.0
**Bead:** sp-ogb  
**Effort:** 10 min  
**Dependencies:** GP-1 through GP-7 all complete  
**Assignable to:** Any polecat (or PM)

**What to do:**
1. Run full test suite: `pytest tests/` — expect 0 failures (excluding skips and optional deps)
2. Verify `git status` is clean
3. `git tag -a v0.1.0 -m "v0.1.0: go-public release"`
4. `git push origin v0.1.0`

**Acceptance criteria:**
- All tests pass (no failures)
- Tag exists on remote
- `pip install synth-panel` from the repo works

---

## Phase 1: Agent-Native Parity (0.1.x → 0.2.0)

Post-launch. ~10h total. Makes MCP a true equal to CLI per CPO directive.

### AN-1: Add `run_prompt` MCP tool
**Effort:** 2h  
**Dependencies:** None

Add a tool that wraps `LLMClient.send()` directly. An agent should be able to
ask a single question without constructing personas or a panel.

**Interface:** `run_prompt(prompt: str, model?: str, system_prompt?: str) → {response: str, usage: {...}, cost: {...}}`

### AN-2: Add `pack_id` parameter to `run_panel`
**Effort:** 1h  
**Dependencies:** None

Allow `run_panel(pack_id="startup-founders", questions=[...])` instead of requiring
the caller to get_pack → extract personas → pass inline. If both `pack_id` and
`personas` are provided, merge them.

### AN-3: Wire structured output through MCP
**Effort:** 3h  
**Dependencies:** None

`StructuredOutputEngine` exists and is tested but neither CLI nor MCP exposes it.
Add optional `response_schema` to `run_panel` and `run_quick_poll`. Pass through
to the structured output engine.

### AN-4: Wire structured output through CLI
**Effort:** 1h  
**Dependencies:** AN-3

Add `--schema` flag to `synth-panel panel run`. Accepts a JSON schema from file
or inline. Passes to `StructuredOutputEngine`.

### AN-5: Extract shared prompt builders
**Effort:** 1h  
**Dependencies:** None (but do before Phase 2)

`_persona_system_prompt()` and `_build_question_prompt()` are duplicated in
`cli/commands.py` and `mcp/server.py`. Extract to `synth_panel/prompts.py`.
Both CLI and MCP import from there.

### AN-6: Single session per panelist
**Effort:** 2h  
**Dependencies:** None

Architect Decision 1: change `_run_panelist()` in orchestrator.py to create one
`Session()` per panelist, accumulating messages across questions. Currently creates
a fresh session per question, losing conversational context.

This is a behavior change — later questions will see prior Q&A as context. This
mirrors real focus groups and is required for Phase 2 (multi-round instruments).

---

## Phase 2: 0.2.0

- Wire REPL input to runtime (3h) — then add REPL back to README
- Instrument schema v2 design (4h design doc)
- Plugin integration tests (3h)
- Pack validation + CLI commands (5h)
- Bundled starter persona packs (4h)

---

## Phase 3: 0.5.0

- Multi-round instrument implementation (20h)
- Expose v2 instruments in MCP (3h)
- Community pack ecosystem (2h)
- Shared result formatting module (2h)

---

## Bead Summary

### Go-Public beads:
| Bead | Title | Priority | Deps |
|------|-------|----------|------|
| sp-cs7 | README rewrite — honest, no stubs, all 4 providers | P0 | — |
| sp-p4z | Remove login/logout stub commands from CLI | P1 | — |
| sp-5io | Fix test_unreachable_base_url (dummy API key) | P1 | — |
| sp-ybh | Clean MCP tool naming (remove tool_ prefix) | P1 | — |
| sp-9uv | Add instrument version field to schema | P1 | — |
| sp-7b5 | pyproject.toml cleanup (authors, remove pytest-httpx) | P2 | — |
| sp-ogb | Tag and release v0.1.0 | P0 | all above + sp-s6o |

### Existing beads folded into GP-1:
| Bead | Status after GP-1 |
|------|-------------------|
| sp-v1m | Closed (deps claim fixed in README rewrite) |
| sp-34q | Closed (REPL demo removed in README rewrite) |
| sp-2h7 | Closed (Gemini added to table in README rewrite) |

### Existing beads unchanged:
| Bead | Phase |
|------|-------|
| sp-s6o | Go-Public (GP-3, fix test mocking) |
| sp-t4h | Already done (LICENSE exists) |
| sp-50q | Gas Town infra, not synth-panel scope |
