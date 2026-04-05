# synth-panel 0.4.0 — Execution Plan

> PM: synthpanel/crew/pm | Date: 2026-04-05  
> Inputs: 0.4.0-VISION.md, ARCHITECTURE-REVIEW-040.md

The 0.4.0 "Conversation Release" makes synth-panel conversational. Multi-round
instruments are the headline — the feature that makes synth-panel impossible to
replicate with manual prompting. `response_sentiment` completes the conditions
system. Panel resume/extend makes the agent surface iterative.

Total effort: 30-40 hours. Two architectural decisions locked by the architect.

---

## Pre-040 Blocker

**PR #22 (sp-1d3): Wire conditions into orchestrator** — OPEN, NOT YET MERGED.

`conditions.py` exists with `evaluate_condition()` and `normalize_follow_up()` but
the orchestrator never calls them. Follow-ups fire unconditionally. This is a 0.3.0
bug that must merge before 0.4.0 work begins. The PR exists and passed tests (307
passed). **Mayor: please merge PR #22 or assign someone to review it.**

---

## Architectural Decisions (Locked)

Per architect review, two decisions are locked:

1. **Session reuse: modify `_run_panelist()` signature** (Option A). Add
   `session: Session | None = None` parameter. Both multi-round and panel extend
   depend on this. No registry pattern — the orchestrator manages the mapping.

2. **Output format: backward-compatible** (Option B). Single-round instruments
   keep the flat format (`results`, `synthesis`). Multi-round uses `rounds` format.
   Flat format deprecated in 0.4.0 docs, removed in 0.5.0.

## CPO Decisions (Locked)

1. **Template syntax:** `synthesis.*` only (flattened as `theme_0`, `recommendation`, etc.)
2. **Budget:** Per-run total, not per-round. Expose `round_cost` in output.
3. **Extend:** Appends to existing result (overwrite with pre-extend snapshot)
4. **Synthesis model escalation:** Intermediate syntheses use panelist model,
   final synthesis uses sonnet (or `--synthesis-model` override)

---

## Scope Boundaries

| Version | In | Out |
|---------|-----|-----|
| **0.4.0** | Multi-round (linear), `response_sentiment`, panel extend | Branching rounds, persona interaction, auto-generated follow-ups |
| **0.5.0** | Branching instruments (conditional round routing), conversation trees, deprecate flat output format | Web UI, hosted service |

---

## Execution Order

Three phases. Phase 1 items are parallelizable. Phase 2 depends on Phase 1.
Phase 3 depends on Phase 1 session reuse but can develop in parallel with Phase 2.

```
Phase 1 — Foundation (parallelizable):
├── F1-A: Session reuse in orchestrator
├── F1-B: Template engine (templates.py)
├── F2-A: response_sentiment condition evaluator
└── F2-B: Instrument v2 parser (round definitions)

Phase 2 — Multi-Round (depends on Phase 1):
├── F1-C: rounds.py — multi-round orchestrator
├── F1-D: CLI + MCP integration for multi-round
└── F1-E: Final cross-round synthesis + examples

Phase 3 — Panel Extend (depends on F1-A):
├── F3-A: Session persistence (save/load per panelist)
└── F3-B: extend_panel MCP tool + CLI command
```

**If time runs short:** Ship multi-round + sentiment without panel extend. The
multi-round demo is the Show HN material. Extend is powerful but not the headline.

---

## Phase 1: Foundation

### F1-A: Session reuse in orchestrator
**Effort:** 2-3h  
**Dependencies:** PR #22 merged (conditions wiring)  
**Parallelizable with:** F1-B, F2-A, F2-B

Modify `_run_panelist()` and `run_panel_parallel()` to accept and return sessions.

**What to change in `orchestrator.py`:**
1. Add `session: Session | None = None` param to `_run_panelist()`. If provided,
   reuse instead of creating new. If None, create as today.
2. Add `sessions: dict[str, Session] | None = None` param to `run_panel_parallel()`.
   Map persona names to sessions, pass to `_run_panelist()`.
3. Return sessions from `run_panel_parallel()`:
   `-> tuple[list[PanelistResult], WorkerRegistry, dict[str, Session]]`

**This is ~15 lines of change** but it's the foundation both multi-round and panel
extend depend on. Lock the signature before anything else builds on it.

**Acceptance criteria:**
- `run_panel_parallel()` with no `sessions` param works as before (backward compat)
- `run_panel_parallel()` with `sessions` param reuses provided sessions
- Sessions are returned in the result tuple
- Existing tests pass unchanged
- New tests verify session reuse (persona remembers prior conversation)

### F1-B: Template engine (templates.py)
**Effort:** 3-4h  
**Dependencies:** None  
**Parallelizable with:** F1-A, F2-A, F2-B

Create `synth_panel/templates.py` (~60 lines).

**What to build:**
1. `build_template_context(synthesis: SynthesisResult) -> dict[str, str]`
   - Flatten synthesis into: `summary`, `recommendation`, `theme_0`, `theme_1`, ...,
     `agreement_0`, ..., `disagreement_0`, ..., `surprise_0`, ...
2. `render_questions(questions: list[dict], context: dict[str, str]) -> list[dict]`
   - Deep copy questions, render `{theme_0}` etc. in `text` fields
   - Use `string.Formatter` with custom `format_field` that returns literal
     placeholder on `KeyError` (safe failure — no eval, no crash)
3. `validate_template(text: str, context: dict[str, str]) -> list[str]`
   - Return list of unresolvable keys (warnings, not errors)
   - Log warnings when rendering — persona gets the literal placeholder

**Acceptance criteria:**
- Templates with valid keys render correctly
- Missing keys render as literal `{key_name}` (not crash, not empty)
- `validate_template()` catches unresolvable keys
- Nested/complex format strings don't trigger eval or injection
- Unit tests for all edge cases (empty synthesis, missing themes, etc.)

### F2-A: `response_sentiment` condition evaluator
**Effort:** 4-6h  
**Dependencies:** PR #22 merged  
**Parallelizable with:** F1-A, F1-B, F2-B

Add LLM-based sentiment classification to `conditions.py`.

**What to change:**
1. Add `client: LLMClient | None = None` optional param to `evaluate_condition()`.
   Default None = backward compatible (non-LLM evaluators ignore it).
2. Add `_eval_sentiment(target: str, response_text: str, client: LLMClient) -> bool`
   - Makes a haiku call: "Classify this response as positive/negative/neutral"
   - Returns True if classification matches target (e.g., `"negative"`)
   - If no client provided, default to True (graceful degradation for tests)
3. Add `sentiment_cache: dict[str, str] | None = None` param to `evaluate_condition()`.
   Same response isn't classified twice.
4. Update `orchestrator.py` to pass `client` and shared cache to `evaluate_condition()`.

**Cost:** ~$0.003 per panel (10 haiku calls worst case). Negligible.

**Acceptance criteria:**
- `condition: "response_sentiment: negative"` fires on negative responses
- `condition: "response_sentiment: positive"` fires on positive responses
- Cache prevents duplicate LLM calls for same response text
- Existing non-LLM conditions (`response_contains`, `always`, `never`) unaffected
- Unit tests with mocked LLM client
- Integration test with real sentiment classification (mark as acceptance)

### F2-B: Instrument v2 parser (round definitions)
**Effort:** 3-4h  
**Dependencies:** None  
**Parallelizable with:** F1-A, F1-B, F2-A

Parse instrument YAML with `rounds` key.

**What to build:**
1. In instrument loading (likely `cli/commands.py`), detect `rounds` key:
   - If `rounds` present: parse as v2 multi-round instrument
   - If only `questions` present: wrap as single round `[{name: "default", questions: [...]}]`
   - Backward compatible — existing instruments work unchanged
2. Validate round definitions:
   - Each round has `name` (string) and `questions` (list)
   - Optional `depends_on` references an earlier round name (no cycles, no forward refs)
   - Reject invalid `depends_on` with clear error message
3. Create example: `examples/multi-round-study.yaml` demonstrating 3-round instrument
   with template variables

**Acceptance criteria:**
- v1 instruments (flat `questions`) parse correctly as single round
- v2 instruments with `rounds` parse correctly
- Invalid `depends_on` (cycle, forward ref, missing round) raises clear error
- Example YAML validates successfully
- Unit tests for parsing and validation

---

## Phase 2: Multi-Round Orchestration

### F1-C: rounds.py — multi-round orchestrator
**Effort:** 8-10h  
**Dependencies:** F1-A (session reuse), F1-B (template engine), F2-B (instrument parser)  
**Parallelizable with:** F3-A (session persistence)

Create `synth_panel/rounds.py` — the new orchestration layer.

**What to build:**

Data models:
```python
@dataclass
class RoundResult:
    name: str
    results: list[PanelistResult]
    synthesis: SynthesisResult
    usage: TokenUsage
    cost: str

@dataclass
class MultiRoundResult:
    rounds: list[RoundResult]
    final_synthesis: SynthesisResult
    total_usage: TokenUsage
    total_cost: str
    panelist_cost: str
    synthesis_cost: str
```

Core function:
```python
def run_multi_round(
    client: LLMClient,
    personas: list[dict],
    rounds: list[dict],         # Parsed round definitions
    model: str,
    *,
    synthesis_model: str | None = None,       # For intermediate syntheses
    final_synthesis_model: str | None = None,  # For final (default: sonnet)
    response_schema: dict | None = None,
    budget: float | None = None,
) -> MultiRoundResult:
```

**The round loop:**
1. Initialize `sessions: dict[str, Session] = {}`
2. For each round:
   a. If `depends_on` set, get prior round's synthesis
   b. Build template context from prior synthesis
   c. Render questions with template context
   d. Call `run_panel_parallel()` with personas, rendered questions, and `sessions`
   e. Update `sessions` from returned sessions
   f. Call `synthesize_panel()` for this round (use panelist model for intermediate)
   g. Check budget gate — stop if exceeded
3. Call `synthesize_panel()` for final synthesis across all rounds (use `final_synthesis_model`)
4. Return `MultiRoundResult`

**Final synthesis prompt:** New prompt in `prompts.py` that receives all round
syntheses and produces an overall finding. Different from per-round synthesis —
it looks across the entire conversation arc.

**Acceptance criteria:**
- 3-round instrument with template vars executes correctly
- Personas remember prior rounds (session reuse verified)
- Template variables resolve from prior round's synthesis
- Missing template keys render as literals (no crash)
- Budget gate stops execution if exceeded mid-run
- Intermediate syntheses use panelist model, final uses sonnet
- `round_cost` visible in output per round
- Unit tests with mocked LLM (no real API calls)

### F1-D: CLI + MCP integration for multi-round
**Effort:** 4-5h  
**Dependencies:** F1-C (rounds.py)

Wire multi-round into both surfaces.

**CLI changes (`cli/commands.py`):**
- Detect v2 instrument (has `rounds`), call `run_multi_round()` instead of
  `run_panel_parallel()` + `synthesize_panel()`
- Format output: show per-round results + synthesis, then final synthesis
- Existing flags work: `--no-synthesis` skips final synthesis only (intermediate
  syntheses are mandatory for template resolution)
- `--compare` works with multi-round: one table per round + summary

**MCP changes (`mcp/server.py`):**
- `run_panel` detects round-based instruments, calls `run_multi_round()`
- Multi-round response format:
  ```json
  {
    "rounds": [{name, results, synthesis}],
    "final_synthesis": {...},
    "total_cost", "panelist_cost", "synthesis_cost"
  }
  ```
- Single-round instruments keep flat format (backward compat, Decision 2)
- `run_quick_poll` stays single-round only

**Acceptance criteria:**
- `synth-panel panel run --instrument multi-round.yaml` runs all rounds
- Output shows per-round breakdown + final synthesis
- MCP `run_panel` with multi-round instrument returns `rounds` format
- MCP `run_panel` with single-round instrument returns flat format (no breaking change)
- `--compare` renders per-round tables
- Tests for both CLI and MCP paths

### F1-E: Final cross-round synthesis prompt + examples
**Effort:** 2-3h  
**Dependencies:** F1-C

**What to build:**
1. Final synthesis prompt in `prompts.py` — takes all round syntheses, produces
   overall finding that traces the research arc (exploration → probing → validation)
2. Update `examples/multi-round-study.yaml` with a compelling 3-round instrument
3. Add example to README showing multi-round usage (the Show HN demo from CPO vision)
4. Document template syntax in CLAUDE.md: available variables, safe failure behavior

**Acceptance criteria:**
- Final synthesis references findings across rounds, not just the last round
- Example instrument is compelling and demonstrates template variables
- README has working multi-round example
- Template syntax is documented

---

## Phase 3: Panel Resume/Extend

### F3-A: Session persistence (save/load per panelist)
**Effort:** 4-5h  
**Dependencies:** F1-A (session reuse)  
**Parallelizable with:** F1-C, F1-D, F1-E (Phase 2)

**What to build:**

Storage layout:
```
~/.synth-panel/results/
  result-<id>.json                    # Panel result (metadata + responses)
  result-<id>.sessions/              # Session directory (opt-in)
    Sarah_Chen.json                   # Per-persona session
    Marcus_Johnson.json
```

Functions in `mcp/data.py`:
1. `save_panel_sessions(result_id: str, sessions: dict[str, Session]) -> Path`
2. `load_panel_sessions(result_id: str) -> dict[str, Session]`
3. `update_panel_result(result_id: str, updated_data: dict) -> None`
   - Save pre-extend snapshot as `{result_id}.pre-extend.json`
   - Overwrite main result file with extended data

Session saving is opt-in — only when `--save-session` flag or multi-round is used.
`Session` already has `to_dict()`/`from_dict()` — just serialize to JSON.

**Acceptance criteria:**
- Sessions save to disk in the correct directory structure
- Sessions load and restore full conversation history
- `update_panel_result()` creates pre-extend snapshot before overwriting
- Round-trip test: save → load → persona remembers conversation
- Cleanup: old `.pre-extend.json` files don't accumulate indefinitely

### F3-B: extend_panel MCP tool + CLI command
**Effort:** 4-5h  
**Dependencies:** F3-A (session persistence), F1-A (session reuse)

**New MCP tool: `extend_panel`**
```python
async def extend_panel(
    result_id: str,
    questions: list[dict[str, Any]],
    model: str | None = None,
    synthesis: bool = True,
    synthesis_model: str | None = None,
) -> str:
```

Flow:
1. Load existing result by `result_id`
2. Load persisted sessions via `load_panel_sessions()`
3. Extract persona list from existing result
4. Call `run_panel_parallel()` with loaded sessions + new questions
5. Synthesize new round (if synthesis=True)
6. Append round to existing result via `update_panel_result()`
7. Return updated result

**New CLI command: `synth-panel panel extend`**
```bash
synth-panel panel extend <result-id> --questions "Tell me more about pricing"
synth-panel panel extend <result-id> --instrument follow-up.yaml
```

**CLI parser:** Add `panel extend` subcommand with `result-id` positional arg,
`--questions` (inline) or `--instrument` (YAML file), plus standard flags.

**Acceptance criteria:**
- `extend_panel` MCP tool loads sessions and resumes conversation
- Personas remember prior panel context in their responses
- Extended result keeps same `result_id`
- Pre-extend snapshot saved
- CLI `panel extend` works with both `--questions` and `--instrument`
- Error handling: missing result_id, missing sessions, corrupted session file
- Tests for both MCP and CLI paths

---

## Bead Summary

| Bead | Title | Priority | Effort | Phase | Parallel? |
|------|-------|----------|--------|-------|-----------|
| sp-yln | Session reuse in orchestrator | P0 | 2-3h | 1 | Yes |
| sp-tsz | Template engine (templates.py) | P0 | 3-4h | 1 | Yes |
| sp-uqu | response_sentiment condition evaluator | P1 | 4-6h | 1 | Yes |
| sp-65a | Instrument v2 parser — round definitions | P0 | 3-4h | 1 | Yes |
| sp-se3 | Multi-round orchestrator (rounds.py) | P0 | 8-10h | 2 | After Phase 1 |
| sp-9rf | Multi-round CLI + MCP integration | P0 | 4-5h | 2 | After sp-se3 |
| sp-1u1 | Cross-round synthesis prompt + examples + docs | P1 | 2-3h | 2 | After sp-se3 |
| sp-60d | Session persistence (save/load per panelist) | P1 | 4-5h | 3 | With Phase 2 |
| sp-e9n | extend_panel MCP tool + CLI command | P1 | 4-5h | 3 | After sp-60d |

**Critical path:** sp-yln + sp-tsz + sp-65a → sp-se3 → sp-9rf (20-26h)  
**Parallel path A:** sp-uqu (4-6h, runs alongside everything in Phase 1)  
**Parallel path B:** sp-60d → sp-e9n (8-10h, runs alongside Phase 2)

---

## Remaining Pre-0.4.0 Items

| Item | Status | Action |
|------|--------|--------|
| PR #22 — conditions wiring (sp-1d3) | Open, not merged | **Merge before starting 0.4.0** |
| sp-ogb — Tag v0.1.0 | Open | Resolve before 0.4.0 |
| sp-2h7 — Gemini in README table | Open | Quick fix, bundle with any commit |
| sp-s6o — Test hits live API | Open | Test fix |
| sp-5qg — test_unreachable_base_url | Open | Test fix |
