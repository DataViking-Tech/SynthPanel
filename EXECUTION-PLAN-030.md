# synth-panel 0.3.0 — Execution Plan

> PM: synthpanel/crew/pm | Date: 2026-04-04  
> Inputs: 0.3.0-VISION.md, ARCHITECTURE-REVIEW-030.md

The 0.3.0 "Synthesis Release" turns synth-panel from "run N prompts in parallel"
into "get a research finding." Three features: panel synthesis (headline),
conditional follow-ups (first branching primitive), comparative output (CLI wow).

All three are architecturally clean additions. No rewrites. Estimated total:
19-25 hours.

---

## Scope Boundaries

| Version | What's in | What's out |
|---------|-----------|------------|
| **0.3.0** | Panel synthesis, conditional follow-ups (`response_contains`/`always`/`never`), comparative output | `response_sentiment` condition, multi-round conversations, persona interaction |
| **0.4.0** | `response_sentiment` (LLM-based), multi-round conversations | Full branching instruments |
| **0.5.0** | Full branching instruments, conversation trees | Web UI, auto-generated personas |

---

## Execution Order

F1 (synthesis) and F2 (conditions) can be developed **in parallel** by different
polecats. F3 (comparative output) depends on F1 (needs synthesis data for summary
row).

**If time runs short:** Ship synthesis + conditions without comparative output.
The demo works without the table; it doesn't work without the synthesis.

```
  F2: Conditions ──────────────┐
  (conditions.py + orchestrator │
   wiring, 5-7h)               │
                                ├──► F3: Comparative Output
  F1: Synthesis ───────────────┤    (output formatting, 4-6h)
  (synthesis.py + CLI/MCP      │
   integration, 8-10h)         │
                                │
                           Integration
                           testing (~2h)
```

---

## Feature 1: Panel Synthesis (Headline)

### F1-A: Core synthesis module
**Effort:** 5-6h  
**Dependencies:** None  
**Parallelizable with:** F2-A, F2-B

Create `synth_panel/synthesis.py` with `synthesize_panel()` function.

**What to build:**
1. `SynthesisResult` dataclass with fields: `summary`, `themes`, `agreements`,
   `disagreements`, `surprises`, `recommendation`, `usage`, `cost`
2. `synthesize_panel(client, panelist_results, questions, model?, custom_prompt?)` function
3. Default synthesis prompt in `prompts.py` — takes structured panelist responses,
   produces the `SynthesisResult` schema
4. Use `StructuredOutputEngine` (tool-use forcing) to guarantee JSON output schema
5. Default model: `sonnet` (quality matters for synthesis, regardless of panelist model)
6. Version the synthesis prompt: include `synthesis_prompt_version` in result

**Acceptance criteria:**
- `synthesize_panel()` returns a valid `SynthesisResult` given mock panelist data
- Structured output engine produces parseable JSON, not free text
- Unit tests with mocked LLM client (no real API calls)
- Synthesis cost tracked separately from panelist cost

### F1-B: CLI + MCP integration for synthesis
**Effort:** 3-4h  
**Dependencies:** F1-A  
**Parallelizable with:** F2-B (after F1-A lands)

Wire synthesis into both surfaces.

**CLI changes:**
- `cli/parser.py`: Add `--no-synthesis` flag, `--synthesis-model` option,
  `--synthesis-prompt` option to `panel run` subparser
- `cli/commands.py`: Call `synthesize_panel()` after `run_panel_parallel()` returns.
  Include synthesis in output. Display synthesis cost separately.

**MCP changes:**
- `mcp/server.py`: Add `synthesis` (bool, default true), `synthesis_model`,
  `synthesis_prompt` params to `run_panel` and `run_quick_poll`
- After orchestrator returns, call `synthesize_panel()` if synthesis enabled
- Add `synthesis` field to return dict. Add `panelist_cost` field alongside
  existing `total_cost` (architect recommendation for backward compat)

**Return contract (additive, not breaking):**
```json
{
  "...existing fields...",
  "panelist_cost": "$0.03",
  "synthesis": {
    "summary": "...",
    "themes": [...],
    "agreements": [...],
    "disagreements": [...],
    "surprises": [...],
    "recommendation": "...",
    "model": "sonnet",
    "usage": {...},
    "cost": "$0.01",
    "prompt_version": 1
  },
  "total_cost": "$0.04"
}
```

When synthesis is skipped: `synthesis` is `null`, `total_cost` equals `panelist_cost`.

**Acceptance criteria:**
- `synth-panel panel run --personas p.yaml --instrument s.yaml` includes synthesis
- `--no-synthesis` skips the synthesis call
- `--synthesis-model opus` uses opus for synthesis
- MCP `run_panel` returns synthesis by default
- MCP `run_panel` with `synthesis: false` omits it
- `total_cost` includes synthesis; `panelist_cost` excludes it
- Tests for both CLI and MCP paths

---

## Feature 2: Conditional Follow-ups

### F2-A: Condition evaluation module
**Effort:** 2-3h  
**Dependencies:** None  
**Parallelizable with:** F1-A, F1-B

Create `synth_panel/conditions.py`.

**What to build:**
1. `evaluate_condition(condition: str, response_text: str) -> bool`
   - `"always"` → True (default when condition omitted)
   - `"never"` → False
   - `"response_contains: <keyword>"` → case-insensitive substring match
   - Unknown condition types → True (forward-compatible)
2. `normalize_follow_up(follow_up: str | dict) -> dict`
   - String → `{"text": string, "condition": "always"}`
   - Dict → pass through (must have `text` key)
3. For structured responses (dict/JSON from `response_schema`), serialize to string
   before condition evaluation. Document this behavior.

**Design for extensibility:** The architect says design with async evaluation in mind.
Use an evaluator registry pattern so `response_sentiment` (LLM-based) can be added
in 0.4.0 without refactoring:
```python
EVALUATORS: dict[str, Callable[[str, str], bool]] = {
    "response_contains": _eval_contains,
    "always": lambda _, __: True,
    "never": lambda _, __: False,
}
```

**Acceptance criteria:**
- All three condition types work correctly
- String follow-ups normalize to dict format
- Unknown conditions default to `always` (forward-compat)
- Structured responses (JSON dicts) are handled
- Unit tests cover all paths including edge cases

### F2-B: Orchestrator wiring for conditions
**Effort:** 3-4h  
**Dependencies:** F2-A  
**Parallelizable with:** F1-B (after F2-A lands)

Wire conditions into the orchestrator's follow-up loop.

**What to change:**
- `orchestrator.py` (~line 341-354): In the follow-up loop, normalize each
  follow-up via `normalize_follow_up()`, then call `evaluate_condition()` against
  the main question's response text. Skip follow-ups where condition is False.
- No changes needed in `mcp/server.py` or `cli/commands.py` — questions pass
  through to orchestrator as-is.

**Backward compatibility:** Existing instruments with string follow-ups (no conditions)
continue to work — `normalize_follow_up()` converts them to `{text: ..., condition: "always"}`.
No version gating required (architect confirmed: conditions are additive, not breaking).

**Acceptance criteria:**
- Existing instruments with string follow-ups work unchanged
- Instruments with `condition: "response_contains: yes"` only fire when response
  contains "yes"
- `condition: "never"` follow-ups are skipped
- Omitted condition defaults to "always"
- Integration test with a multi-question instrument using mixed conditions

---

## Feature 3: Comparative Output

### F3-A: Comparison matrix builder + renderer
**Effort:** 4-6h  
**Dependencies:** F1-B (needs synthesis for summary row)  
**Parallelizable with:** Nothing (final feature)

**What to build:**

1. **Matrix builder** in `cli/output.py`:
   - Pivot `PanelistResult` list into question×persona matrix
   - Structure: `{personas: [names], questions: [{text, responses: {name: text}}]}`
   - Truncate responses to fit terminal width (~30 chars with ellipsis)

2. **Terminal table renderer** in `cli/output.py`:
   - ASCII table (no box-drawing — works everywhere, CPO didn't specify fancy)
   - Personas as columns, questions as rows
   - Summary row at bottom when synthesis is present
   - Auto-detect terminal width (default 80), truncate columns proportionally
   - Cap at ~6 persona columns; for larger panels, fall back to vertical layout

3. **CLI integration:**
   - `cli/parser.py`: Add `--compare` flag and `compare` to `--output-format` choices
   - `cli/commands.py`: Pass compare flag to output

4. **JSON integration:**
   - When `--output-format json` + `--compare`: add `comparison` field with matrix structure
   - MCP `run_panel`: optionally include `comparison` field (low priority — agents
     rarely need a formatted table, but the structured matrix is useful)

**No new dependencies.** String formatting + box-drawing characters only.

**Acceptance criteria:**
- `synth-panel panel run --compare` renders a readable table
- Table includes synthesis summary row when synthesis is enabled
- `--output-format json` with `--compare` includes `comparison` field
- Tables handle 3-6 personas gracefully
- Tables degrade gracefully on narrow terminals
- ASCII rendering works on all platforms

---

## Bead Summary

| Bead | Title | Priority | Effort | Parallel? |
|------|-------|----------|--------|-----------|
| sp-wdz | Panel synthesis: core module (synthesis.py) | P0 | 5-6h | Yes (with F2) |
| sp-oby | Panel synthesis: CLI + MCP integration | P0 | 3-4h | After sp-wdz |
| sp-id2 | Conditional follow-ups: condition evaluation module | P1 | 2-3h | Yes (with F1) |
| sp-0gt | Conditional follow-ups: orchestrator wiring | P1 | 3-4h | After sp-id2 |
| sp-l5i | Comparative output: matrix builder + renderer | P2 | 4-6h | After sp-oby |
| — | Integration testing across all three features | P1 | 2h | After all above |

**Critical path:** F1-A → F1-B → F3-A (12-16h)  
**Parallel path:** F2-A → F2-B (5-7h, runs alongside F1)

---

## Deferred to 0.4.0+

- `response_sentiment` condition type (needs LLM-based classifier — architect recommends deferring)
- Multi-round conversations (different orchestrator model)
- Persona interaction/debate
- Auto-generated personas
- Web UI / dashboard

---

## Remaining 0.2.0-era beads

These are still open and should be resolved before or alongside 0.3.0 work:

| Bead | Title | Note |
|------|-------|------|
| sp-ogb | Tag and release v0.1.0 | Needs to happen before 0.3.0 work begins |
| sp-2h7 | README provider table missing Gemini | Quick fix, bundle with any commit |
| sp-s6o | test_alias_is_resolved_in_send hits live API | Test fix |
| sp-5qg | test_unreachable_base_url fails | Test fix |
