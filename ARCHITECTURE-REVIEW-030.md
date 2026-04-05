# synth-panel 0.3.0 Architecture Feasibility Review

**Author:** synthpanel/crew/architect | **Date:** 2026-04-04  
**Inputs:** 0.3.0-VISION.md, current codebase (post-0.2.0, 31 commits since last review)

---

## Executive Summary

All three 0.3.0 features are feasible without architectural rewrites. The 0.2.0 foundation (single session per panelist, shared prompt builders, structured output wiring) was specifically designed to support this work. Estimated total effort: **18-24 hours**. One contract change (synthesis in `run_panel` return) needs careful handling — it's additive, not breaking, if done right.

---

## Feature 1: Panel Synthesis

**CPO priority: HEADLINE FEATURE**  
**Verdict: FEASIBLE — clean insertion point, additive contract change**  
**Effort: 8-10h**

### Where it fits architecturally

The synthesis is a post-processing step after all panelists complete. The natural insertion point is between the orchestrator returning `list[PanelistResult]` and the result being serialized/saved.

**Two options for placement:**

| Option | Location | Pros | Cons |
|--------|----------|------|------|
| A | Inside `_run_panel_sync()` (mcp/server.py:69-104) | MCP and CLI share the synthesis path | Couples synthesis to the MCP module; CLI would need to call through MCP internals or duplicate |
| B | New `synth_panel/synthesis.py` module, called by both CLI and MCP | Clean separation, testable in isolation, both surfaces invoke it | Extra module |

**Recommendation: Option B.** Create `synth_panel/synthesis.py` with a `synthesize_panel()` function. Both `cli/commands.py:handle_panel_run()` and `mcp/server.py:_run_panel_sync()` call it after `run_panel_parallel()` returns. This mirrors the pattern already established with `prompts.py` — shared logic extracted to a common module.

### Implementation sketch

```python
# synth_panel/synthesis.py

@dataclass
class SynthesisResult:
    summary: str                    # The synthesized finding
    themes: list[str]               # Identified themes
    agreements: list[str]           # Points of consensus
    disagreements: list[str]        # Points of divergence
    surprises: list[str]            # Unexpected insights
    recommendation: str | None      # Optional actionable recommendation
    usage: TokenUsage               # Tokens used for synthesis call
    cost: str                       # Formatted USD cost

def synthesize_panel(
    client: LLMClient,
    panelist_results: list[PanelistResult],
    questions: list[dict[str, Any]],
    model: str | None = None,       # Defaults to sonnet (not haiku)
    custom_prompt: str | None = None,
) -> SynthesisResult:
    ...
```

### The synthesis prompt

The default synthesis prompt should:
1. Receive all panelist responses as structured input (not raw conversation transcripts)
2. Use tool-use forcing via `StructuredOutputEngine` to guarantee the output schema
3. Be overridable via `custom_prompt` parameter (CLI: `--synthesis-prompt`, MCP: `synthesis_prompt` field)

Using `StructuredOutputEngine` here is critical — it guarantees the synthesis output is parseable JSON, not free text. The engine is already proven and wired through the codebase.

### Model selection

CPO says synthesis model can differ from panelist model. This is correct — use sonnet for synthesis (quality matters), haiku for panelists (cost matters). Implementation:

- New parameter: `synthesis_model` (CLI: `--synthesis-model`, MCP: `synthesis_model` field)
- Default: `sonnet` (regardless of panelist model)
- Opt-out: `--no-synthesis` / `synthesis: false` in MCP

### Contract change impact

**Current `run_panel` return (MCP):**
```json
{
  "result_id": "...",
  "model": "...",
  "persona_count": 3,
  "question_count": 2,
  "total_cost": "$0.03",
  "total_usage": {...},
  "results": [...]
}
```

**Proposed addition:**
```json
{
  ...existing fields...
  "synthesis": {
    "summary": "3 of 4 personas would pay...",
    "themes": ["price sensitivity", "integration needs"],
    "agreements": ["onboarding is too complex"],
    "disagreements": ["self-host vs SaaS"],
    "surprises": ["developer segment rejects SaaS entirely"],
    "recommendation": "Lead with integration story, offer self-host option",
    "model": "sonnet",
    "usage": {...},
    "cost": "$0.01"
  },
  "total_cost": "$0.04"  // Now includes synthesis cost
}
```

**This is additive, not breaking.** Existing consumers that don't read `synthesis` are unaffected. The `total_cost` and `total_usage` fields now include synthesis overhead — this is the correct behavior (total should mean total).

**When synthesis is skipped** (`--no-synthesis` or `synthesis: false`): the `synthesis` field is `null`. Consumers should handle this.

### What changes per file

| File | Change | Lines affected |
|------|--------|----------------|
| `synth_panel/synthesis.py` | **NEW** — synthesizer module | ~120 lines |
| `mcp/server.py` | Add synthesis params to `run_panel`/`run_quick_poll`, call `synthesize_panel()` after orchestrator, merge into result | ~30 lines |
| `cli/commands.py` | Add `--no-synthesis`/`--synthesis-model` args, call `synthesize_panel()`, include in output | ~25 lines |
| `cli/parser.py` | Add synthesis flags to panel run subparser | ~5 lines |
| `mcp/data.py` | No change — persistence is schema-agnostic (saves whatever dict it receives) | 0 |
| `prompts.py` | Add default synthesis prompt template | ~20 lines |
| Tests | `test_synthesis.py` (new), updates to `test_mcp_server.py`, `test_cli.py` | ~100 lines |

### Risks

1. **Synthesis cost surprise.** A sonnet call on top of haiku panelists could dominate total cost. Mitigation: display synthesis cost separately in output, document the default model choice.
2. **Synthesis quality varies by model.** Haiku synthesis will be shallow; opus would be expensive. Mitigation: default to sonnet, let users override.
3. **Synthesis prompt engineering.** The default prompt needs to produce consistently structured output. Mitigation: use `StructuredOutputEngine` with explicit schema, not free text.

---

## Feature 2: Conditional Follow-ups

**CPO priority: FIRST BRANCHING PRIMITIVE**  
**Verdict: FEASIBLE — minimal orchestrator change, backward-compatible**  
**Effort: 5-7h**

### Where it fits architecturally

Follow-ups are currently handled in `orchestrator.py:341-354`. The loop iterates `question.get("follow_ups", [])` and runs each as a turn. Adding conditions means filtering this list based on the main question's response.

### Condition types

CPO proposes four condition types: `response_contains`, `response_sentiment`, `always`, `never`.

| Condition | Feasibility | Implementation |
|-----------|-------------|----------------|
| `response_contains: <keyword>` | Trivial | Case-insensitive substring match on response text |
| `response_sentiment: positive/negative/neutral` | Needs LLM call or heuristic | **See below** |
| `always` | Trivial | Default behavior (backward-compatible) |
| `never` | Trivial | Skip follow-up |

**`response_sentiment` concern:** This requires either (a) a keyword heuristic (fragile, inaccurate) or (b) an extra LLM call per follow-up decision (expensive, slow). Neither is good for v0.3.0.

**Recommendation: Ship `response_contains`, `always`, `never` in 0.3.0. Defer `response_sentiment` to 0.4.0** where it can be implemented properly as an LLM-based classifier with caching. Three condition types are enough to demonstrate the branching primitive. The CPO's own example only uses `response_contains`.

### Follow-up schema change

**Current (flat strings):**
```yaml
follow_ups:
  - "What price feels right?"
  - "What would change your mind?"
```

**Proposed (objects with optional condition):**
```yaml
follow_ups:
  - text: "What price feels right?"
    condition: "response_contains: yes"
  - text: "What would change your mind?"
    condition: "response_contains: no"
```

**Backward compatibility:** If a follow-up is a string (not a dict), treat it as `{text: <string>, condition: "always"}`. This means existing instruments work unchanged.

### Instrument version

This is an instrument v2 feature. The `version` field already exists (defaulting to 1). Conditional follow-ups should require `version: 2` or higher. Instruments with `version: 1` that contain condition fields should warn, not error — graceful upgrade path.

Actually, reconsider: the CPO says "omitting `condition` means `always` (backward-compatible)." This means v1 instruments with the new follow-up dict format would work. **Don't gate on version number.** The version field is for future breaking changes (like full branching in 0.5.0). Conditional follow-ups are purely additive.

### Implementation location

```python
# New: synth_panel/conditions.py (~40 lines)

def evaluate_condition(condition: str, response_text: str) -> bool:
    """Evaluate a follow-up condition against a panelist response."""
    if not condition or condition == "always":
        return True
    if condition == "never":
        return False
    if condition.startswith("response_contains:"):
        keyword = condition.split(":", 1)[1].strip().lower()
        return keyword in response_text.lower()
    # Unknown condition type — default to always (forward-compatible)
    return True

def normalize_follow_up(follow_up: str | dict) -> dict:
    """Normalize string follow-ups to dict format."""
    if isinstance(follow_up, str):
        return {"text": follow_up, "condition": "always"}
    return follow_up
```

### What changes per file

| File | Change | Lines affected |
|------|--------|----------------|
| `synth_panel/conditions.py` | **NEW** — condition evaluation | ~40 lines |
| `orchestrator.py` | Import conditions module, normalize follow-ups, filter by condition before executing | ~15 lines in `_run_panelist()` |
| `prompts.py` | No change — follow-up text extraction unchanged | 0 |
| `mcp/server.py` | No change — questions pass through to orchestrator | 0 |
| `cli/commands.py` | No change — instrument loading already passes dicts through | 0 |
| Tests | `test_conditions.py` (new), updates to `test_orchestrator.py` | ~60 lines |

### Risks

1. **Condition evaluation on structured responses.** If a main question uses `response_schema` (structured output), the response is a dict, not a string. `response_contains` needs to operate on the serialized JSON or a specific field. **Mitigation:** Serialize structured responses to string for condition evaluation. Document this behavior.
2. **Follow-up format migration.** Existing YAML files with string follow-ups must continue to work. **Mitigation:** `normalize_follow_up()` handles both formats.

---

## Feature 3: Comparative Output

**CPO priority: THIRD (CLI wow factor)**  
**Verdict: FEASIBLE — pure output formatting, no core changes**  
**Effort: 4-6h**

### Where it fits architecturally

This is entirely an output formatting concern. The data needed (all responses, grouped by question and persona) already exists in the `PanelistResult` list. The work is:
1. A new output formatter that pivots the data into a question×persona matrix
2. Terminal table rendering
3. Integration with synthesis (summary row)

### Implementation approach

**No new dependencies.** The CPO spec says "use built-in formatting, no new deps." Terminal tables can be rendered with string formatting and box-drawing characters. The table doesn't need to be fancy — it needs to be readable.

**Column width strategy:** Truncate persona responses to fit terminal width. Show first ~30 chars with ellipsis. Full responses remain available in JSON output. The table is a summary view, not the complete data.

### Output format integration

Two ways to invoke:
- `--output-format compare` (new format alongside text/json/ndjson)
- `--compare` flag (shorthand, implies text output with table)

For JSON output with `--compare`: add a `comparison` field containing the matrix structure. This lets agents consume the pivoted data without rebuilding it.

```json
{
  ...existing fields...
  "comparison": {
    "personas": ["Sarah", "Marcus", "Priya"],
    "questions": [
      {
        "text": "Would you pay?",
        "responses": {
          "Sarah": "Yes, if integrations...",
          "Marcus": "No, unless self-host...",
          "Priya": "Yes, already would"
        }
      }
    ]
  }
}
```

### What changes per file

| File | Change | Lines affected |
|------|--------|----------------|
| `cli/output.py` | Add `COMPARE` format, table renderer, matrix builder | ~80 lines |
| `cli/parser.py` | Add `--compare` flag, `compare` to format choices | ~5 lines |
| `cli/commands.py` | Pass compare flag through to output | ~5 lines |
| `mcp/server.py` | Optionally include `comparison` field in result | ~10 lines |
| Tests | `test_output.py` updates, new comparison tests | ~40 lines |

### Risks

1. **Terminal width.** Wide tables break on narrow terminals. **Mitigation:** Detect terminal width (or default to 80), truncate columns proportionally.
2. **Many personas or long questions.** A 10-persona panel won't fit in a table. **Mitigation:** Cap at ~6 columns, overflow to vertical layout for larger panels. Document the limit.
3. **Unicode box-drawing on Windows.** Some Windows terminals don't render box-drawing chars. **Mitigation:** Fall back to ASCII table chars (`+`, `-`, `|`) if box-drawing fails. Or just use ASCII by default — simpler, works everywhere.

---

## Dependency Order

```
                    ┌─────────────────────┐
                    │  conditions.py (F2)  │
                    │  (no dependencies)   │
                    └──────────┬──────────┘
                               │
┌──────────────────┐           │           ┌─────────────────────┐
│  synthesis.py    │           │           │  output.py compare  │
│  (F1 — headline) │           │           │  (F3 — formatting)  │
│  needs: prompts, │           │           │  needs: nothing new  │
│  structured out  │           │           │                     │
└────────┬─────────┘           │           └──────────┬──────────┘
         │                     │                      │
         └─────────────┬───────┘                      │
                       │                              │
              ┌────────▼──────────┐                   │
              │  orchestrator.py  │                   │
              │  wire conditions  │                   │
              │  (F2 integration) │                   │
              └────────┬──────────┘                   │
                       │                              │
              ┌────────▼──────────────────────────────▼─┐
              │  CLI + MCP integration                   │
              │  (synthesis call, compare output,        │
              │   conditional follow-ups pass-through)   │
              └──────────────────────────────────────────┘
```

**Recommended execution order:**
1. **F2: Conditional follow-ups** (smallest, unblocks instrument v2 story)
2. **F1: Panel synthesis** (headline, highest value)
3. **F3: Comparative output** (depends on synthesis for summary row)

F1 and F2 can be developed in parallel by different polecats. F3 should follow F1 (needs synthesis data to render summary row).

---

## Contract Change Assessment

The bead specifically calls out: "synthesis changes the `run_panel` return contract."

### Impact analysis

| Consumer | Impact | Breaking? |
|----------|--------|-----------|
| MCP tool callers (agents) | New `synthesis` field in response | **No** — additive field, `null` when skipped |
| CLI users | New synthesis section in text output | **No** — additional output |
| `save_panel_result()` persistence | Stores whatever dict it receives | **No** — schema-agnostic |
| `get_panel_result()` retrieval | Returns stored dict including synthesis | **No** — pass-through |
| `list_panel_results()` metadata | Unchanged (only returns id, model, counts) | **No** |
| `total_cost` / `total_usage` | Now includes synthesis overhead | **Semantically correct** but numerically different. Agents budgeting on total_cost may need adjustment. |

### The `total_cost` concern

Before synthesis: `total_cost = sum(panelist costs)`.  
After synthesis: `total_cost = sum(panelist costs) + synthesis cost`.

This is semantically correct (total should mean total) but could surprise agents that budget based on `total_cost`. **Mitigation:** Add `synthesis.cost` as a separate field so consumers can subtract it if needed. Also add `panelist_cost` field for the pre-synthesis total.

**Proposed:**
```json
{
  "total_cost": "$0.04",       // Everything
  "panelist_cost": "$0.03",    // Just panelists
  "synthesis": {
    "cost": "$0.01",           // Just synthesis
    ...
  }
}
```

This is fully backward-compatible. Existing consumers reading `total_cost` get the true total. New consumers can distinguish.

---

## Architectural Concerns

### Concern 1: Synthesis adds latency

A sonnet call after all panelists complete adds 2-5 seconds. For CLI users this is fine (they're waiting anyway). For agent workflows, it could matter.

**Mitigation:** `--no-synthesis` / `synthesis: false` for latency-sensitive workflows. Document that synthesis adds one LLM call.

### Concern 2: Synthesis prompt stability

The default synthesis prompt is a product-level decision, not a technical one. If it changes between versions, results won't be reproducible.

**Mitigation:** Version the synthesis prompt. Include `synthesis_prompt_version` in the result. This is a 2-line addition but enables reproducibility.

### Concern 3: Condition evaluation is synchronous and in-process

Conditions evaluate after each main question response, inside the panelist thread. This is fine for `response_contains` (microseconds). If `response_sentiment` is added later (requires LLM call), it would add latency per-question per-panelist.

**Mitigation:** The `conditions.py` module should be designed with async evaluation in mind from the start, even if v0.3.0 only uses synchronous evaluators. Make `evaluate_condition()` accept a registry of evaluators, so LLM-based evaluators can be added without refactoring.

### Concern 4: No decisions required NOW

Unlike the 0.2.0 review which flagged 3 decisions to lock before launch, **0.3.0 has no lock-now decisions.** All three features are additive. The contract change is backward-compatible. The condition system is extensible. The synthesis module is isolated. This is a sign that the 0.2.0 foundation was designed correctly.

---

## Effort Summary

| Feature | Effort | Risk | Parallelizable |
|---------|--------|------|----------------|
| F1: Panel Synthesis | 8-10h | Medium (prompt engineering, cost visibility) | Yes (with F2) |
| F2: Conditional Follow-ups | 5-7h | Low (straightforward filter logic) | Yes (with F1) |
| F3: Comparative Output | 4-6h | Low (pure formatting) | After F1 |
| **Total** | **17-23h** | | |

Add ~2h for integration testing across all three features together. **Grand total: 19-25h.**

---

## Recommendation

Ship it. All three features are architecturally clean additions. The 0.2.0 foundation supports them without modification. The contract change is additive. The CPO's Show HN demo is achievable.

**One process note:** F1 (synthesis) should be the first feature merged and tested, even though F2 (conditions) could be developed first. The synthesis is the headline — if time runs short, ship synthesis + conditions without comparative output. The demo still works without the table; it doesn't work without the synthesis.
