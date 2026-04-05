# synth-panel 0.4.0 Architecture Feasibility Review

**Author:** synthpanel/crew/architect | **Date:** 2026-04-05  
**Inputs:** 0.4.0-VISION.md, current codebase (post-0.3.0 synthesis/conditions/compare work)

---

## Executive Summary

Multi-round instruments are the most significant architectural change since the project's inception. The 0.2.0 foundation (single session per panelist, shared prompt builders) and 0.3.0 additions (synthesis module, conditions module) were designed with this in mind — but multi-round requires a new orchestration layer on top of the existing parallel panelist execution. Panel resume/extend is a natural extension of multi-round. `response_sentiment` is straightforward. Total effort: **30-40 hours**.

Two architectural decisions need to be locked before implementation begins.

---

## CPO Questions — Architect Answers

### Q1: Template syntax scope — `synthesis.*` only, or also per-persona references?

**Answer: `synthesis.*` only for 0.4.0. Agree with CPO.**

Per-persona references (`round[0].results["Sarah"].response`) create a combinatorial explosion in the template engine and make instruments fragile (what if the persona name changes?). The synthesis is the abstraction layer — it distills per-persona responses into structured findings. If an instrument designer needs to reference a specific persona's response, they're working at the wrong abstraction level for a template.

Implementation: Python `str.format_map()` with a flattened synthesis dict. Templates like `{themes_0}` or `{recommendation}` are simple and don't require a custom parser. Bracket indexing (`{synthesis.themes[0]}`) requires either `eval()` (dangerous) or a custom resolver.

**Recommended template syntax:**
```yaml
# Simple — uses str.format_map()
- text: "The panel flagged '{theme_0}' — can you describe a specific instance?"
- text: "The recommendation was '{recommendation}'. Would you switch?"
```

Where the template context is:
```python
{
    "summary": "...",
    "theme_0": synthesis.themes[0],  # Indexed themes
    "theme_1": synthesis.themes[1],
    "agreement_0": synthesis.agreements[0],
    "disagreement_0": synthesis.disagreements[0],
    "surprise_0": synthesis.surprises[0],
    "recommendation": synthesis.recommendation,
}
```

This avoids nested attribute access (`synthesis.themes[0]`), avoids `eval()`, and is trivially implementable. Missing keys render as `{theme_0}` literally (Python's `format_map` with a defaultdict), which is a safe failure mode — the persona gets asked a question with an unresolved template, which is suboptimal but not broken.

### Q2: Round-level cost budgets — per-round or entire run?

**Answer: Entire run. Agree with CPO.**

The `BudgetGate` (cost.py:305-337) currently tracks cumulative tokens against a max budget. It's already designed for total-run budgets. No change needed to the budget mechanism — just ensure the same `BudgetGate` instance is passed through all rounds.

Expose `round_cost` in output per the CPO's recommendation. This is a formatting concern, not a budget mechanism change.

### Q3: `extend_panel` persistence — new result or append?

**Answer: Append, but with caveats. Partially agree with CPO.**

Appending to an existing result is the right UX for agents (one result ID to track). But the current `save_panel_result()` writes a JSON file atomically — it doesn't support incremental append. Two options:

| Option | Approach | Pros | Cons |
|--------|----------|------|------|
| A | Overwrite the existing result file with the extended data | Simple, result ID stays the same | Loses the pre-extension state; not recoverable |
| B | Write new result file, link to parent via `extends: <parent_id>` | Preserves history, recoverable | Agent must track multiple IDs or follow the chain |

**Recommendation: Option A with a snapshot.** When extending, save a snapshot of the pre-extension state as `{result_id}.pre-extend.json`, then overwrite the main file. The result ID stays the same (CPO's requirement), history is preserved for debugging, and agents see a single growing result.

This requires a new `update_panel_result()` function in `mcp/data.py` (~15 lines).

### Q4: Synthesis model escalation — different models for intermediate vs final?

**Answer: Yes. Agree with CPO.**

Intermediate syntheses (between rounds) use the panelist model — they're working documents, not the final output. Final synthesis uses `--synthesis-model` (default: sonnet). This is already supported by the `model` parameter on `synthesize_panel()`. The round orchestrator just passes different model values.

One nuance: intermediate syntheses are used as template input for the next round's questions. Quality matters for templating — if the intermediate synthesis has poor theme extraction, the next round's questions will be weak. **Consider using sonnet for all syntheses if the budget allows, with haiku-for-intermediates as the cost-optimized default.**

---

## Feature 1: Multi-Round Orchestration

**Effort: 18-24h**  
**Risk: MEDIUM — new orchestration layer, template engine, contract change**

### Architecture

The current execution flow is:

```
run_panel_parallel() → [PanelistResult] → synthesize_panel() → output
```

Multi-round adds a loop:

```
for each round:
    template questions with prior synthesis
    run_panel_parallel() → [PanelistResult]  (reusing panelist sessions)
    synthesize_panel() → RoundSynthesis
final_synthesize() → FinalSynthesis (across all rounds)
output
```

### New module: `synth_panel/rounds.py`

This is the new orchestration layer. It sits between the CLI/MCP entry points and the existing `run_panel_parallel()`.

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

def run_multi_round(
    client: LLMClient,
    personas: list[dict[str, Any]],
    rounds: list[dict[str, Any]],      # Round definitions from instrument v2
    model: str,
    *,
    synthesis_model: str | None = None,
    final_synthesis_model: str | None = None,
    response_schema: dict[str, Any] | None = None,
    budget: float | None = None,
) -> MultiRoundResult:
    ...
```

### Session persistence across rounds

**This is the key architectural challenge.** Currently, `run_panel_parallel()` creates a fresh `Session()` per panelist inside `_run_panelist()`. For multi-round, the same session must carry across rounds.

**Two approaches:**

| Approach | Description | Effort | Risk |
|----------|-------------|--------|------|
| A: In-memory sessions | Keep `{persona_name: Session}` dict in `run_multi_round()`, pass to `_run_panelist()` | 4h | Low — sessions are small, no persistence needed between rounds |
| B: Persisted sessions | Save sessions to disk after each round, load before next round | 8h | Medium — adds I/O, enables resume but increases complexity |

**Recommendation: Approach A for multi-round, Approach B only for panel resume/extend.**

Multi-round instruments execute in a single invocation — there's no reason to persist sessions between rounds. Keep them in memory. Panel resume/extend (Feature 3) requires persistence, but that's a separate code path.

**Required change to `_run_panelist()`:** Add an optional `session: Session | None = None` parameter. If provided, reuse it instead of creating a new one. This is a 3-line change.

```python
def _run_panelist(
    ...,
    session: Session | None = None,  # NEW — reuse session across rounds
) -> PanelistResult:
    system_prompt = system_prompt_fn(persona)
    if session is None:
        session = Session()
    runtime = AgentRuntime(client=client, session=session, ...)
    ...
```

**Required change to `run_panel_parallel()`:** Accept optional `sessions: dict[str, Session] | None` mapping persona names to existing sessions. Pass through to `_run_panelist()`.

### Template engine

New module: `synth_panel/templates.py` (~60 lines)

```python
def build_template_context(synthesis: SynthesisResult) -> dict[str, str]:
    """Flatten a SynthesisResult into a template context dict."""
    ctx = {
        "summary": synthesis.summary,
        "recommendation": synthesis.recommendation,
    }
    for i, theme in enumerate(synthesis.themes):
        ctx[f"theme_{i}"] = theme
    for i, agreement in enumerate(synthesis.agreements):
        ctx[f"agreement_{i}"] = agreement
    for i, disagreement in enumerate(synthesis.disagreements):
        ctx[f"disagreement_{i}"] = disagreement
    for i, surprise in enumerate(synthesis.surprises):
        ctx[f"surprise_{i}"] = surprise
    return ctx

def render_questions(
    questions: list[dict[str, Any]],
    context: dict[str, str],
) -> list[dict[str, Any]]:
    """Render template variables in question texts."""
    ...
```

Use `string.Formatter` with a custom `format_field` that returns the literal placeholder on `KeyError`. This is safe (no eval), simple, and handles missing keys gracefully.

### Instrument v2 schema

```yaml
instrument:
  version: 2
  rounds:
    - name: exploration
      questions:
        - text: "What frustrates you about project management tools?"
        - text: "How do you currently track work across teams?"

    - name: probe
      depends_on: exploration
      questions:
        - text: "The panel flagged '{theme_0}' — describe a specific instance?"
        - text: "Several disagreed on '{disagreement_0}'. What drives your perspective?"

    - name: validation
      depends_on: probe
      questions:
        - text: "If a tool addressed {recommendation}, would you switch?"
          follow_ups:
            - text: "What would switching cost look like?"
              condition: "response_contains: yes"
            - text: "What would change your mind?"
              condition: "response_contains: no"
```

**Backward compatibility:** Instruments without `rounds` are treated as a single-round instrument. The existing `questions` key at the top level is interpreted as `rounds: [{name: "default", questions: [...]}]`. No breaking change.

**`depends_on` semantics:** Linear only for 0.4.0. `depends_on: exploration` means "run after exploration's synthesis is available." The round orchestrator validates that `depends_on` references an earlier round (no cycles, no forward references). Branching (`depends_on` with conditions) is 0.5.0.

### MCP contract change

**Current:**
```json
{
  "result_id": "...",
  "results": [...],
  "synthesis": {...},
  "total_cost": "$0.04"
}
```

**Multi-round (additive):**
```json
{
  "result_id": "...",
  "rounds": [
    {
      "name": "exploration",
      "results": [...],
      "synthesis": {"summary": "...", ...}
    },
    {
      "name": "probe",
      "results": [...],
      "synthesis": {"summary": "...", ...}
    }
  ],
  "final_synthesis": {"summary": "Across 3 rounds...", ...},
  "total_cost": "$0.12",
  "panelist_cost": "$0.08",
  "synthesis_cost": "$0.04"
}
```

**Single-round backward compatibility:** When a single-round instrument is used, the output retains the current flat structure (`results`, `synthesis`). The `rounds` field is only present for multi-round instruments. This means existing agent integrations are unaffected.

**Alternative: always use `rounds` format, even for single-round.** This is cleaner but technically breaking — agents reading `results` at the top level would need to change to `rounds[0].results`. **Recommendation: keep backward compat for 0.4.0, deprecate the flat format in 0.5.0.** This avoids breaking agent workflows.

### What changes per file

| File | Change | New lines |
|------|--------|-----------|
| `synth_panel/rounds.py` | **NEW** — multi-round orchestrator | ~180 |
| `synth_panel/templates.py` | **NEW** — template engine | ~60 |
| `orchestrator.py` | Add `session` param to `_run_panelist()` and `sessions` to `run_panel_parallel()` | ~15 |
| `mcp/server.py` | Add round support to `run_panel`, call `run_multi_round()` when instrument has rounds | ~40 |
| `cli/commands.py` | Detect v2 instruments, call `run_multi_round()`, format round output | ~50 |
| `cli/parser.py` | No new flags needed — rounds come from instrument YAML | ~0 |
| `mcp/data.py` | Store round data in results (already flexible) | ~5 |
| `prompts.py` | Add final synthesis prompt (cross-round) | ~15 |
| Tests | `test_rounds.py`, `test_templates.py`, updates to `test_mcp_server.py`, `test_cli.py` | ~250 |

---

## Feature 2: `response_sentiment` Condition

**Effort: 4-6h**  
**Risk: LOW — isolated addition to existing conditions registry**

### Implementation

Add an LLM-based sentiment evaluator to the conditions registry.

```python
# In conditions.py

def _eval_sentiment(keyword: str, response_text: str, client: LLMClient | None = None) -> bool:
    """Classify response sentiment and match against keyword (positive/negative/neutral)."""
    if client is None:
        # Fallback: keyword heuristic for testing
        return True
    # Make a haiku call to classify sentiment
    ...
```

### The client dependency problem

The current `evaluate_condition()` signature is `(condition: str, response_text: str) -> bool`. Adding `response_sentiment` requires an LLM client. Two options:

| Option | Approach | Pros | Cons |
|--------|----------|------|------|
| A | Pass `client` to `evaluate_condition()` | Simple, direct | Changes the function signature — all callers must update |
| B | Registry evaluators are objects with optional `client` injection | Extensible, clean | More complex, over-engineered for one evaluator |

**Recommendation: Option A.** Add `client: LLMClient | None = None` as an optional parameter. Non-LLM evaluators ignore it. This is a one-line signature change with backward compatibility (default `None` means existing callers don't break).

### Caching

The CPO mentions caching results per-response. This matters because the same response might be evaluated against multiple conditions (e.g., both `response_sentiment: positive` and `response_sentiment: negative`).

**Implementation:** A simple `{response_hash: sentiment}` dict passed through the round loop. Sentiment classification only runs once per unique response text.

### Cost

One haiku call per unique response that needs sentiment classification. For a 5-persona, 3-question panel with 2 conditional follow-ups: worst case ~10 haiku calls × ~$0.0003 = $0.003. Negligible.

### What changes per file

| File | Change | New lines |
|------|--------|-----------|
| `conditions.py` | Add `_eval_sentiment()`, update signature with `client` param, add cache | ~50 |
| `orchestrator.py` | Pass `client` to `evaluate_condition()` calls | ~3 |
| Tests | `test_conditions.py` updates | ~40 |

---

## Feature 3: Panel Resume / Extend

**Effort: 8-10h**  
**Risk: MEDIUM — requires session persistence, new MCP tool, new CLI command**

### Architecture

Panel resume/extend requires:
1. **Persisting panelist sessions** — after a panel run, save each panelist's session to disk
2. **Loading sessions** — when extending, load the sessions and resume conversation
3. **New entry points** — `extend_panel` MCP tool, `synth-panel panel extend` CLI command

### Session persistence strategy

Currently, panelist sessions exist only in memory during `run_panel_parallel()`. For resume, they must be saved alongside the panel result.

**Option A: Embed sessions in the result JSON.**
- Simple but results become very large (every message in every session)
- Loading a result to check metadata also loads all session data

**Option B: Separate session files alongside results.**
```
~/.synth-panel/results/
  result-20260405-123456-abc123.json       # Panel result (metadata + responses)
  result-20260405-123456-abc123.sessions/  # Session directory
    Sarah_Chen.json                         # Per-persona session
    Marcus_Johnson.json
```

**Recommendation: Option B.** Session storage is opt-in (only when `--save-session` or multi-round is used). Results stay lightweight. Sessions are loaded only when extending.

### Required changes to orchestrator

`run_panel_parallel()` must return the sessions alongside results:

```python
def run_panel_parallel(
    ...,
    sessions: dict[str, Session] | None = None,  # Reuse existing sessions
) -> tuple[list[PanelistResult], WorkerRegistry, dict[str, Session]]:
    #                                              ^^^^^^^^^^^^^^^^^ NEW
```

The `dict[str, Session]` maps persona names to their sessions. The caller decides whether to persist them.

### New MCP tool: `extend_panel`

```python
@mcp.tool()
async def extend_panel(
    result_id: str,
    questions: list[dict[str, Any]],
    model: str | None = None,
    synthesis: bool = True,
    synthesis_model: str | None = None,
    ctx: Context = None,
) -> str:
```

Flow:
1. Load existing result by `result_id`
2. Load persisted sessions for that result
3. Run new questions against existing sessions (personas remember prior conversation)
4. Synthesize the new round
5. Append to existing result (per Q3 answer: overwrite with snapshot)
6. Return updated result

### New CLI command: `synth-panel panel extend`

```bash
synth-panel panel extend <result-id> --questions "Tell me more about pricing"
synth-panel panel extend <result-id> --instrument follow-up.yaml
```

### What changes per file

| File | Change | New lines |
|------|--------|-----------|
| `orchestrator.py` | Return sessions from `run_panel_parallel()`, accept sessions param | ~20 |
| `mcp/data.py` | `save_panel_sessions()`, `load_panel_sessions()`, `update_panel_result()` | ~60 |
| `mcp/server.py` | New `extend_panel` tool | ~50 |
| `cli/commands.py` | New `handle_panel_extend()` | ~60 |
| `cli/parser.py` | Add `panel extend` subcommand | ~15 |
| `persistence.py` | No change — Session already has `to_dict()`/`from_dict()` | 0 |
| Tests | `test_extend.py` (new), updates to `test_mcp_server.py`, `test_cli.py` | ~120 |

---

## Architectural Decisions to Lock NOW

### Decision 1: Session reuse mechanism

**The question:** How does the orchestrator reuse sessions across rounds?

**Option A: Modify `_run_panelist()` signature** — add `session: Session | None = None`.  
**Option B: Session registry** — a `SessionRegistry` class that `_run_panelist()` queries by persona name.

**Recommendation: Option A.** It's simpler, explicit, and the orchestrator already manages the mapping. A registry adds indirection without benefit — there's only one consumer (the round loop).

**Lock this now** because both multi-round (Feature 1) and panel extend (Feature 3) depend on it. If built differently, one will need refactoring.

### Decision 2: Multi-round vs single-round output format

**The question:** When a single-round instrument runs, does the output use the new `rounds` format or the existing flat format?

**Option A: Always `rounds` format** — single-round is `rounds: [{...}]` with one element.  
**Option B: Flat for single-round, `rounds` for multi-round** — backward compatible.

**Recommendation: Option B for 0.4.0, migrate to Option A in 0.5.0.** The reasoning: agents that integrated with 0.3.0's flat format shouldn't break. But the flat format is a maintenance burden — two code paths for formatting output. Add a deprecation notice in 0.4.0 docs, remove flat format in 0.5.0.

**Lock this now** because the MCP tool response schema and the persistence format both depend on it.

---

## Risk Assessment

### Risk 1: Template rendering failures (MEDIUM)

If a synthesis produces fewer themes than the template expects (e.g., `{theme_2}` but only 1 theme was found), the template renders with literal `{theme_2}` text. The persona gets asked: "The panel flagged '{theme_2}' — describe a specific instance?"

**Mitigation:** Validate templates against the synthesis output before rendering. If a referenced key is missing, log a warning and either skip the question or render a fallback. Implement in `templates.py` with a `validate_template()` function.

### Risk 2: Token growth across rounds (MEDIUM)

Each round adds messages to the panelist session. A 3-round study with 3 questions per round accumulates ~18 messages per panelist (3 rounds × 3 questions × 2 messages per turn). With 5 personas, that's 90 messages total in context.

For haiku with 200K context, this is fine. But the synthesis prompt also grows — it receives all panelist responses across all rounds for the final synthesis.

**Mitigation:** Offer optional session compaction between rounds (`Session.compact()` already exists). The round orchestrator can compact after each round, keeping only the summary + last N messages. This is opt-in — default behavior keeps full context.

### Risk 3: Multi-round latency (LOW)

A 3-round study with 5 personas runs: 5 parallel panelists × 3 questions + synthesis, repeated 3 times + final synthesis. That's roughly: (5 parallel × 3 sequential + 1 synthesis) × 3 rounds + 1 final = ~33 LLM calls, ~10 sequential.

With haiku (~1-2s per call): ~20-30 seconds total. With sonnet: ~45-60 seconds. Acceptable for a research tool — real focus groups take hours.

### Risk 4: `extend_panel` race conditions (LOW)

If two agents extend the same panel simultaneously, the overwrite semantics create a race. The last writer wins, potentially losing the other's extension.

**Mitigation:** File locking on the result file during extend. Or accept the limitation and document it — this is a single-user CLI tool, not a multi-tenant service. Race conditions require deliberate misuse.

---

## Dependency Order

```
Phase 1 (Foundation — no user-visible changes):
├── Session reuse in _run_panelist() [Decision 1]
├── Template engine (templates.py)
├── conditions.py integration in orchestrator (wire evaluate_condition)
└── response_sentiment evaluator

Phase 2 (Multi-round — headline feature):
├── rounds.py module [depends on: session reuse, template engine]
├── Instrument v2 parser (round definitions)
├── Per-round synthesis calls
├── Final cross-round synthesis
└── MCP + CLI integration [depends on: Decision 2]

Phase 3 (Panel extend — agent-native feature):
├── Session persistence (save/load per panelist)
├── update_panel_result() in mcp/data.py
├── extend_panel MCP tool [depends on: session persistence]
└── panel extend CLI command [depends on: session persistence]
```

Phase 1 items are parallelizable across polecats. Phase 2 depends on Phase 1. Phase 3 depends on Phase 1 (session reuse) but can be developed in parallel with Phase 2.

---

## Effort Summary

| Feature | Effort | Risk | Parallelizable |
|---------|--------|------|----------------|
| F1: Multi-round orchestration | 18-24h | Medium | Foundation in parallel, integration sequential |
| F2: `response_sentiment` | 4-6h | Low | Fully parallel with F1 |
| F3: Panel resume/extend | 8-10h | Medium | Phase 1 parallel with F1, Phase 3 after F1 foundation |
| **Total** | **30-40h** | | |

### Recommended polecat assignment

- **Polecat A:** F1 (multi-round) — this is the critical path, assign strongest implementer
- **Polecat B:** F2 (sentiment) + F3 Phase 1 (session persistence) — independent foundation work
- **Integration:** After both deliver, wire F3 extend into the multi-round infrastructure

---

## One More Thing: Wire Conditions in 0.3.0

The exploration revealed that `conditions.py` has `evaluate_condition()` and `normalize_follow_up()` implemented and tested, but **they are not yet called from the orchestrator**. Follow-ups in `_run_panelist()` still fire unconditionally.

This is a 0.3.0 gap, not a 0.4.0 feature. If conditional follow-ups were announced in 0.3.0, this wiring should land before 0.4.0 work begins. It's ~10 lines in `orchestrator.py:341-354`:

```python
# Before (current):
for fu in question.get("follow_ups", []):
    summary = runtime.run_turn(fu if isinstance(fu, str) else fu.get("text", str(fu)))

# After (wired):
for fu_raw in question.get("follow_ups", []):
    fu = normalize_follow_up(fu_raw)
    if evaluate_condition(fu.get("condition", "always"), last_response_text):
        summary = runtime.run_turn(fu["text"])
```

**Flag this for immediate action** — it's a bug if 0.3.0 claims conditional follow-ups work but they don't.
