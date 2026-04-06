# synth-panel 0.5.0 — Execution Plan

> PM: synth-panel/pm | Date: 2026-04-06
> Inputs: `0.5.0-VISION.md` (CPO), `0.5.0-ARCHITECTURE-REVIEW.md` (architect)

The 0.5.0 "Adaptive Release" turns the orchestrator into a router. Branching
instruments make synth-panel choose its research path from what the panel said.
Instrument packs ship the second ecosystem flywheel alongside persona packs.
Flat output gets retired before 1.0.0.

Total effort: ~30-40h, parallelizable across 2-3 polecats once Phase 1 lands.

---

## Locked Decisions (Do Not Relitigate)

From the architect review, endorsed by CPO:

1. **Predicate form on disk: structured dict, not string.** YAML handles parsing;
   no parser, no eval, no grammar. The string form `"synthesis.themes contains 'pricing'"`
   is *display syntax only* — rendered from `{field, op, value}` for docs/READMEs.
2. **DAG validation at parse time.** Five-rung ladder: structural → goto resolution
   → acyclicity → else completeness → reachability (warning, not error).
3. **`else` is mandatory.** Every `route_when` block must terminate in `else`.
   No silent fall-through. Use `else: __end__` to terminate the run explicitly.
4. **`__end__` is the reserved terminal sentinel.** Any `goto: __end__` ends the
   run and triggers final synthesis on the executed-rounds-only path.
5. **Instrument packs are single-file YAML** with manifest fields (`name`, `version`,
   `description`, `author`) at top level and `instrument:` nested.
6. **`path` is in the default MCP/CLI response**, not behind a flag.
7. **Final synthesis sees executed rounds only.** Skipped branches are not passed
   to the synthesizer prompt — describes what happened, not what could have.
8. **Hand-rolled predicate engine** in a new `routing.py`. Three operators
   (`contains`, `equals`, `matches`) plus `else`. ~50 LOC.

---

## Scope Boundaries

| Version | In | Out |
|---------|-----|-----|
| **0.5.0** | Branching instruments (DAG, no loops), instrument packs (5 bundled), flat-output removal, `instruments graph` command | LLM-authored questions/routing, loops/back-edges, persona-to-persona interaction, hosted registry, web UI for DAGs |
| **0.6.0** | Stabilization release. Bug fixes, perf, expanded instrument library. | New features. |

---

## Execution Order

The architect's sequencing is the contract. Phase 1 is the critical path;
Phase 2 runs in parallel; Phase 3 depends on Phases 1+2.

```
Phase 1 — Foundation (critical path, sequential):
├── F1-A: routing.py — predicate engine + router
├── F1-B: v3 instrument parser + DAG validator    [parallel with F1-A]
└── F1-C: orchestrator wiring (branching loop)    [depends on F1-A + F1-B]

Phase 2 — Parallel tracks (no dependencies on Phase 1):
├── F2-A: Flat-output removal + --legacy-output
└── F2-B: Pack loader generalization (dir vs file)

Phase 3 — Surface + content (depends on Phase 1 + Phase 2):
├── F3-A: CLI surface (instruments subcommands + branching wiring)
├── F3-B: MCP surface (instrument-pack tools + branching response shape)
├── F3-C: Bundled instrument packs (5 YAMLs, branching demos)
├── F3-D: instruments graph command (Mermaid renderer)
├── F3-E: Text-mode path renderer
├── F3-F: v1/v2/v3 backward-compat test matrix
├── F3-G: 0.5.0 README + theme-matching guidance (R3 mitigation)
└── F3-H: extend_panel-vs-branching docs clarification
```

**Critical path:** F1-A + F1-B → F1-C → F3-A + F3-B → F3-C → F3-G (~18-22h)
**Parallel track:** F2-A, F2-B (~4h, runs alongside Phase 1)

---

## Phase 1 — Foundation

### F1-A: routing.py — predicate engine + router
**Effort:** 2-3h | **Deps:** none | **Parallelizable with:** F1-B, F2-A, F2-B

New module `synth_panel/routing.py` (~50 LOC).

**What to build:**
1. `evaluate_predicate(predicate: dict, context: dict) -> bool`
   - `predicate` shape: `{field: str, op: str, value: str}`
   - Supported ops: `contains`, `equals`, `matches` (regex)
   - `field` resolves into `SynthesisResult` shape: `themes`, `recommendation`,
     `disagreements`, `summary`, `agreements`, `surprises`
   - Unknown field → raise `KeyError` with the offending name (caught upstream
     for clear error)
2. `route_round(route_when: list[dict], context: dict) -> str`
   - Iterate clauses; first matching predicate wins
   - Each clause: `{if: <predicate>, goto: <round_name>}` or `{else: <round_name>}`
   - Returns the target round name (or `__end__`)
3. Reserved sentinel: `__end__` is the terminal target — orchestrator runs
   final synthesis on the path so far.

**Acceptance criteria:**
- Unit tests for all three operators against a synthetic `SynthesisResult`-shaped dict
- `contains` does substring match against any list entry
- `equals` does exact string match
- `matches` does `re.search` (not `re.match`)
- Unknown field raises `KeyError` with field name
- `route_round` returns first match, falls through to `else`
- No predicate clause with no match and no `else` → raises (parser catches earlier)

### F1-B: v3 instrument parser + DAG validator
**Effort:** 3-4h | **Deps:** none | **Parallelizable with:** F1-A, F2-A, F2-B

Edit `synth_panel/instrument.py` (~80 LOC delta).

**What to build:**
1. **Two-pass parse:**
   - Pass 1: collect all round names from `rounds[]`
   - Pass 2: validate `depends_on` and `route_when.goto` references against
     the name set (forward refs now allowed; old "earlier-only" constraint relaxed)
2. **Add `route_when` field to `Round` dataclass** (optional, list of dicts)
3. **DAG validator** runs at parse time, before any LLM call. Five rungs:
   1. **Structural** — every round has `name` + `questions`; names unique
   2. **Goto resolution** — every `goto` (and `else` target) names a defined
      round (or `__end__`)
   3. **Acyclicity** — build edge set from `depends_on` ∪ `route_when.goto`,
      topo sort, fail with the cycle path on detection
   4. **Else completeness** — every `route_when` block ends in `else`. Reject
      with: `round '<name>' has no else clause; add 'else: <round_name>' or 'else: __end__'`
   5. **Reachability** — forward-traverse from entry round; emit warning (not
      error) on unreachable rounds. Warning surfaces via `warnings: []` in result
4. **v1/v2 stay as degenerate v3:** v1 (flat questions) → wrap as one-round v3;
   v2 (linear rounds, no `route_when`) → still valid v3 with implicit linear
   edges. No breaking change.

**Acceptance criteria:**
- v1, v2, v3 instruments all parse through `parse_instrument()`
- Cycle: `a → b → a` rejected at parse time with cycle path in error
- Forward ref: `goto: probe_pricing` defined later in file resolves correctly
- Missing `else`: rejected with clear error naming the offending round
- Unreachable round: warning emitted, parse succeeds
- Bad `goto` target: rejected with `round '<n>' goto '<target>' does not exist`
- Existing 0.4.0 multi-round tests still pass unchanged

### F1-C: Orchestrator wiring — branching loop
**Effort:** 2-3h | **Deps:** F1-A, F1-B

Edit `synth_panel/orchestrator.py` (~30 LOC delta).

**What to change:**
1. Replace the linear `for round in rounds` loop with a router-driven
   `while next_round and next_round != '__end__'` loop
2. After each round's synthesis, call
   `routing.route_round(round.route_when, template_context)` if `route_when`
   is set; otherwise fall back to existing `depends_on` linear next-round logic
3. Track executed-rounds list and **path entries** as the loop runs:
   `{round: <name>, branch: <rendered predicate string>, next: <target>}`
4. **Final synthesis tags the terminal round** of the executed path — not the
   syntactic last round in the file
5. Pass **only executed rounds** to the final synthesis prompt (architect Q6)
6. Existing budget gate (`BudgetGate`) wraps each round; no behavior change
7. Returned `MultiRoundResult` gains `path: list[dict]` and `warnings: list[str]`

**Acceptance criteria:**
- Linear v2 instruments execute identically to 0.4.0 (no path divergence test)
- v3 instrument with one branch executes correct branch based on synthesis content
- `__end__` sentinel terminates the run after the current round; final synthesis
  fires immediately
- `path` array correctly records each round's branching decision in order
- Final synthesis prompt receives only executed rounds (verified by mocking
  the synthesizer and asserting on its inputs)
- Budget gate still triggers mid-run on a multi-round branching instrument

---

## Phase 2 — Parallel Tracks

### F2-A: Flat-output removal + --legacy-output flag
**Effort:** 2h | **Deps:** none | **Parallel with:** all of Phase 1

Edit `synth_panel/output.py` (~40 LOC delta) plus an audit pass.

**What to change:**
1. Single-round instruments now emit the same `rounds`-shaped output as
   multi-round (one-element rounds list). The old `results`/`synthesis` flat
   shape goes away.
2. Add `--legacy-output` CLI flag that emits the old shape with a `DeprecationWarning`
   on stderr. Removed in 0.6.0.
3. **MCP gets the new shape unconditionally.** Agents post-date the deprecation
   per CPO call.
4. **Audit `examples/`, `skills/`, README, and any 0.3.0/0.4.0 doc references**
   for flat-shape assumptions; update or annotate. Architect risk R6.

**Acceptance criteria:**
- `synth-panel panel run` (no flag) emits `rounds` shape for single-round instrument
- `--legacy-output` emits old shape + stderr warning
- MCP `run_panel` always emits `rounds` shape for single-round instrument
- All existing examples and skills work with the new shape
- Tests cover both shapes (legacy via flag, new by default)

### F2-B: Pack loader generalization
**Effort:** 2h | **Deps:** none | **Parallel with:** all of Phase 1

Edit `synth_panel/mcp/data.py` (~40 LOC delta).

**What to change:**
1. Generalize the pack loader: if path is a directory → walk for personas
   (existing behavior); if path is a file → parse as instrument with manifest
   keys at top level
2. Manifest fields are the same for both types: `name`, `version`, `description`,
   `author`. Persona packs keep these in `manifest.yaml`; instrument packs
   keep them at top level of the instrument YAML
3. Storage convention: `~/.synth-panel/packs/personas/<name>/` (dir) vs
   `~/.synth-panel/packs/instruments/<name>.yaml` (file)
4. New helpers in `data.py`: `list_instrument_packs()`, `load_instrument_pack(name)`,
   `save_instrument_pack(name, content)`

**Acceptance criteria:**
- Existing persona pack tests pass unchanged
- Round-trip: save → list → load returns identical instrument YAML
- File-vs-directory dispatch is unambiguous (file extension or path type check)
- Listing returns both pack types correctly tagged

---

## Phase 3 — Surface + Content

### F3-A: CLI surface — instruments subcommands + branching wiring
**Effort:** 4h | **Deps:** F1-A, F1-B, F1-C, F2-B

Edit `synth_panel/cli/commands.py` and `cli/parser.py` (~120 LOC delta).

**What to build:**
1. **New subcommand group: `synth-panel instruments`**
   - `instruments list` — list installed packs (name, version, description)
   - `instruments install <name>` — install from `synth-panel/instruments` repo
   - `instruments show <name>` — print the YAML
   - `instruments graph <file-or-name>` — render Mermaid DAG to stdout (see F3-D)
2. **`panel run --instrument <name>` accepts pack names** in addition to file paths
3. **Output formatting** for branching results:
   - Show executed path: `exploration → probe[themes contains 'pricing'] → probe_pricing → validation`
   - Per-round breakdown stays as-is
   - Surface `warnings: []` to stderr
4. `--compare` extends to render per-round tables for the executed path

**Acceptance criteria:**
- `instruments list/show` work against an installed pack
- `instruments install` clones from the curated repo (or warns if offline)
- `panel run --instrument pricing-discovery` resolves to the installed pack
- Branching run output shows the path string clearly
- Warnings render to stderr, not stdout

### F3-B: MCP surface — instrument-pack tools + branching response shape
**Effort:** 3h | **Deps:** F1-A, F1-B, F1-C, F2-B

Edit `synth_panel/mcp/server.py` (~80 LOC delta).

**What to build:**
1. **New tools** (mirroring persona-pack tools):
   - `list_instrument_packs() -> list[dict]`
   - `get_instrument_pack(name: str) -> dict`
   - `save_instrument_pack(name: str, content: dict) -> dict`
2. **`run_panel` response shape gains `path` and `warnings` keys** for v3
   instruments. Single-round and v2 multi-round responses are unchanged
   structurally (path length 1 or N).
3. **`extend_panel` interaction with branching** is unchanged (architect note 3):
   extend appends a single ad-hoc round, not a re-entry into the DAG. Document
   this in the tool's docstring explicitly.

**Acceptance criteria:**
- All 3 new tools callable via MCP, return shapes match persona-pack equivalents
- `run_panel` with v3 instrument returns `rounds` + `path` + `warnings`
- `run_panel` with v1/v2 instrument returns same shape (path length 1 or N)
- `extend_panel` docstring spells out "improvised round, not DAG re-entry"
- Tests cover both v3 and v2 paths through `run_panel`

### F3-C: Bundled instrument packs (5 YAMLs, branching)
**Effort:** 4h | **Deps:** F1-A, F1-B, F1-C
**Parallelizable with:** F3-A, F3-B once Phase 1 lands

Author 5 hand-vetted instrument packs in `packs/instruments/`. CPO mandate:
all five must demonstrate branching to justify the library's existence.

**Packs:**
1. `pricing-discovery.yaml` — exploration → branch on price-objection vs
   value-objection → validation
2. `name-test.yaml` — exposure → branch on positive-recall vs negative-recall vs neutral → probe → validation
3. `feature-prioritization.yaml` — exploration → branch on tradeoff-axis (cost/time/quality) → probe → validation
4. `landing-page-comprehension.yaml` — first-impression → branch on confusion-source → clarification → validation
5. `churn-diagnosis.yaml` — context → branch on churn-driver → probe → validation

Each pack must include:
- Top-level manifest (`name`, `version`, `description`, `author: synth-panel/community`)
- A v3 instrument with `route_when` clauses
- Comments explaining the branching logic
- A theme-tag hint in the synthesis prompt to mitigate R3 (architect risk):
  "When applicable, use these tags in themes: pricing, integrations, ux, performance, ..."

**Acceptance criteria:**
- All 5 packs parse successfully
- All 5 demonstrate at least one branching decision
- A documented test runs each pack against `examples/personas.yaml` (acceptance,
  gated on API key)
- README references at least the `pricing-discovery` pack as the demo

### F3-D: instruments graph command (Mermaid renderer)
**Effort:** 1h | **Deps:** F1-B

Add to `cli/commands.py` (~30 LOC).

**What to build:**
- `synth-panel instruments graph <file-or-name>` reads the instrument, walks
  the round DAG (depends_on ∪ route_when targets), and emits a Mermaid diagram
  to stdout
- No graphviz dependency — pure string template
- Predicate labels on edges (rendered string form: `themes contains 'pricing'`)

**Acceptance criteria:**
- Output is valid Mermaid `flowchart TD` syntax
- Each round is a node; each edge is labeled with its branching predicate (or unlabeled for `depends_on`)
- `__end__` rendered as a terminal node
- Works for v1, v2, v3 instruments

### F3-E: Text-mode path renderer
**Effort:** 1h | **Deps:** F1-C

Edit `synth_panel/output.py` (~20 LOC).

Render `path` in text mode as:
`exploration → probe[themes contains 'pricing'] → probe_pricing → validation`

**Acceptance criteria:**
- Text output for branching runs shows path string above the per-round breakdown
- Single-round and linear v2 runs render path as well (degenerate cases)

### F3-F: v1/v2/v3 backward-compat test matrix
**Effort:** 2h | **Deps:** F1-B, F1-C

Edit `tests/test_instrument.py` and `tests/test_acceptance.py`.

Add a parameterized test matrix that runs the same orchestrator path against
v1 (flat), v2 (linear rounds), and v3 (branching) instruments. Acceptance
tests gated on API key; unit tests use mocked LLM.

**Acceptance criteria:**
- All three formats parse successfully
- All three formats execute through the same orchestrator
- v1 and v2 outputs structurally match (single rounds list)
- v3 output adds `path` and may include `warnings`

### F3-G: 0.5.0 README + theme-matching guidance (R3 mitigation)
**Effort:** 2h | **Deps:** F3-A, F3-B, F3-C

Edit `README.md` and any 0.5.0-relevant docs in `CLAUDE.md`.

**What to add:**
1. Branching demo using `pricing-discovery` pack (the "$0.20 adaptive research" demo from CPO vision)
2. **Theme-matching expectations section (R3):** explicit warning that
   `themes contains 'pricing'` matches against the synthesizer's exact tag
   output, and the recommended pattern of including a theme-tag hint in the
   synthesis prompt
3. Predicate reference: dict-form on disk, three operators, `else` mandatory
4. `instruments` subcommand reference
5. Update version table, mark 0.4.0 as shipped (assuming it has by 0.5.0 work)

**Acceptance criteria:**
- README has a working branching example (pack + command)
- Theme-matching gotcha is documented loudly
- Predicate grammar reference fits on a screen
- No mention of features not in 0.5.0

### F3-H: extend_panel-vs-branching docs clarification
**Effort:** 30min | **Deps:** F3-B

Add a paragraph to `extend_panel`'s docstring and to `CLAUDE.md` explaining
that `extend_panel` always appends a single ad-hoc round and is *not* a
re-entry into the authored DAG. Architect note 3.

**Acceptance criteria:**
- Docstring spells out the contract
- CLAUDE.md has the same paragraph in its `extend_panel` reference
- No code change required

---

## Bead Summary

| Bead | ID | Title | Priority | Effort | Phase | Deps |
|------|----|-------|----------|--------|-------|------|
| F1-A | sp-khy | routing.py — predicate engine + router | P0 | 2-3h | 1 | — |
| F1-B | sp-zdh | v3 instrument parser + DAG validator | P0 | 3-4h | 1 | — |
| F1-C | sp-y5e | Orchestrator wiring — branching loop | P0 | 2-3h | 1 | sp-khy, sp-zdh |
| F2-A | sp-zg4 | Flat-output removal + --legacy-output | P1 | 2h | 2 | — |
| F2-B | sp-chg | Pack loader generalization (dir vs file) | P1 | 2h | 2 | — |
| F3-A | sp-xsu | CLI — instruments subcommands + branching wiring | P0 | 4h | 3 | sp-y5e, sp-chg |
| F3-B | sp-yiz | MCP — instrument-pack tools + branching shape | P0 | 3h | 3 | sp-y5e, sp-chg |
| F3-C | sp-l7w | Bundled instrument packs (5 branching YAMLs) | P1 | 4h | 3 | sp-y5e |
| F3-D | sp-irf | instruments graph command (Mermaid) | P1 | 1h | 3 | sp-zdh |
| F3-E | sp-8u9 | Text-mode path renderer | P2 | 1h | 3 | sp-y5e |
| F3-F | sp-rc3 | v1/v2/v3 backward-compat test matrix | P1 | 2h | 3 | sp-zdh, sp-y5e |
| F3-G | sp-2yb | 0.5.0 README + theme-matching guidance | P1 | 2h | 3 | sp-xsu, sp-yiz, sp-l7w |
| F3-H | sp-ilf | extend_panel-vs-branching docs clarification | P2 | 30m | 3 | sp-yiz |

**Critical path:** F1-A + F1-B → F1-C → F3-A + F3-B → F3-C → F3-G (~18-22h)
**Parallel tracks:** F2-A, F2-B (~4h alongside Phase 1); F3-D, F3-E, F3-F, F3-H (alongside Phase 3)

**Total estimated:** 28-33h critical + parallel tracks. Comparable to 0.4.0.

---

## Risks Inherited From Architect Review

| ID | Risk | Severity | Mitigation in this plan |
|----|------|----------|-------------------------|
| R1 | Predicate string vs dict bikeshed | LOW | Locked: dict on disk, string in docs (decision §1) |
| R2 | Forward references break parse ordering | LOW | F1-B two-pass parser |
| R3 | Fuzzy theme matching against LLM lists | **MEDIUM** | F3-C pack-level theme-tag hints; F3-G README warning; `matches` operator can target `summary`/`recommendation` as escape hatch |
| R4 | Reachability warnings vs errors | LOW | F1-B emits warnings via `warnings: []`; F3-A surfaces to stderr; F3-B includes in MCP response |
| R5 | Path serialization in text output | LOW | F3-E |
| R6 | Flat-output removal blast radius | LOW–MED | F2-A includes audit pass on `examples/`, `skills/`, docs |
| R7 | Terminal sentinel design | LOW | Locked: `__end__` reserved name |
| R8 | Cost bound on misauthored DAG | LOW | DAG-no-loops + `BudgetGate` already covers |

---

## Pre-0.5.0 Items

| Item | Status | Action |
|------|--------|--------|
| 0.4.0 work | In progress per vision doc | **0.5.0 work begins after 0.4.0 lands.** Verify with CPO before starting F1-A. |
| PR #22 (sp-1d3) conditions wiring | Verified merged per architect review | None — `evaluate_condition()` is called at orchestrator.py:349-356 |
| Bundled persona packs | Shipped 0.2.0 | None — F2-B generalizes loader without changing persona behavior |

---

## Show HN Posture

Per CPO: 0.5.0 is the stronger Show HN target. Branching demo > linear
multi-round demo. The 0.5.0 demo is the `pricing-discovery` pack run against
`examples/personas.yaml` — one command, ~$0.20, the panel chooses its own
path. F3-C and F3-G are the show pieces; they need to land polished, not just
functional.

---

## Out of Scope (Confirmed Anti-Scope)

- LLM-authored questions or routing predicates
- Loops, back-edges, "rerun an earlier round"
- Persona-to-persona interaction
- Hosted instrument registry (marketplace, ratings, payments)
- Web UI for branching instruments (use `instruments graph` for visualization)
- 1.0.0 release (gated on real adoption + 3 minor releases of API stability)
