# Research Map — sp-inline-calibration

Synthesis of two research dimensions for the D-phase gate. Ticket now revealed: **sp-0ku0 — inline SynthBench calibration score in panel output.**

## What the codebase currently provides

**SynthBench → synthpanel flow is wired.** `SynthPanelProvider` at `src/synthbench/providers/synthpanel.py:287-400` consumes synthpanel JSON output via two paths: direct API (`synth_panel.llm.client.LLMClient` import + `ThreadPoolExecutor`) and CLI subprocess fallback. Provider reads only `data["rounds"][0]["results"][*].responses[i].response` per panelist per question; metadata fields (cost, usage) are optional.

**Question abstraction is option-string-keyed.** `Question` dataclass (`src/synthbench/datasets/base.py:34-50`) carries `options: list[str]` and `human_distribution: dict[str, float]` keyed by option text. Distributions normalized in `__post_init__` if sum deviates by >1%. No ordinal-index or enum-token layer; option text is the key.

**Question matching is explicit by dataset structure.** SynthBench orchestrates: loads `Question` from the adapter, passes `question.text + question.options` to the provider. No fuzzy matching, no synthpanel→synthbench question-key registry. Integration assumes synthpanel is invoked only by SynthBench's provider.

**JSD is simple and well-typed.** `metrics/distributional.py:6-34`:
```python
def jensen_shannon_divergence(p, q) -> float:
    # union supports, pad missing with 0.0, scipy.spatial.distance.jensenshannon base=2 squared
```
Returns single float in [0.0, 1.0]. Disjoint supports handled. Normalized internally.

**Synthpanel → SynthBench is one-way.** `synthpanel/pyproject.toml:46-48` declares `[convergence]` extra requiring `synthbench>=0.1`. Lazy import in `src/synth_panel/convergence.py:554-608::load_synthbench_baseline()` — imports inside the function, raises `SynthbenchUnavailableError` if missing. CLI flag `--convergence-baseline DATASET:QUESTION` (sp-yaru) is the current opt-in surface.

**Convergence output structure exists and is sibling-safe.** `convergence` top-level section added by sp-yaru, populated by `ConvergenceTracker.build_report()`. Contains `final_n`, `overall_converged_at`, `tracked_questions`, `per_question: {key: {converged_at, curve, support_size}}`, and `human_baseline` spliced verbatim from the loaded SynthBench payload.

**Nine SynthBench datasets with redistribution tiers:**
- `full` (per-question distributions publishable): **gss, ntia**
- `gated` (authenticated R2 bucket): wvs, michigan, pewtech, eurobarometer, opinionsqa, globalopinionqa, subpop
- Not in current inventory: `aggregates_only`, `citation_only` (reserved for future)

**Only GSS and NTIA are redistribution-safe for inline surfaces**. Others require authenticated access or aggregate-only consumption.

## Instrument ↔ dataset coverage (reality check)

Comparing synthpanel's 8 bundled instruments to SynthBench's 9 datasets:

**All 8 synthpanel instruments are 100% free-text today.** Every question declares `response_schema: {type: text}`. No Likert, enum, or yes-no. Branching instruments route on post-hoc theme extraction by the synthesizer, not pre-defined options.

**All 9 SynthBench datasets are bounded.** Enum (opinionsqa, pewtech, eurobarometer, ntia, subpop, wvs, gss, michigan), Likert (globalopinionqa, wvs, gss, 0-10 scales), or binary (ntia).

**Overlap matrix is sparse:**

| Instrument | Datasets with MED overlap | Datasets with LOW or none |
|---|---|---|
| **general-survey** | pewtech, opinionsqa, globalopinionqa, michigan, wvs, gss, subpop | eurobarometer (LOW), ntia (LOW) |
| product-feedback (NPS) | pewtech, globalopinionqa | others |
| churn-diagnosis | globalopinionqa, pewtech, michigan, wvs, subpop | opinionsqa, eurobarometer (all LOW) |
| market-research | wvs | others LOW |
| feature-prioritization | pewtech (LOW) | all others: none |
| pricing-discovery | — | all: none |
| landing-page-comprehension | — | all: none |
| name-test | — | all: very weak |

**Easy comparisons (no transformation):**
1. `general-survey × {pewtech, opinionsqa, globalopinionqa}` — free-text opinion Q extracted as enum via `PICK_ONE_SCHEMA` → JSD vs enum distribution
2. `product-feedback NPS × globalopinionqa` — extract 0-10 integer → Likert distribution, compare directly

**Moderate (light transformation):** `product-feedback feature-satisfaction × pewtech`, `churn-diagnosis × michigan` — require sentiment/category mapping.

**Hard or impossible:** pricing-discovery, landing-page-comprehension, name-test, feature-prioritization — SynthBench has no corresponding ground-truth distributions.

## Observable seams (integration points)

1. **`load_synthbench_baseline(spec)` is already implemented.** Returns baseline payload dict (human distribution + convergence metadata). Currently spliced under `convergence.human_baseline`.

2. **Per-question metric attachment is a near-miss.** `convergence.per_question[key]` already exists with JSD-based curves. Adding `convergence.per_question[key].human_jsd` (model-distribution vs human-distribution single number) is additive; wires into existing `_run_check_locked()` in `convergence.py`.

3. **Baseline question registry does not exist.** If instruments tagged questions with SynthBench keys (e.g., `--- synthbench-key: gss:SPKATH`), `load_synthbench_baseline()` could accept a full mapping payload rather than one question at a time. Enables bulk calibration without manual alignment.

4. **`synthesis.per_question_synthesis`** — map-reduce intermediate summaries keyed by question index. Empty today (not consumed). A sibling `synthesis.per_question_calibration` is the natural home for calibration metrics if we surface them in the synthesis block rather than a convergence sibling.

5. **`extract_schema` path** — `src/synth_panel/structured/` already transforms free-text responses into structured data via tool-use. The last-mile pipeline from free-text → enum/likert → distribution → JSD already exists component-wise; wiring is the question.

6. **MCP server tool surface** — `run_panel` today returns structured panel data. An adjacent `compare_against_synthbench(panel_result, dataset, question_key)` tool is a pure-function add-on.

## Design-space tensions (for D-phase discussion)

**Inline-during-run vs post-hoc tension.** (a) `--calibrate-against DATASET:QUESTION` during `panel run` couples calibration to the run and pays live cost; (b) `synthpanel calibrate RESULT.json --against DATASET` as post-hoc subcommand decouples but requires a second invocation. Both have legitimate use cases.

**Automatic vs explicit alignment tension.** Synthpanel has no `--synthbench-key` tagging convention today. Two forks: (i) require instruments to tag questions with SynthBench keys (schema change, instrument migration cost), or (ii) provide a `--calibrate-against DATASET:QUESTION` flag where user specifies alignment per panel.

**Which datasets are practical for inline.** Only `gss` and `ntia` are redistribution-full. Inline calibration on `gated` datasets requires either an authenticated cache or a user's own local copy. Realistic inline UX likely focuses on GSS for the first wedge.

**Output-shape tension.** Single JSD number per question? A full curve? A narrative? The product signal (from panelist audit Q4) asked for "the number surfaced right next to the panel result" — suggests a scalar or small struct, not a narrative. But aligning that to the existing `convergence` block vs carving a new `calibration` block is a real choice.

**Dataset coverage reality-check.** 5 of 8 instruments have **no overlap** with any SynthBench dataset. Inline calibration's value is constrained by how often users hit the overlap zone (general-survey is the best-covered; pricing/landing-page/name-test are empty quadrants). Feature may have narrower audience than the panelist ask suggested.

**Coupling tension.** The `[convergence]` extra already exists. Does inline calibration live there (keeping the fence) or graduate into core? Adjacent question: if GSS is the only practical inline source, does calibration belong behind its own `[calibration]` extra or under `[convergence]`?

**Bounded-question requirement.** All synthpanel bundled instruments are free-text today. Practical inline calibration requires either (a) instrument authors adding `response_schema` with enum/likert types, or (b) runtime extraction via `--extract-schema` that produces bounded distributions. Either is a real pipeline. Neither is free.

## What the research does not answer

- Whether surfacing calibration inline is more valuable as a research-validity signal (tight integration, few datasets) or as a marketing-visible number (broader coverage, shallower signal)
- How much adoption friction the `--synthbench-key` tagging convention adds for instrument authors
- Whether the first-wedge target is `general-survey × pewtech` (most-overlapping pair) or `product-feedback × globalopinionqa` (easiest NPS conversion)
- Whether inline calibration should drive future instrument authoring (new bundled instruments with known-good SynthBench analogues)
- The latency cost of running a SynthBench calibration pass inline vs post-hoc on typical panel sizes (n=50 to n=10k)

## Files

- `research/synthbench-flow-map.md`
- `research/instrument-dataset-coverage.md`

## Ready for D-phase gate

No design direction proposed. The path from panelist response to JSD number is visible in code today; the architectural choice is *where* to plug it in, *which* audience to optimize for, and *which* dataset coverage to treat as the realistic wedge. Human brain-surgery gate next.
