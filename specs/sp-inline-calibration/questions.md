# Q-phase: Cross-project calibration signal flow

*Ticket hidden per QRSPI discipline. These questions drive objective code-mapping only; do not invent design direction from them.*

## Problem (shared with R polecats)

SynthBench computes distributional comparisons (JSD, Kendall's tau, convergence curves) between synthpanel output and real human survey data. Map the current data flow end-to-end: how does synthpanel output reach SynthBench, what transformations happen on each side, where does the comparison result live, and who reads it.

## Research dimensions

### 1. SynthBench's ingestion of synthpanel output

- How does SynthBench discover a synthpanel panel result today? Is there a provider adapter? A file-watcher? A manual invocation?
- `src/synthbench/providers/synthpanel.py` exists — what does it expose, and what's its call contract?
- What subset of the panel-result JSON does SynthBench read? Which fields are load-bearing for the metric computation?
- Is the provider pattern idempotent / re-runnable on the same panel result? Cached?

### 2. SynthBench's dataset adapters + Question abstraction

- `Question` in `src/synthbench/datasets/base.py` carries `human_distribution` — how is that distribution keyed (option names? ordinal indexes? enum tokens?)
- How does SynthBench match a synthpanel output question to a SynthBench dataset question? By key, text similarity, explicit mapping, manual alignment?
- Which of the nine datasets have redistribution-full status vs gated vs aggregates-only? What does that mean for inline consumption?
- Is there any "question registry" that synthpanel could reference by name to say "this question is the same as GSS's SPKATH"?

### 3. Metric computation and output shape

- Where is JSD computed (`src/synthbench/metrics/distributional.py`)? What's its input contract — two distribution dicts, a pair of Questions, paired arrays?
- What's the output shape of a full SynthBench run against a synthpanel provider — single JSD per question, a table, a leaderboard entry, a published artifact?
- What's the lifecycle of a computed calibration metric? In-memory only, persisted to disk, uploaded to R2, published to the leaderboard site?
- Are there any lightweight / cheap versions of the metric suitable for inline use (e.g. per-question single-number JSD vs full report)?

### 4. Cross-package dependency posture

- Does synthpanel already import synthbench anywhere, or is the provider one-way (synthbench → synthpanel)?
- What's synthpanel's extras mechanism (`pyproject.toml` → `[project.optional-dependencies]`)? Are there precedents for optional extras that pull in a specific sibling package?
- What's SynthBench's current install size / dependency tree? Would requiring it inline affect synthpanel's PyPI install experience?
- How does the existing `--convergence-baseline` flag (sp-yaru) intend to consume SynthBench? Is that path already built or speculative?

### 5. Inline-consumption access patterns in the codebase

- What does synthpanel currently do with per-question output that could attach a metric? (e.g. `per_model_results[*].results[i]` is the row level; `per_question_synthesis` is the map-reduce per-question summary.)
- Is there a natural "per question" structure today, or is the output flattened across questions?
- Where in the output JSON shape would additional metric data land without breaking backward compatibility? Which sibling keys are already present (`metadata`, `synthesis`, `convergence`, `warnings`)?
- How does sp-yaru's `convergence` section get populated — live during the run or post-hoc? What's its data shape?

### 6. User-facing surfaces for calibration signal

- What CLI subcommands exist today that explicitly reference validation / calibration / ground-truth? (`synthpanel analyze`, `synthpanel panel inspect`, others?)
- Does the MCP server expose any "compare against baseline" tool? What's its contract?
- How is the existing `cost_is_estimated: true/false` / `warnings: [...]` pattern documented? Is that a template for other sanity-check signals?
- What does the current `panel run` stderr output look like during a run? Is there a natural spot to surface an on-run calibration number, or is the output strictly post-run?

### 7. Dataset coverage vs synthpanel question types

- Which of synthpanel's 8 bundled instruments (product-feedback, market-research, feature-prioritization, pricing-discovery, landing-page-comprehension, churn-diagnosis, name-test, general-survey) have *any* topical overlap with any SynthBench dataset question?
- Conversely, for each SynthBench dataset, what kinds of synthpanel instruments would naturally ask questions that have a human distribution to compare against?
- Is the overlap primarily on bounded / enum questions (where distributions make sense), or free-text questions (where comparison is harder)?
- What's the precedent for free-text→categorical extraction in the codebase (the `--extract-schema` flag, `src/synth_panel/structured/*`)? Is that already plumbed into something SynthBench-adjacent?

## Deliverable

`specs/sp-inline-calibration/research/<dim>.md` per dimension, written objectively. Research phase synthesized into `research-map.md` for the human D-phase gate.
