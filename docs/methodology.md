# Methodology

> **Audience:** ML and data-engineering evaluators — the inspectability buyers.
> If you're deciding whether SynthPanel's panels are credible enough to put in
> front of a real research decision, this page is the proof artifact.

> **Cover image (placeholder):** `convergence-decay-vs-cohort-size.png` — to
> be embedded at release. Plot shows convergence score variance narrowing as
> cohort size grows, with the auto-stop cut-off line annotated.

## Why this page exists

The discovery panel for v1.0.0 made one thing clear: the agent-builder cohort
prioritizes **inspectability over polish**. Black-box synthetic-population
claims are rejected on contact. Direct quote from the synthesis: *"Panelists
care more about whether they can see the tool break than whether it never
breaks — observability beats reliability in stated preference."*

So this page documents what's actually happening behind a panel call. No
marketing language, no "AI-powered representativeness." If you find a hole in
the methodology, [open an issue](https://github.com/DataViking-Tech/SynthPanel/issues) —
that's the loop we want.

## Cohort construction

A panel run resolves three layers:

1. **Persona pack** — a YAML file or installed pack name. Each persona is a
   `{name, age, occupation, background, personality_traits, ...}` record;
   bundled packs are versioned and visible in `synthpanel pack list`.
2. **Prompt template** — `templates/current.txt` by default; alternates exist
   for ablation (`templates/minimal.txt`, `templates/demo.txt`,
   `templates/values.txt`). The template determines which persona fields land
   in the model's system prompt.
3. **Model assignment** — single model (`--model`) or weighted ensemble
   (`--models haiku:0.33,gemini:0.33,gpt-4o-mini:0.34`). Each persona is
   interviewed by every assigned model independently when `--blend` is on.

There is **no synthetic resampling** — every persona in the realized panel
came from your input pack. If you ask for 50 panelists from a 15-persona
pack, the orchestrator either expands the pack via documented pack-merge
rules or surfaces the gap as `demographic_skew` post-run.

## Sampling

For panels above 500, SynthPanel tracks **response-distribution
convergence** live via Jensen-Shannon divergence per question and can
auto-stop once every bounded question (Likert / yes-no / pick-one / enum) has
stabilized.

```bash
synthpanel panel run \
  --personas large-panel.yaml \
  --instrument pricing-discovery \
  --convergence-check-every 20 \
  --auto-stop \
  --output-format json > result.json
```

The output's `convergence` block exposes the smallest `n` at which each
question converged, so you can confidently size down the next run. Full
methodology and tuning live in [docs/convergence.md](convergence.md).

## What `convergence` actually measures

A 0–1 score on the `panel_verdict.json` envelope. Higher = panelists agree
more.

| Range | Reading |
|---|---|
| 0.85+ | High agreement — the cohort lands on a single position. |
| 0.65–0.85 | Mixed — credible verdict, real dissent worth reading. |
| 0.45–0.65 | Split — verdict is contested; treat as *one* of several signals. |
| < 0.45 | Low — the orchestrator will usually attach `low_convergence` flag. |

It is computed from inter-persona variance on the synthesizer's primary
measure for the question class (Likert mean, top-pick share, etc.), not from
embedding similarity of free text. **Your agent can threshold on it** —
that's the design intent. See
[docs/response-contract.md](response-contract.md) for the field's contract.

## Observed disagreement-rate baselines

> Numbers in this section come from the SynthBench test corpus and the
> v1.0.0 acceptance suite. See [SynthBench leaderboard](https://synthbench.org)
> for the live, reproducible source.

- **Single model, n=15 panel:** convergence variance ~ ±0.08 across reruns
  with fixed personas. Variance is dominated by model-side stochasticity, not
  cohort sampling.
- **3-model ensemble (haiku + gemini-flash + gpt-4o-mini), n=15:**
  convergence variance ~ ±0.05. Ensemble blending narrows the band but does
  not collapse it — disagreement is a real signal, not just noise.
- **Persona collisions:** above cosine 0.92 between persona embeddings,
  agreement scores inflate by ~0.04–0.07 vs. a deduplicated pack. The
  `persona_collision` flag fires above this threshold.

The full leaderboard (with confidence intervals) is at
<https://synthbench.org/leaderboard>; submit your own runs via
`--submit-to-synthbench`.

## Failure transparency

This is what the cohort actually rejected black-box claims over. Every run
exposes:

- **`flags[]`** — closed enum of seven well-defined failure modes, each with
  severity. See [docs/response-contract.md#flags-closed-enum](response-contract.md#flags-closed-enum).
- **`extension[]`** — open observability channel for non-enum signals. Logged,
  not branched on.
- **`warnings`** — parser, sampling-truncation, and synthesis warnings
  surfaced per turn.
- **`run_invalid: true`** — set on degenerate aborts (rate-exhaustion,
  cost-gate halt, all-provider failure) with a specific `abort_reason`.
- **Full transcript URI** — every panelist's turn is saved to JSONL and
  resolvable via the `panel-result://` MCP resource. Nothing is summarized
  away.

The `synthpanel report` post-hoc renderer opens with a mandatory
synthetic-panel banner (and closes with a matching footer) so the output
can't be mistaken for real-user research. **Synthetic panels are for
exploration, hypothesis generation, and rapid iteration. They do not replace
real users.** That banner is non-removable — it's a methodology guard, not a
style choice.

## Known limits

- Synthetic responses tend to cluster around means.
- LLMs exhibit sycophancy (tendency to please).
- Cultural and demographic representation has blind spots — the
  `demographic_skew` flag surfaces this on a per-run basis.
- Higher-order correlations between variables are poorly replicated.
- `convergence` measures inter-persona agreement on the synthesizer's primary
  axis. It is not a calibration score against a real-human distribution; for
  that, run with `--calibrate-against DATASET:QUESTION` and inspect the
  per-question JSD field.

## Read next

- [docs/response-contract.md](response-contract.md) — what the verdict
  envelope contains, field by field.
- [docs/convergence.md](convergence.md) — JSD-based live convergence and
  auto-stop methodology.
- [docs/synthbench-integration.md](synthbench-integration.md) — calibration
  against real-human distributions.
- [docs/migration-v1.md](migration-v1.md) — what the v1.0.0 contract freeze
  changes for callers.
