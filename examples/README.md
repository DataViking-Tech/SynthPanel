# synthpanel examples

Drop-in YAML files for getting started. Pair any persona pack with any
instrument pack — the CLI takes both as separate `--personas` and
`--instrument` arguments.

## Files

| File | Kind | What it shows |
|------|------|---------------|
| [`personas.yaml`](personas.yaml) | Persona pack | Three personas used by every instrument example below |
| [`survey.yaml`](survey.yaml) | v1 instrument | Flat list of questions, no rounds — the simplest possible shape |
| [`multi-round-study.yaml`](multi-round-study.yaml) | v2 instrument | Three sequential rounds (`discovery` → `deep_dive` → `validation`) with linear `depends_on` |
| [`pricing-segmentation-study.yaml`](pricing-segmentation-study.yaml) | v3 branching | Demographic routing: splits into premium-tier vs value-tier probes based on income themes that surface in discovery |
| [`concept-test-ab.yaml`](concept-test-ab.yaml) | v3 branching | A/B concept test with a conditional divergence probe driven by the `matches` (regex) predicate |

The `instruments/` subdirectory holds three additional standalone
instrument files (`general_survey.yaml`, `market_research.yaml`,
`product_feedback.yaml`) kept for backward compatibility. Prefer the
bundled packs (`synthpanel instruments list`) for new work.

## Running an example

```bash
# v1: single round, flat questions
synthpanel panel run \
  --personas examples/personas.yaml \
  --instrument examples/survey.yaml

# v2: linear multi-round
synthpanel panel run \
  --personas examples/personas.yaml \
  --instrument examples/multi-round-study.yaml

# v3: branching — pricing segmentation
synthpanel panel run \
  --personas examples/personas.yaml \
  --instrument examples/pricing-segmentation-study.yaml

# v3: branching — A/B concept test
synthpanel panel run \
  --personas examples/personas.yaml \
  --instrument examples/concept-test-ab.yaml
```

## Inspecting a branching instrument

Render the round DAG to sanity-check routing before you spend tokens:

```bash
synthpanel instruments graph examples/pricing-segmentation-study.yaml --format mermaid
synthpanel instruments graph examples/concept-test-ab.yaml --format mermaid
```

Validate a file without running it (parse-only — fails on bad DAGs,
missing `else` clauses, unreachable rounds, etc.):

```bash
synthpanel instruments install examples/concept-test-ab.yaml --name concept-test-ab
```

`install` parses the file through the full v3 validator before writing
anything to the pack directory, so a successful install means the
instrument is well-formed.

## Authoring v3 branching instruments

Two gotchas are worth internalizing before you write your own:

1. **Theme predicates match exact substrings.** `contains: price` only
   fires if the synthesizer emitted a theme containing the literal
   substring `price`. Add a canonical-tag comment block at the top of
   your instrument (both branching examples above include one) so the
   synthesizer prefers those tags. See the main
   [README "Theme Matching" section](../README.md#theme-matching-the-r3-caveat)
   for the full explanation.

2. **`else` is mandatory.** Every `route_when` block must end with an
   `else:` clause. The target can be another round name or the
   reserved sentinel `__end__` (which terminates the run and triggers
   final synthesis over the path traversed so far).

Predicate operators:

| Op | Meaning | Use when |
|----|---------|----------|
| `contains` | Substring match | You control the canonical tag vocabulary (most cases) |
| `equals` | Exact string match | You know the synthesizer emits one of a fixed small set |
| `matches` | Python regex | You want to catch a family of related phrasings (see `concept-test-ab.yaml`) |
