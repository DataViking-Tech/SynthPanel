# synthpanel examples

Drop-in YAML files for getting started. Pair any persona pack with any
instrument pack — the CLI takes both as separate `--personas` and
`--instrument` arguments.

## Generic shapes

These pair the shared [`personas.yaml`](personas.yaml) with one
instrument apiece, so you can compare schema versions side-by-side
without distraction.

| File | Kind | What it shows |
|------|------|---------------|
| [`personas.yaml`](personas.yaml) | Persona pack | Three personas used by every instrument example below |
| [`survey.yaml`](survey.yaml) | v1 instrument | Flat list of questions, no rounds — the simplest possible shape |
| [`multi-round-study.yaml`](multi-round-study.yaml) | v2 instrument | Three sequential rounds (`discovery` → `deep_dive` → `validation`) with linear `depends_on` |
| [`pricing-segmentation-study.yaml`](pricing-segmentation-study.yaml) | v3 branching | Demographic routing: splits into premium-tier vs value-tier probes based on income themes that surface in discovery |
| [`concept-test-ab.yaml`](concept-test-ab.yaml) | v3 branching | A/B concept test with a conditional divergence probe driven by the `matches` (regex) predicate |

## Industry-specific runnable examples

Each is a complete `personas.yaml` + `instrument.yaml` + `README.md`
under its own directory — built around a real research question, not a
schema demo. Use these as starting points for your own studies.

| Directory | Panel | Pattern | Question it answers |
|-----------|-------|---------|---------------------|
| [`saas-onboarding-friction/`](saas-onboarding-friction/) | 8 mid-market SaaS users (trial / post-trial / churned) | v2 multi-round, Likert + free-text | Where does our onboarding break down, and how severe is each break? |
| [`devtools-pricing/`](devtools-pricing/) | 12 developers (junior → staff) | v3 branching, 4 tier-specific probes | Where does each developer cohort anchor on price, and what tips them up or down a tier? |
| [`consumer-name-test/`](consumer-name-test/) | 6 general consumers (age 22–64) | v1 flat, enum forced-choice + Likert | Of 5 candidate names, which wins on gut reaction across age and life-stage? |

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

# Industry-specific: SaaS onboarding friction (v2)
synthpanel panel run \
  --personas examples/saas-onboarding-friction/personas.yaml \
  --instrument examples/saas-onboarding-friction/instrument.yaml

# Industry-specific: developer tools pricing (v3 branching)
synthpanel panel run \
  --personas examples/devtools-pricing/personas.yaml \
  --instrument examples/devtools-pricing/instrument.yaml

# Industry-specific: consumer name test (v1)
synthpanel panel run \
  --personas examples/consumer-name-test/personas.yaml \
  --instrument examples/consumer-name-test/instrument.yaml
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
