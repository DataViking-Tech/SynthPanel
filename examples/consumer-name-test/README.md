# Consumer Name Test

**Question this example answers:** *Of 5 candidate names for a
consumer product, which one wins on gut reaction across age and
life-stage — and which would embarrass people to recommend?*

The simplest possible v1 instrument: 6 panelists react to 5 names
with a forced-choice (enum) winner pick, a 1–5 confidence score, and
candid follow-ups about wear-in and recommendation friction.

## Composition

- **6 personas** spanning age 22–64, deliberately spread across
  life-stages and tech-comfort levels so name reactions aren't
  biased by a single demographic. Sweet spot for name-tests:
  enough voices to surface obvious patterns, few enough that each
  voice gets weight.
- Includes a deliberate "low-tech, gut-reaction-driven" persona
  (Frank, 64) — these voices catch made-up-feeling names that
  digital-native panelists may give a pass to.

## What the output looks like

You get four artifacts per panelist:

1. **Free-text gut reaction** — favorite + loser, unfiltered.
2. **Forced-choice winner** (one of the 5 names) — the enum schema
   makes this machine-readable for tallying.
3. **Confidence score** (1–5) — surfaces whether the panel's choice
   is high-conviction or coin-flip.
4. **"Embarrassed to recommend" name** — often the most useful
   datum. A name that no one picks but everyone hates is a clear
   reject; a name that wins but embarrasses one cohort is a quiet
   risk you should probably catch before launch.

When you aggregate, expect:

- A clear plurality winner OR a 2-name tie (treat ties as a real
  signal, not noise — N=6 isn't statistically significant; the
  tie is telling you both names work)
- One outlier name that everyone winces at (your reject)
- A confidence histogram — bimodal usually means panelists who
  picked different names also picked them with different conviction

## Run it

```bash
synthpanel panel run \
  --personas examples/consumer-name-test/personas.yaml \
  --instrument examples/consumer-name-test/instrument.yaml
```

This is a v1 instrument so there is no DAG to render.

## Filling in the candidates

Before running, substitute `{name_a}` through `{name_e}` and
`{product_category}` with your real candidates. The simplest path is
to copy the instrument and edit in place:

```bash
cp examples/consumer-name-test/instrument.yaml my-name-test.yaml
# edit the {name_*} placeholders in my-name-test.yaml
synthpanel panel run \
  --personas examples/consumer-name-test/personas.yaml \
  --instrument my-name-test.yaml
```

You can also pass placeholders through `--prompt-context` if your
CLI version supports it; see the main README for current syntax.

## Pack substitution

For a wider 8-persona panel, swap to the bundled
[`general-consumer.yaml`](../../src/synth_panel/packs/general-consumer.yaml)
pack:

```bash
synthpanel panel run \
  --personas-pack general-consumer \
  --instrument examples/consumer-name-test/instrument.yaml
```

The instrument is name-agnostic — it works on any 5-name slate as
long as you fill in the placeholders.

## Why a v1 (not branching) for name tests

Name tests benefit from being short. Each additional round dilutes
gut-reaction signal — by round 3 panelists are rationalizing rather
than reacting. The v3 [concept-test-ab](../concept-test-ab.yaml)
example shows the right shape for *concept* tests where you want
panelists to think harder; name tests want the opposite.
