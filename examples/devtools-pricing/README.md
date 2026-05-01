# Developer Tools Pricing Study

**Question this example answers:** *Across the experience-level spread
of developers, where does each cohort actually anchor on price for
a tiered devtool — and what specifically tips them up or down a tier?*

A v3 branching panel: discovery surfaces the panel's anchor tier,
then routes into the right tier-specific probe. Each probe ends with
a Likert (scale) question so you can quantify intent across cohorts.

## Composition

- **12 developers**, evenly split across four experience cohorts:
  - 3 junior (0–3 yrs, free-tier-driven)
  - 3 mid (3–7 yrs, small-team approval authority)
  - 3 senior (7–12 yrs, internal advocate, ~$20k discretion)
  - 3 staff (12+ yrs, enterprise-tier procurement)
- Roles span frontend, backend, mobile, platform, ML, devops, and
  SRE so pricing reactions aren't biased by a single specialty.

## How the routing works

The discovery round prompts each panelist about a 4-tier pricing
structure (Free / Pro $20 / Team $60 / Enterprise contact). The
synthesizer is asked to tag the panel-level anchor with one of:

| Tag | Routes to | Use |
|-----|-----------|-----|
| `wtp_enterprise` | `probe_enterprise` | Panel willing to engage sales |
| `wtp_team` | `probe_team_tier` | Panel anchors on $60/seat |
| `wtp_pro` | `probe_pro_tier` | Panel anchors on $20/seat |
| `objection` | `probe_objections` | Strong objection regardless of tier |
| (else) | `probe_pro_tier` | Fallback — most common landing |

This is a **substring match** (R3 caveat) — the synthesizer must emit
one of these literal tags. The instrument's preamble lists the
canonical vocabulary so the synthesizer prefers them.

## What the output looks like

You get four kinds of signal:

1. **Tier anchor distribution** — does your panel land on Pro, Team,
   or Enterprise? Junior cohorts will skew toward Free → Pro; staff
   cohorts toward Team → Enterprise.
2. **Tier-specific objections** — the probe rounds extract candid
   objections within each tier. The "compliance-tax" objection (SSO
   gated behind Team) is particularly valuable.
3. **Likert intent scores** — convert-to-Pro likelihood, walk-away
   rate from opaque enterprise pricing, etc. Aggregate these by
   cohort to see where intent is concentrated.
4. **A pricing change recommendation per panelist** in `wrap_up`.

## Run it

```bash
synthpanel panel run \
  --personas examples/devtools-pricing/personas.yaml \
  --instrument examples/devtools-pricing/instrument.yaml
```

Render the routing graph (4-way branch from discovery):

```bash
synthpanel instruments graph \
  examples/devtools-pricing/instrument.yaml --format mermaid
```

Validate without running (catches DAG / theme issues):

```bash
synthpanel instruments install \
  examples/devtools-pricing/instrument.yaml --name devtools-pricing-test
```

## Pack substitution

The bundled [`developer.yaml`](../../src/synth_panel/packs/developer.yaml)
pack covers similar ground with 16 personas spanning a wider stack.
For richer breadth:

```bash
synthpanel panel run \
  --personas-pack developer \
  --instrument examples/devtools-pricing/instrument.yaml
```

The instrument is pricing-structure-agnostic — substitute your real
tier numbers in the `discovery` round prompt and rerun.
