# SaaS Onboarding Friction Study

**Question this example answers:** *Where does our onboarding actually
break down for mid-market SaaS users, and how severe is each break?*

A three-round v2 panel: free-text recall, Likert severity rating, and
a "would-fix" round that anchors on the themes the panel surfaced.

## Composition

- **8 personas**, deliberately mixed across three lifecycle stages:
  - 3 currently in trial
  - 3 post-trial converts (paying 30–90 days)
  - 2 churned (signed up, cancelled in trial)
- Mid-market only (180–600 employee companies). Enterprise and
  prosumer onboarding have different failure modes — a deliberately
  narrow study finds sharper signal.

## What the output looks like

Round 1 (`recall`) produces vivid first-48-hour stories — these are
the qualitative spine. Round 2 (`severity`) attaches a 1–5 score to
the friction so you can sort panelists by how painful their experience
was. Round 3 (`would_fix`) gives you a list of concrete proposed
changes per panelist, each with a likely-to-recover-churn score.

When you aggregate across the panel, expect to see:

- 1–2 themes dominating recall (these are your headline issues)
- A bimodal severity distribution: trial-stage and churned panelists
  rate higher; post-trial converts rate lower because they pushed
  through. This bimodality IS the signal — it tells you which friction
  stops people vs. which is just annoying
- A short list of high-confidence fixes from the would_fix round

## Run it

```bash
synthpanel panel run \
  --personas examples/saas-onboarding-friction/personas.yaml \
  --instrument examples/saas-onboarding-friction/instrument.yaml
```

Render the round structure:

```bash
synthpanel instruments graph \
  examples/saas-onboarding-friction/instrument.yaml --format mermaid
```

## Pack substitution

If you need to extend the panel beyond 8, the bundled
[`enterprise-buyer`](../../src/synth_panel/packs/enterprise-buyer.yaml)
pack overlaps on the decision-maker personas (VPs, Directors). Layer
it in or substitute as the persona pack to widen the panel.

## Tweaking

- Drop the `severity` round if you want pure qualitative recall —
  the example will still work as a 2-round flow.
- Swap a `theme_0` reference for a domain term (e.g. "{theme_0}"
  → "the SSO requirement") if you want to anchor the round on a
  specific friction rather than the panel's emergent theme.
- For different onboarding flows, only the persona pack changes —
  the instrument is product-agnostic.
