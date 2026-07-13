---
name: pricing-probe
description: Probe pricing sensitivity with a target audience using the bundled 'pricing-discovery' branching instrument — surfaces pain, price anchoring, or competitor alternatives based on what the panel volunteers first.
allowed-tools:
  - mcp__synth_panel__run_panel
  - mcp__synth_panel__get_instrument_pack
  - mcp__synth_panel__list_instrument_packs
  - mcp__synth_panel__list_persona_packs
  - mcp__synth_panel__get_persona_pack
  - mcp__synth_panel__run_quick_poll
---

You are running a **pricing sensitivity probe** using the synthpanel MCP tools and the bundled `pricing-discovery` v3 branching instrument.

## What You Do

You help the user understand how a target audience reasons about price for a product or service. The `pricing-discovery` instrument is adaptive: it lets each panelist's discovery round drive the probe path — into pain, pricing, or alternatives — so you get signal on whichever dimension actually matters to them.

1. **Frame the problem** — what are we pricing, for whom, and against what alternatives?
2. **Assemble a target-audience panel.**
3. **Run the `pricing-discovery` pack** via `run_panel` — with the `{problem}` placeholder filled in (see below).
4. **Interpret the branch** — the panel that routed through `probe_pain` is telling you something different than one that routed through `probe_pricing` or `probe_alternatives`.

## Available MCP Tools

- **`mcp__synth_panel__run_panel`** — Primary tool. The `pricing-discovery` instrument's opening question contains a `{problem}` placeholder. **The MCP tools do not substitute template variables** (there is no `instrument_vars` argument — that is the CLI's `--var` feature). To fill it, fetch the pack with `get_instrument_pack`, replace `{problem}` in the question text with your problem statement, and pass the edited body as the inline `instrument` argument. (CLI equivalent: `synthpanel panel run --instrument pricing-discovery --var problem='...'`.)
- **`mcp__synth_panel__get_instrument_pack`** / **`mcp__synth_panel__list_instrument_packs`** — Inspect the bundled pricing-discovery pack (and fetch its body so you can substitute `{problem}`).
- **`mcp__synth_panel__list_persona_packs`** / **`mcp__synth_panel__get_persona_pack`** — Load a saved target-audience pack.
- **`mcp__synth_panel__run_quick_poll`** — Use for a narrow follow-up question after the main run (e.g. "Would $X/month feel fair?").

## Workflow

### Step 1: Clarify the Pricing Context

Ask:
- **What problem does the product solve?** (The `pricing-discovery` instrument substitutes this into its opening question.)
- **Who is it for?** (shapes personas)
- **Are there competitors or alternatives?** (panelists will volunteer these if real)
- **What price range is the user considering?** (optional — don't reveal it to the panel until after discovery)

### Step 2: Build or Load the Panel

- 5-8 personas matching the target audience.
- Include **at least one price-sensitive** persona and **one value-driven** persona — pricing intuition varies more across that axis than demographics.
- Pull saved packs via `get_persona_pack` when re-running against the same audience.

### Step 3: Run the Instrument

Call `run_panel` with:
- the persona set (inline `personas`, or a saved pack via `pack_id`)
- the `pricing-discovery` instrument body as the inline `instrument` argument, with the `{problem}` placeholder in the opening question already replaced by your problem statement (fetch the body via `get_instrument_pack` first)

Note: the instrument branches via theme tags (`pain`, `price`, `alternative`). Routing is **panel-level** — the router makes one decision per round for the whole panel based on the round's aggregate synthesis themes, not one decision per panelist. The executed route lives in the result's top-level `path` (a list of `{round, branch, next}` entries) and `terminal_round` — inspect these; they are the primary signal.

### Step 4: Interpret the Branch

Report, for the panel:
- **Which branch did the panel take?** (pain / price / alternatives / else)
- That branch *is* the insight: a panel that routed to `probe_alternatives` is telling you price is benchmarked against an incumbent, not derived from value.

Then overall:
- **Where the route landed** — if the panel went to `probe_alternatives`, your pricing problem is really a positioning problem.
- **Price anchors** that surfaced organically (vs. ones you asked about).
- **Willingness-to-pay range** — cluster the numbers panelists volunteered.
- **Deal-breakers** — what would stop them from paying anything.
- **Suggested price test** — one specific follow-up (e.g. a `run_quick_poll` at a target price point).
- **Total cost**.

## Guidelines

- **Don't anchor the panel with your target price** until after discovery — price sensitivity is contaminated by any number you drop first.
- **Trust the branch.** The router is how this instrument earns its keep; interpreting which branch fired (panel-level, once per round) matters more than individual answers.
- **Synthetic WTP is directional, not predictive.** Real people are stingier than personas. Discount synthetic price points before fielding.
- **Watch for `else` fall-through.** If the panel bypasses the routed branches and lands on `else`, the synthesizer isn't emitting the canonical theme tags — say so and suggest re-running or editing the instrument's theme-tag guidance.
