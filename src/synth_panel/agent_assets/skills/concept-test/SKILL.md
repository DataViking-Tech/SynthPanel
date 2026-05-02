---
name: concept-test
description: Validate a product concept or value proposition with a target audience — surfaces whether the problem is real, whether the solution lands, and what would prevent adoption.
allowed-tools:
  - mcp__synth_panel__run_panel
  - mcp__synth_panel__run_quick_poll
  - mcp__synth_panel__list_persona_packs
  - mcp__synth_panel__get_persona_pack
  - mcp__synth_panel__save_persona_pack
  - mcp__synth_panel__list_instrument_packs
---

You are running a **concept test** using the synthpanel MCP tools.

## What You Do

You help the user pressure-test an early-stage product concept, value prop, or feature idea against a specific target audience before they build or ship it. The goal is to surface whether the problem is real and whether the proposed solution actually addresses it — not to validate a decision that's already been made.

1. **Clarify the concept and the audience.**
2. **Design concept-oriented personas** drawn from (or biased toward) the target audience.
3. **Run the panel** with questions that probe pain, fit, willingness-to-adopt, and objections.
4. **Synthesize** — is the problem real, does the concept resonate, and what blocks adoption?

## Available MCP Tools

- **`mcp__synth_panel__run_panel`** — Primary tool. Run a multi-question panel with target-audience personas.
- **`mcp__synth_panel__run_quick_poll`** — Use for a single "would you try this?" temperature check.
- **`mcp__synth_panel__list_persona_packs`** / **`mcp__synth_panel__get_persona_pack`** — Reuse saved target-audience packs.
- **`mcp__synth_panel__save_persona_pack`** — Save a new audience pack if the user is likely to re-test.
- **`mcp__synth_panel__list_instrument_packs`** — Check for bundled packs that fit (e.g. `product-feedback`, `market-research`).

## Workflow

### Step 1: Frame the Concept

Ask the user for:
- A 2-3 sentence description of the concept (what it is, who it's for, what problem it solves).
- The target audience (role, demographic, or psychographic).
- What decision this test is meant to inform ("should we keep exploring?" vs. "which direction?").

### Step 2: Build or Load Personas

- 4-8 personas, biased toward the target audience but with **at least one skeptic** and **at least one adjacent non-target** to stress-test the boundaries.
- Each persona should have a plausible reason the problem may or may not apply to them.

### Step 3: Design the Instrument

Include questions in this order:
1. **Problem probe** — "Does this describe something you've actually experienced?" (don't lead with the solution)
2. **Concept reveal** — present the concept, ask for gut reaction.
3. **Fit** — "Who do you know this would be for?" (reveals whether they see themselves in it)
4. **Adoption blockers** — "What would stop you from trying this?"
5. **Willingness** — rough price sensitivity or alternative-they'd-pick.

Add 1-2 follow-ups on the concept reveal for depth.

### Step 4: Run and Synthesize

After results return, report:
- **Problem validity** — did panelists actually experience the problem, or did you have to convince them?
- **Concept resonance** — who lit up, who shrugged, who pushed back.
- **Top 2-3 adoption blockers** across panelists.
- **Surprising insights** — anything you didn't expect.
- **Recommended next step** — iterate the concept, test with real users, or drop it.
- **Total cost**.

## Guidelines

- **Don't sell the concept to the panel.** Describe it neutrally. Leading questions produce leading answers.
- **Respect dissent.** If 2 of 6 panelists hate it, that's signal — don't average it away.
- **Distinguish "I'd use this" from "This is a good idea."** Only the first predicts adoption.
- **Concept tests are exploration, not validation.** Say so when the user asks "should we build this?" — the honest answer is "this is one data point; go talk to real users."
