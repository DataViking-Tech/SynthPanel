---
name: name-test
description: Quickly test 1-3 candidate product or feature names with a synthetic panel — surfaces confusion, pronounceability, and memorability concerns in one pass.
allowed-tools:
  - mcp__synth_panel__run_quick_poll
  - mcp__synth_panel__run_panel
  - mcp__synth_panel__list_persona_packs
  - mcp__synth_panel__get_persona_pack
  - mcp__synth_panel__list_instrument_packs
  - mcp__synth_panel__get_instrument_pack
---

You are running a **quick name test** using the synthpanel MCP tools.

## What You Do

You help the user decide between candidate names for a product, feature, or brand. The workflow is deliberately short — a name test should take minutes, not hours.

1. **Collect candidates** — 1 to 3 names, plus a one-line description of what the thing actually is.
2. **Pick a lens** — quick single-question poll, or the branching `name-test` instrument pack for deeper probing.
3. **Run the panel** — small, diverse personas; keep it cheap.
4. **Report the verdict** — winner, loser, and the specific concern that tipped each (confusion, pronunciation, memorability).

## Available MCP Tools

- **`mcp__synth_panel__run_quick_poll`** — Single-question poll across personas (fastest, cheapest).
- **`mcp__synth_panel__run_panel`** — Full panel run using the bundled `name-test` branching instrument that probes meaning, pronounceability, or memorability based on first reactions. Its opening question has a `{candidates}` placeholder; see Step 2 for how to fill it.
- **`mcp__synth_panel__list_persona_packs`** / **`mcp__synth_panel__get_persona_pack`** — Reuse saved personas instead of inventing new ones.
- **`mcp__synth_panel__list_instrument_packs`** — Confirm the `name-test` pack is available.

## Workflow

### Step 1: Gather Inputs

Ask for:
- The candidate names (comma-separated).
- A one-sentence description of what the product/feature does.
- Target audience (so personas are relevant).

### Step 2: Choose Depth

- **Quick gut check** → `run_quick_poll` with a question that has the description and candidate names written directly into the `question` string, e.g. *"Which of these names best fits a budget travel app — Wander, Roamly, Tr9? Why?"*
- **Full branching evaluation** → `run_panel` with the bundled `name-test` instrument. The MCP tools do **not** substitute template variables (there is no `instrument_vars` argument — that is the CLI's `--var` feature), so fetch the pack with `get_instrument_pack`, replace the `{candidates}` placeholder in the opening question with the actual comma-separated names, and pass the edited body as the inline `instrument` argument. (CLI equivalent: `synthpanel panel run --instrument name-test --var 'candidates=Name A, Name B'`.) The instrument branches into meaning-probe, pronounce-probe, or memorability-probe based on what surfaces first.

### Step 3: Run

Default to 4-6 personas. If the user hasn't supplied them, generate a small diverse set tuned to the target audience, or pick a saved pack via `get_persona_pack`.

### Step 4: Report

Produce a tight summary:
- **Winner** and why (quote 1-2 panelists).
- **Loser** and the specific failure mode (misread, hard to say, forgettable).
- **Risks for the winner** — concerns that still showed up even for the preferred name.
- **Total cost** of the run.

## Guidelines

- Keep it small — 4-6 personas is plenty for a name test.
- Don't over-interpret a single panel — flag when reactions were genuinely mixed rather than forcing a winner.
- If all names scored poorly, say so and suggest the user brainstorm a new set instead of picking the least-bad.
- Name tests are exploratory signal, not validation. Say this when the user asks "should we ship this?"
