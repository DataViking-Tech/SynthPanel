# VS Code Extension — Decision Doc

**Bead:** sp-2cw.7
**Date:** 2026-04-16
**Companion:** [docs/agent-integration-landscape.md § C.6](agent-integration-landscape.md)
**Status:** Decision made — not a build bead

---

## Decision: **DEFER**

Do not build a dedicated VS Code extension at this time. Revisit no earlier
than Q3 2026, and only if the revisit triggers below fire.

---

## Reasoning

### 1. MCP already covers the tool-integration path

VS Code AI Toolkit shipped first-class MCP support in March 2026. Any
VS Code user running Claude Code, Cursor, Copilot, or the AI Toolkit can
point their editor at `synthpanel mcp-serve` and get all 12 SynthPanel
tools natively. A custom extension cannot improve on that for tool
invocation — it can only add GUI surface.

### 2. The GUI surface is nice-to-have, not adoption-blocking

A dedicated extension would offer: persona-card editing, instrument flow
preview (Mermaid already works in the terminal via `synthpanel instruments
graph`), and result visualization panels. These are quality-of-life
improvements for power users, not the thing gating agent-framework adoption.
The top 3 unfiled gaps (Python SDK, framework examples, Composio) each
move the needle further per week of effort.

### 3. Effort/impact ratio is poor

- **Effort:** 2–4 weeks TypeScript for v1, plus ongoing maintenance
  (VS Code API churn, Marketplace re-submissions, security reviews,
  user support in a new channel).
- **Impact (landscape doc matrix):** MEDIUM impact, HIGH effort — the
  worst quadrant in the prioritization grid. Compare to the top 3, all
  MEDIUM–HIGH effort with HIGH impact.
- Maintainer cost compounds: every release requires testing a second
  distribution channel with its own failure modes.

### 4. Discovery value is real but capturable more cheaply

VS Code Marketplace ranking and badges are a legitimate discovery channel.
But cheaper substitutes exist and are not yet saturated:
- README "Works with VS Code (via MCP)" section pointing at AI Toolkit.
- An MCP-config snippet in the top-level README.
- `awesome-mcp-servers` registry entry (already filed — see
  `docs/registry-submissions.md`).
- Skills library expansion (sp-2cw.4) surfaces inside Claude Code, which
  is where the target audience already lives.

Exhaust these first.

### 5. No competitor pressure

Competitor check (per landscape doc § G):
- **Synthetic Users** — ships Python + TypeScript SDKs. No VS Code extension.
- **FocusPanel.ai** — SaaS web product. No VS Code extension.
- **Open-source synthetic-research tools** surveyed in sp-2cw — none ship a
  dedicated VS Code extension.

No competitor is capturing "in-editor synthetic research UX" today. If a
future competitor does, the calculus changes — but shipping first without
demand is a maintenance tax we don't need to pay.

---

## Revisit triggers

Reopen this decision if **any** of the following fire:

1. **Inbound demand.** >5 unique users request a GUI extension via issues,
   Discord, or mail within a single quarter.
2. **Top-3 shipped, gap remains.** All of sp-2cw.1 (SDK), sp-2cw.2
   (examples), sp-2cw.3 (Composio) have landed and integration adoption
   has plateaued — suggests the gap is now UX, not reach.
3. **Competitor ships one.** A peer synthetic-research tool ships a
   VS Code extension and it gains traction (Marketplace install count
   >1k).
4. **MCP-GUI gap widens.** VS Code AI Toolkit's MCP experience for YAML
   authoring / result browsing proves insufficient in observed user
   workflows.

---

## What to do instead (already filed)

The same budget of 2–4 weeks is better spent on:
- `sp-2cw.1` Python SDK — serves every Python agent framework.
- `sp-2cw.2` Framework examples — addresses the discoverability gap
  directly.
- `sp-2cw.3` Composio connector — reaches 850+ tool catalog.
- `sp-2cw.4` Skills library expansion — surfaces in Claude Code, the
  primary target editor, without shipping a second distribution.

---

## No follow-on build bead spawned

Per acceptance criteria, a build bead would be spawned only if the
decision were 'build'. Decision is 'defer', so no follow-on is filed.
If a revisit trigger fires, file a new decision bead (not a build bead)
to re-run this analysis with fresh inputs.
