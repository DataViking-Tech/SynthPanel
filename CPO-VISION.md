# synth-panel — CPO Vision

**Author:** synthpanel/crew/cpo | **Date:** 2026-04-04

---

## 1. Positioning

"Cheap fast focus groups" is accurate but forgettable. It describes the mechanism, not the outcome.

**The narrative:** synth-panel is the first open-source harness for synthetic qualitative research. It turns any LLM into a research panel. No SaaS subscription, no vendor lock-in, no web UI standing between you and the signal. You define personas, ask questions, get structured responses — from your terminal, in your pipeline, with your model.

The competitive field (Synthetic Users, OpinioAI, SYMAR) is entirely SaaS: opaque pricing, proprietary personas, walled gardens. synth-panel is the `curl` of synthetic research — composable, scriptable, provider-agnostic. That's not a limitation; it's the identity.

**One-liner positioning:** "Open-source synthetic focus groups. Any LLM. Your terminal."

---

## 2. Target Audience: Beachhead

The PM listed three audiences: product teams, UX researchers, indie hackers. That's correct but unranked. We lead with **indie hackers and solo founders**.

**Why they're the beachhead:**
- **Acute pain.** They can't afford UserTesting.com or a research consultant. The alternative is guessing or asking Twitter.
- **Zero switching cost.** No existing research process to displace. synth-panel isn't competing with a workflow — it's creating one.
- **Natural distribution.** They share tools on Hacker News, Twitter/X, and in newsletters. One good Show HN post could be the entire launch.
- **CLI-native.** They live in the terminal. A `pip install` and an API key they already have is zero friction.

**Sequence:**
1. **Now:** Indie hackers, solo founders, devs validating side projects
2. **3 months:** Product managers at startups (the "I need signal before the sprint planning meeting" use case)
3. **6 months:** UX research teams who want to prototype studies before spending budget on real participants

Do not try to sell to enterprise UX teams at launch. They need compliance stories, SSO, and audit trails we don't have. They'll find us when the tool is undeniable.

---

## 3. Competitive Landscape

| Player | Model | Pricing | Lock-in |
|--------|-------|---------|---------|
| Synthetic Users | SaaS | Per-interview, enterprise | Proprietary personas, RAG on your data |
| OpinioAI | SaaS | Enterprise | AI moderator, group dynamics |
| SYMAR | SaaS | Enterprise | Interactive personas |
| **synth-panel** | **Open source CLI** | **Free (bring your own API key)** | **None** |

**Our moat is openness.** Every competitor charges for the orchestration layer on top of the same underlying LLMs. We give away the orchestration. Our users pay only for tokens — at their own negotiated API rates, with their own provider choice.

This is a deliberate strategic choice. We are not leaving money on the table; we are building adoption on a foundation that SaaS competitors structurally cannot match. They can't open-source their orchestration without destroying their business model.

**What we don't compete on (and shouldn't try):**
- Group dynamics simulation (multi-persona interaction) — interesting but premature
- RAG-enriched personas from company data — that's a platform feature, not a CLI feature
- Managed infrastructure — we're a tool, not a service

---

## 4. The REPL Question

**Decision: Remove the REPL demo from the README. Don't caveat it — delete it.**

The PM recommended caveating ("REPL input coming soon"). I disagree. A caveat next to a demo that doesn't work makes the product look apologetic. It draws attention to a gap instead of a strength.

The product's value lives in two commands:
- `synth-panel prompt "question"` — single-shot research query
- `synth-panel panel run` — structured multi-persona study

These work. They're complete. They're the product.

The REPL is a convenience feature for iterative exploration. It's nice to have but it's not why someone installs synth-panel. Ship what works, don't advertise what doesn't. When the REPL is wired up, add it back to the README and announce it as a feature update.

**For the login/logout stubs:** Remove them from the CLI entirely. Dead commands erode trust. If we add auth later, we add the commands then.

---

## 5. Version Signal

**Ship as 0.1.0.**

The PM raised this as an open question. It's not close.

- **0.1.0** says: "This works. We're iterating. The API may change." That's honest and correct.
- **1.0.0** says: "This is stable. Build on it." That's a promise we can't keep yet — we have stubs in the codebase and no plugin ecosystem in the wild.

0.1.0 is not a lack of confidence. It's a signal that we respect semver. The kind of developer who installs a CLI tool from PyPI understands this signal and respects it. Shipping 1.0.0 with known stubs would actually reduce trust with our target audience.

**Version roadmap:**
- **0.1.0** — Go-public. Core works, honest README.
- **0.2.0** — REPL wired up, plugin integration tests, first community feedback incorporated.
- **0.5.0** — Multi-round conversations, branching instruments, session management.
- **1.0.0** — When someone has built something real on top of synth-panel and we've held the API stable for 3+ minor releases.

---

## 6. Go-Public Messaging

**Portfolio site (one sentence):**

> "Run synthetic focus groups from your terminal — any LLM, any persona, five minutes."

**PyPI description (already good):** The existing `pyproject.toml` description works. Don't overthink it.

**Show HN title:**
> "synth-panel — open-source synthetic focus groups from the command line"

**Show HN body (skeleton):**
- What: CLI tool that runs AI personas through structured research instruments
- Why: Real user research is slow and expensive; this gives you a 5-minute pre-filter
- How: `pip install synth-panel && synth-panel prompt "What frustrates you about password managers?" --model haiku`
- Differentiator: Provider-agnostic, open source, MCP-integrated, no SaaS
- What it's NOT: A replacement for talking to real users

---

## 7. Six-Month Direction

synth-panel becomes **the default way developers get qualitative signal without leaving their workflow.**

Not a platform. Not a dashboard. A tool that's always one command or one MCP call away.

**Three bets:**

1. **MCP as the primary interface.** The CLI is for standalone use. But the real unlock is when synth-panel is a tool inside Claude Code, Cursor, or any MCP-capable agent. "Before you ship this feature, run it past the panel" becomes a one-line agent instruction. This is where the distribution is — embedded in existing workflows, not competing for screen time.

2. **Instrument sophistication.** v0.1 instruments are flat question lists. The direction is branching logic, conditional follow-ups, multi-round conversations where persona responses in round 1 shape questions in round 2. This is what makes synth-panel more than "I could just prompt ChatGPT with a persona" — the harness does research design, not just prompting.

3. **Community-contributed persona packs.** A startup founder persona pack. A healthcare patient persona pack. An enterprise buyer persona pack. The MCP server already has `save_persona_pack` / `list_persona_packs`. The infrastructure exists — we need the ecosystem. This is where the flywheel starts: the more persona packs exist, the more useful the tool is, the more people contribute packs.

**What we are NOT building:**
- A web UI or dashboard (CLI-first is the identity, not a phase)
- A hosted service (the value prop is "bring your own LLM")
- Real-time collaboration features (this is a single-player tool)
- Persona auto-generation from data (users define their own — that's a feature)

---

## Disagreements with the PM Plan

The PM plan is solid tactical work. Two pushbacks:

1. **"Caveat the REPL" is wrong.** Don't caveat, remove. (See section 4.) The PM optimized for speed; I'm optimizing for first impression. A new user's first impression should be "this does exactly what it says" not "this mostly works but some parts are coming soon."

2. **The execution order should front-load the README rewrite.** The PM's order starts with small fixes. I'd rewrite the README first — it's the product's face, and every other fix is invisible until the README is honest. The README should be written to match what the product does *today*, not patched line-by-line to fix lies.

Everything else in the PM plan — the priority list, the effort estimates, the defer list — I endorse. The PM correctly identified that this is documentation hygiene, not architectural work. The product is ready; the packaging isn't.

---

## Summary

synth-panel is the only open-source tool in a market full of SaaS walled gardens. That's the positioning. Lead with indie hackers who live in the terminal. Ship 0.1.0 with an honest README. Remove what doesn't work instead of apologizing for it. Bet on MCP integration as the primary growth vector. Build the persona pack ecosystem.

The product is good. Make the packaging match.
