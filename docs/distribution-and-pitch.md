# SynthPanel Distribution Surfaces + DataViking Elevator Pitch

**Author:** cpo (synthpanel)
**Date:** 2026-04-19
**Subject:** (1) AI-era distribution channels beyond MCP, ranked by reach-to-effort. (2) Three candidate DataViking elevator pitches, with a recommendation.

---

## Part 1 — Additional AI-focused distribution surfaces

### What's already shipped (baseline)

Through the MCP server alone, SynthPanel is reachable from: Claude
Code, Claude Desktop, Cursor, Windsurf, Zed, VS Code (1.102+), OpenAI
Agents SDK, LangChain / LangGraph, LlamaIndex, CrewAI, Microsoft Agent
Framework, n8n, and Composio. That is roughly 13 surfaces for the cost
of one server binary. The additional work below is **listing work and
asset work**, not integration work.

### TIER 1 — Table-stakes (ship this week; high reach, low effort)

These are the channels where *not being present reads as "doesn't
exist"* to an engineer searching for tools. Effort here is almost
entirely submission + listing copy.

| Channel | Effort | Reach | Why it matters |
|---|---|---|---|
| **awesome-mcp-servers merge** | follow-up on PR #4930 (0.5d) | Everyone who searches "MCP servers" on GitHub | Table-stakes. Mentioning it pre-merge as "upstream PR open" is fine; merged is a credibility mark. |
| **mcp.so listing** | 0.5d | Primary MCP directory | First place MCP-curious devs browse. |
| **Smithery.ai listing** | 0.5d | Growing MCP registry | Second-most-trafficked MCP directory. |
| **PulseMCP / Glama / MCP Market** | 0.5d combined | Long tail of MCP directories | Submit once, collect backlinks. |
| **Anthropic MCP official servers list** | 0.5d PR | Anthropic docs traffic | Highest authority of any MCP directory. |
| **Cursor Extensions Directory** | 0.5d | Cursor user base (growing fast) | Cursor's MCP registry is the native install path. |
| **VS Code MCP registry / marketplace** | 0.5d | VS Code MCP-native users | 1.102+ has auto-discovery; a marketplace listing with one-click config JSON still earns installs. |
| **Windsurf plugin store** | 0.5d | Windsurf users | Small but native-MCP audience. |
| **Zed extensions** | 0.5d | Zed's engineer-heavy userbase | Miami crowd is over-indexed on Zed. |
| **Claude Code /plugin registry** | verify already-live `/plugin install synthpanel` (0.25d) | Claude Code users | Already shipped per CHANGELOG; confirm discoverability. |

**Total TIER 1 effort:** ~4 developer-days across one focused sprint.
This is the single most important block of work after tonight's demo.

### TIER 2 — Differentiators (ship within 30 days; compounds credibility)

These are higher-effort but carry disproportionate signal for the AI
Engineer Miami / agent-builder audience.

| Channel | Effort | Reach | Why it matters |
|---|---|---|---|
| **Anthropic Cookbook PR** | 1-2d | Anthropic-developer inbound | A cookbook notebook showing "MCP + sampling + SynthPanel end-to-end" is the single highest-trust referral we can earn. Miami attendees grep the Cookbook. |
| **OpenAI Cookbook PR** | 1-2d | OpenAI-developer inbound | Same argument, different tribe. Use the Agents SDK example we already have. |
| **Hugging Face Space** | 1-2d | ML/AI researcher crowd | A public Space running `synthpanel mcp-serve` behind a Gradio UI turns the README into a live try-it-now link. HF has outsized reach into the research community. |
| **ChatGPT custom GPT ("SynthPanel Research Assistant")** | 2-3d | Non-dev PMM/founder audience who live in ChatGPT | This is how we reach buyers who will never `pip install` anything. Wraps `run_quick_poll` as GPT Actions against a hosted endpoint. |
| **GPT Store listing** | 0.5d after the GPT exists | GPT Store browse | Free distribution once the GPT is built. |
| **n8n Community Node (official)** | 2-3d | n8n operator audience | Upgrade from the workflow-JSON example we ship to an official n8n node with first-class UX. n8n's marketplace is where no-code ops people live. |
| **Zapier MCP connector** | 2-3d | Zapier's entire long tail | Zapier added MCP support in 2026; listing earns a meaningful SMB channel. |
| **LangChain Hub template** | 1d | LangChain users | Submit a LangGraph template using the synthpanel MCP adapter. Lands in LangChain docs surface area. |
| **CrewAI community tools repo PR** | 1d | CrewAI users | Same pattern. |
| **Raycast extension** | 2-3d | Mac power-user audience | "Run a synthetic focus group from Raycast" is novel enough to get organic Twitter pickup. Low priority but high delight. |

**Total TIER 2 effort:** ~15-20 developer-days. Spread over four weeks,
this is one engineer's quarter.

### TIER 3 — Watch / skip

- **Replicate** — SynthPanel is not a model; poor fit. **Skip.**
- **Slack/Discord bots** — customer-specific. Only build if a specific
  design-partner requests it. **Skip for now.**
- **Notion AI connector** — immature ecosystem, low reach. **Skip.**
- **Google Vertex AI tools / Gemini extensions** — emerging but
  pre-product-market-fit as a distribution surface. **Watch.**
- **JetBrains Junie / AI Assistant MCP** — early. Listing once the
  registry matures costs nothing. **Watch.**

### What's table-stakes vs differentiator for AI Engineer Miami

**Table-stakes for Miami (must be present):**
- awesome-mcp-servers merge
- mcp.so and Smithery.ai listings
- VS Code + Cursor + Zed + Windsurf listings
- `/plugin install synthpanel` verified discoverable

If any of these are missing when a Miami attendee searches post-talk,
they will assume we are earlier-stage than we are. Ship this block
inside seven days.

**Differentiators that will be *remembered* from Miami:**
- Anthropic Cookbook PR (because "we're in the Cookbook" is an
  unfakeable signal)
- Hugging Face Space (because "try it now in the browser" converts a
  conference conversation into a login)
- ChatGPT GPT (because it crosses the dev/non-dev line and opens the
  PM/founder buying motion that pairs with the dev-champion motion)

Of those three, **the Anthropic Cookbook PR is the single highest
ROI post-Miami investment.** It turns every future "how do I use MCP?"
search into a synthpanel impression.

### ROI summary

| Priority | Channel | Effort (dev-days) | Recommended window |
|---|---|---|---|
| P0 | awesome-mcp-servers, mcp.so, Smithery, HQ MCP directories | 2.5 | this week |
| P0 | Cursor / VS Code / Zed / Windsurf registries | 2 | this week |
| P1 | Anthropic Cookbook PR | 2 | within 14 days |
| P1 | Hugging Face Space | 2 | within 14 days |
| P1 | ChatGPT Custom GPT + Store listing | 3 | within 21 days |
| P2 | OpenAI Cookbook PR | 2 | within 30 days |
| P2 | n8n Community Node | 3 | within 30 days |
| P2 | Zapier MCP connector | 3 | within 30 days |
| P3 | LangChain / CrewAI community repos | 2 | opportunistic |
| P3 | Raycast extension | 3 | opportunistic |

**Total P0+P1 budget to unlock:** ~11-14 developer-days. One engineer,
three weeks. This is the biggest distribution lever available without
writing a line of product code.

---

## Part 2 — DataViking Technologies elevator pitch

### Context from Wesley's profile

From https://traitprint.com/wesley-johnson:
- Ex-Peloton data leader; scaled a data org from 0→6+ engineers and
  stakeholder coverage from 5→50+
- Founder of DataViking Technologies (Jan 2026)
- Stated interests: fintech, gaming (Steam), SaaS (SwipeMatch beta),
  modern data infra
- Philosophy: psychological safety, "delegation as strategic leverage,"
  data-driven threshold-setting, distributed ownership
- DataViking self-description: *"AI-assisted products at the
  intersection of data, analytics, and LLM technologies"*

### The three products, as an honest narrative thread

- **Traitprint** — represents real people. A career-identity platform
  that turns a professional into a structured, durable profile ("trait
  print").
- **SynthPanel** — represents synthetic people. Personas-as-research-
  instruments for teams that need to learn fast and cheap.
- **SynthBench** — measures whether synthetic people are any good.
  A benchmark comparing synthetic responses to real human data across
  providers and model configurations.

The unifying observation: **DataViking is building the stack for
understanding people — real or synthetic — at AI speed, with
measurable fidelity.** Three products, one substrate: structured
representations of human signal.

### Candidate pitch 1 — "AI-native people infrastructure"

> DataViking Technologies builds AI-native infrastructure for
> understanding people. Traitprint captures who professionals
> actually are — durable, structured, portable. SynthPanel runs
> synthetic focus groups in your terminal or your agent's tool call.
> SynthBench measures whether those synthetic responses match real
> humans. Three products, one stack: we turn messy human signal into
> reliable inputs for the teams building the next generation of AI
> products.

**Angle:** technical-infrastructure positioning. Lands hardest with
engineers and platform buyers.

**Strength:** clean architecture story. Easy to re-tell.

**Weakness:** "understanding people" is slightly abstract on first
listen; listener may need the second sentence before it clicks.

### Candidate pitch 2 — "The research stack for small teams moving fast"

> Real user research is expensive and slow. DataViking builds the
> research stack for teams that can't afford either. Traitprint turns
> a career into a queryable profile in minutes. SynthPanel runs a
> synthetic focus group in seconds, from your terminal or any MCP
> host. SynthBench tells you how close the synthetic answers are to
> real humans. One founder, three tools — we ship the infrastructure
> a five-person company needs to move like a fifty-person one.

**Angle:** founder-empathy positioning. Leans into Wesley's Peloton
"0→6, 5→50" scaling story as implicit credibility.

**Strength:** names the pain. Lets the listener self-identify as the
buyer. "Move like a fifty-person company" is sticky.

**Weakness:** frames Traitprint as research-stack-adjacent, which is
a stretch — Traitprint is identity, not research. Honesty risk.

### Candidate pitch 3 — "Measurable AI research, not vibes"

> The AI research industry runs on vibes. We're building the version
> that runs on measurement. SynthPanel lets any team run synthetic
> focus groups through their existing AI agent. SynthBench publishes
> an open leaderboard of how well synthetic responses match real
> human data — so you can pick the model that actually works for
> your question. Traitprint applies the same rigor to how people
> represent themselves. We're DataViking. Measurable research, at AI
> speed.

**Angle:** category-creator positioning. Stakes out "measurable"
versus competitors' implicit vibes-based offerings.

**Strength:** sharpest and most differentiated. The "vibes vs
measurement" frame is quotable and Miami-aligned. SynthBench becomes
the proof point, not an afterthought.

**Weakness:** Traitprint is forced into the frame awkwardly — it's
not obviously "measurable research." Requires a crisper Traitprint
story before this lands cleanly.

### Recommendation: **Pitch 1, lightly hybridized with Pitch 3's teeth**

Why pitch 1 wins:

1. It is the only one of the three where all three products sit
   comfortably under the umbrella without stretching.
2. "AI-native infrastructure for understanding people" is specific
   enough to be memorable and broad enough to license future products
   without rewriting the thesis.
3. It matches Wesley's own self-description ("AI-assisted products at
   the intersection of data, analytics, and LLM technologies") but in
   a sharper, more product-oriented form.
4. Pitch 3's "measurable, not vibes" line is too good to abandon. Bolt
   it in as the credibility beat after the three-product round.

### Final recommended 30-second pitch (hybrid, ~65 words / ~25 seconds spoken)

> DataViking Technologies builds AI-native infrastructure for
> understanding people. **Traitprint** captures who professionals
> really are — structured, durable, portable. **SynthPanel** runs
> synthetic focus groups from your terminal or any AI agent's tool
> call. **SynthBench** measures how close those synthetic responses
> are to real humans — open leaderboard, any provider. One stack:
> measurable research, at AI speed.

### Delivery notes for Wesley

- Land on "understanding people" in the first sentence, then slow
  down for the three product names — each gets one verb.
- The closing "measurable research, at AI speed" is the take-home
  line. Rehearse it as its own beat, not as a trailing phrase.
- If an engineer asks "what does that actually mean?", the one-sentence
  follow-up is: *"If you're building an agent and you want to know how
  real humans would answer a question — we give you a cheap, fast,
  measurable way to simulate that."*
- If a business/investor asks, the one-sentence follow-up is: *"We
  turn the most expensive part of product development — human
  research — into infrastructure."*

— cpo
