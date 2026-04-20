# SynthPanel — AI Engineer Miami Readiness Assessment

**Author:** cpo (synthpanel)
**Date:** 2026-04-19
**Subject:** Pre-spotlight readiness for tonight's AI Engineer Miami demo
**Version assessed:** synthpanel v0.9.3 (live on PyPI, site https://synthpanel.dev HTTP 200)

---

## 1. Readiness rating: **4 / 5** — ship it, but tighten the cold open

Net: the product is demo-ready. The assets that matter to this audience
(MCP server, sampling fallback, public SDK, framework integration examples,
Docker image, AEO article) are all live. The one-point deduction is for
**demo-UX friction that the founder will hit on stage if unrehearsed** —
not for anything missing from the product itself.

- ✅ PyPI v0.9.3 publishes the advertised surface (`quick_poll`, `run_panel`, etc. — the 0.9.2 fix for the empty `__init__.py` is in)
- ✅ MCP sampling fallback (`sp-6at`) — zero-config demo is real
- ✅ 8 agent-framework examples live (OpenAI Agents, LangChain/LangGraph, CrewAI, LlamaIndex, MS Agent Framework, n8n, plus 2 Composio flavors)
- ✅ awesome-mcp-servers PR #4930 pending — can reference it ("we're getting listed upstream")
- ⚠️ CLI ergonomic gotcha: `--model` must precede the subcommand (`synthpanel --model sonnet prompt …`). Trivial, but it will trip the founder live if they type it the other way.
- ✅ Framework-count drift resolved (sp-dub): README table and `examples/integrations/` both list **8 framework entries** (6 MCP-native + 2 Composio bridges). Founder's script, README, and directory now agree on "eight agent frameworks." Cite that number on stage.

---

## 2. Audience: what AI Engineer Miami attendees actually care about

This is an engineer-practitioner audience, not a UXR or market-research
audience. The people at Miami are building agents, evaluating MCP servers,
fighting eval infrastructure, and picking tools they can drop into a
CrewAI/LangGraph/Agents-SDK pipeline by Monday. What resonates:

1. **"Works with my stack without a SynthPanel-specific SDK."** The fact
   that the integration is just "point your MCP client at `synthpanel
   mcp-serve`" is the whole pitch. Every framework-specific wrapper they've
   had to maintain has been a tax; MCP is the escape hatch.
2. **Zero-config via MCP sampling.** "No API key needed if your host
   supports sampling" is the line that will make people actually try it
   during the talk. This is the most differentiated moment in the deck.
3. **Structured, cost-tracked output.** Engineers have been burned by eval
   tools that return free-text and charge surprise bills. Per-turn cost
   telemetry and tool-use-forced structured output are table-stakes
   features but surprisingly rare — lead with them.
4. **Reproducibility.** Session persistence, v3 branching DAGs, and the
   ability to re-run a panel are credibility markers — they say "this was
   built by someone who actually runs evals," not "someone who cosplayed a
   research tool."
5. **What they do NOT care about:** our persona-pack library, the
   traditional-UXR comparison ("$5k focus groups"), or the qualitative-
   research framing. Those are for a PMM audience. For Miami, lead with
   the MCP / agent-tooling story and treat personas as implementation detail.

---

## 3. Gap analysis

### Installable in under 60 seconds?
Yes. `pip install synthpanel[mcp]` → set one provider key → `synthpanel
mcp-serve` is the happy path. Docker path (`docker run ghcr.io/dataviking-
tech/synthpanel …`) is live as a fallback.

### Is the MCP story clear?
Mostly yes, but **the sampling fallback is the sharpest message and it's
buried**. `sp-6at` makes `run_prompt` and `run_quick_poll` work through
the host's sampling capability when no provider key is configured. That's
the difference between "pip install and paste a key" and "it just works
inside your agent." That should be the first 20 seconds of the demo. If
the current README/landing lead is still about personas-and-instruments,
flip the fold.

### Gaps I'd flag

- **No API key in at least one development environment I tried** — I could
  not dogfood the survey below without one. If a founder's own CPO agent
  can't run `synthpanel prompt` without extra setup, the conference
  hands-on booth should have at minimum a pre-seeded `.env` sample or a
  one-liner that explicitly demonstrates the **no-key sampling path** via
  Claude Code as host. The sampling story is the answer to this friction;
  make sure that's the path attendees see first.
- **CLI flag ordering (`--model` before the subcommand)** is unusual
  argparse behavior and will bite on stage. Either switch to a subcommand-
  local `--model` (ideal), or bake the model into the demo alias so the
  founder types `synthpanel demo` instead of an argparse puzzle. Do not
  leave this to live typing.
- **"Works with X" matrix needs a live link on the landing page** if it
  is not already there. Engineers at Miami will ask "does it work with
  LangGraph?" — clicking through to the integration file in 5 seconds
  matters more than a pretty hero image.
- **awesome-mcp-servers PR is pending, not merged.** Mentioning it is
  fine; claiming we are listed would be a credibility own-goal if someone
  checks during the Q&A. Phrase it as "upstream PR open."
- **No visible benchmark / SynthBench callout** from the landing page.
  Miami is a benchmark-literate crowd. If SynthBench has any comparison
  data, a single graph ("pricing segmentation study, 5 providers, $/100Q")
  would punch well above its weight. If the data isn't ready, skip it
  rather than hand-wave.

---

## 4. Synthetic survey — dogfood result

**Transparency note:** I attempted to run the requested `synthpanel
prompt` with a simulated AI-engineer persona. My crew environment has no
`ANTHROPIC_API_KEY` (or any other provider key) configured, so the live
CLI call errored with `Missing API key: set ANTHROPIC_API_KEY`. **This is
itself the single most actionable finding in this report** — see §3. The
sampling-fallback path would have rescued this exact scenario if I had
been running inside an MCP host, which is precisely the demo story.

Rather than fabricate a `run_panel` transcript, below is a synthesized
projection of what a three-panelist AI-engineer audience would likely say
based on (a) what the product actually ships today and (b) the audience
profile in §2. Treat this as a CPO's structured estimate, not a tool run.

> **Strong recommendation:** the founder should run this exact survey
> *live on stage* using sampling through Claude Code as the host. That
> turns the "I couldn't run it without a key" friction into the demo's
> punchline: "watch — no key, no config, the host does the sampling."

### Projected panel response (three synthetic AI engineers)

**Panelist A — Staff engineer at a mid-size AI startup, builds internal
agents with LangGraph:**
> "Honestly, the MCP angle is what gets me off the fence. I've been
> burned twice by research tools that ship a bespoke Python SDK I have
> to pin and wrap. If I can point my existing `MultiServerMCPClient` at
> `synthpanel mcp-serve` and get structured persona output with cost
> tracking, yes — I'd try it next week. My one hesitation: I want to see
> the cost-per-turn telemetry before I commit it to a workflow my PM can
> trigger."

**Panelist B — Solo founder / AI researcher evaluating eval tools:**
> "Synthetic focus groups is the kind of phrase that makes me skeptical,
> but the framing 'any LLM, structured output, MCP server' is just an
> evaluation harness for persona-conditioned prompts. That I get. What I
> want to know is whether the persona library is opinionated in a way
> that biases results — because if it is, I can't use it for the
> pre-launch concept tests I actually need. Show me reproducibility and
> I'll try it."

**Panelist C — ML engineer at a Series B, skeptical of non-production
tools:**
> "I'll try anything that installs in one line and doesn't require
> another SaaS account. The sampling fallback is the thing I haven't
> seen elsewhere — that actually is novel. Whether I keep it depends on
> whether the output is reliable enough to ship into a CI step, which I
> can't judge from a demo. Give me a GitHub repo with an end-to-end
> example and I'll run it tonight."

**Net signal from projection:** positive intent to try, conditional on
(a) the MCP/sampling story being the lead, (b) clear reproducibility +
cost evidence, and (c) a copy-paste example they can run after the talk.
All three are gaps that can be closed tonight.

---

## 5. Recommended 3-minute demo script

Everything below assumes a laptop with Claude Code installed, no
provider key set, and synthpanel v0.9.3 already `pip install`ed. Total
walk is ~180 seconds with 15s of buffer.

### 0:00 – 0:20 — Cold open, land the hook
> "I'm going to run a synthetic focus group on this audience in the next
> three minutes. No API key on this laptop. Watch."

Open a terminal with Claude Code running as the MCP host. Have
`synthpanel mcp-serve` registered. Say out loud: "Claude, use synthpanel
to run a quick poll of three AI engineers: 'would you try an MCP server
for synthetic focus groups?'"

The sampling fallback answers through the host. That's your whole
value-prop demonstrated in 20 seconds.

### 0:20 – 1:10 — Show the structured output
Flip to the returned panel result. Point out, specifically:
- Three distinct persona voices (don't read them — scroll, let the
  audience see they're not templated)
- The per-panelist cost field
- The `structured` response schema hitting

Sentence to land: *"Every response is schema-validated via tool-use
forcing. You get JSON, not vibes."*

### 1:10 – 2:00 — The "works with your stack" moment
Open `examples/integrations/openai_agents.py` in a second pane (or the
web page if it renders better). Scroll — do not read. Say:

> "Eight agent frameworks, one MCP server. No SynthPanel-specific SDK.
> If your framework speaks MCP, you already have a SynthPanel adapter."

Highlight the shortest example (OpenAI Agents SDK is ~20 lines). Call
out that the same server hosts `run_panel`, `extend_panel`, persona
packs, and v3 branching DAGs.

### 2:00 – 2:40 — v3 branching as the "but can it do real research?" beat
Run:
```
synthpanel instruments graph pricing-discovery --format mermaid
```
Show the Mermaid DAG. One sentence: *"This is a branching instrument —
the router picks the next round based on what the panel said. It's
session-persisted; re-runs are reproducible."*

This closes the "is this a toy?" question without labouring the point.

### 2:40 – 3:00 — Close
Three beats, one each:
1. `pip install synthpanel[mcp]`
2. Docker image on GHCR for zero-Python installs
3. Landing page + awesome-mcp-servers PR pending

End with: *"synthpanel.dev. Try it with the host you're already using.
If it doesn't work with your stack, I want to hear about it at the
booth."*

---

## What to fix in the next hour if possible (ordered by ROI)

1. **Alias `synthpanel demo`** to a pre-flagged prompt that uses
   sampling + the most visually-impressive pack. Avoids live argparse
   typos.
2. **Promote the sampling-fallback line** to the first paragraph of the
   README and the landing hero. It is the strongest differentiator we
   have and it's currently under-sold.
3. **Pin a terminal window** with `synthpanel mcp-serve` already running
   inside Claude Code before walking on stage. Avoid cold-start risk.
4. **Have the `examples/integrations/` directory open** in a second tab
   so any follow-up question ("does it work with LangGraph?") is a
   one-click answer.
5. **Prepare the awesome-mcp-servers PR URL** on a cue card — phrase it
   as "upstream PR open," never "we're listed." If the PR merges between
   now and 8pm, update the language.

---

## Risks to call out

- **API-key demo-fail risk:** if any part of the script falls back to a
  direct CLI call instead of MCP-sampling and the laptop doesn't have
  a key set, the demo dies. Mitigation: the aliased `synthpanel demo`
  above should force the MCP-sampling path, not a direct provider call.
- ~~**Framework-integration claim drift:**~~ **Resolved (sp-dub):** README
  table, `examples/integrations/` directory, and founder's script all
  now agree on **eight agent frameworks** (6 MCP-native + 2 Composio
  bridges). Use that number on stage.
- **Dolt / backend dependency:** if any of the demo assets (panel
  result persistence, saved sessions) touches the Dolt server and it
  hiccups mid-demo, there's no graceful fallback on stage. For tonight,
  run everything out of local files; no Dolt dependency on the demo
  laptop. (Per my /Users/openclaw/gastown-dev/CLAUDE.md awareness.)

---

## Bottom line

Rating: **4/5**. Ship tonight, lead with MCP sampling, alias the demo
command, and let the live synthetic survey be the cold open. The gap
between 4 and 5 is entirely about stage-craft, not product.

— cpo
