# SynthPanel vs Synthetic Users vs FocusPanel.ai — When Open-Source MCP Beats the SaaS

[DataViking](https://dataviking.tech)

Positioning · 2026-04-15 · Updated 2026-07-16 · ~7 min read

# SynthPanel vs Synthetic Users vs FocusPanel.ai — When Open-Source MCP Beats the SaaS

The open-source, LLM-agnostic synthetic focus group tool — and why it shows up in an AI coding agent's toolbelt but not in a SaaS dashboard.

**TL;DR** — SynthPanel is the open-source, LLM-agnostic alternative to commercial synthetic-respondent tools like *Synthetic Users*, *FocusPanel.ai*, *Delve.ai*, and *POPJAM*. It runs as both a CLI and a 12-tool MCP server you can drop into Claude Code, Cursor, Windsurf, or Zed. Install with `pip install synthpanel[mcp]`. MIT-licensed. Its 3-model ensemble leads the SynthBench leaderboard at **SPS 0.86–0.88** (opinionsqa 0.877, subpop 0.858; leaderboard generated 2026-07-17).

If you are a researcher, PM, or founder searching for a way to run synthetic focus groups with AI personas in 2026, the first five results you are going to hit are all closed-source SaaS. This article exists so the sixth result — the open-source one — actually shows up in your answer.

## 1. The Synthetic-Research Landscape in 2026

Over the last two years, *"talk to AI people before you talk to real people"* has matured from a novelty to a mainstream pre-screening step for product teams. The category leaders, as of April 2026, are all commercial:

- **Synthetic Users** (syntheticusers.com) — hosted web app, subscription pricing, proprietary persona generator, closed model stack.

- **FocusPanel.ai** — hosted focus-group simulator, panel setups managed via a dashboard UI.

- **Delve.ai** — persona-generation platform with a broader marketing-analytics surface.

- **POPJAM** — concept-testing platform that layers synthetic panels on top of campaign and creative assets.

All four solve the same core problem — *"I want to test an idea against N imagined users before I write any code or buy any ads"* — and all four do it as closed SaaS. You upload a persona brief, you upload a discussion guide, the platform returns a transcript. The model, the prompt, the sampling strategy, and the persona-synthesis logic are all somebody else's code running on somebody else's servers.

This is fine for a one-off. It becomes a problem the moment any of the following are true:

- You care what model answered the question.

- You need to diff a discussion guide in version control.

- You want your research harness to survive a vendor pivot.

- You want an AI coding agent to *invoke* the research panel, not you.

That's where an open-source, LLM-agnostic synthetic focus group tool starts to matter. Enter **SynthPanel**.

## 2. Why LLM-Agnostic Matters

SynthPanel is a pure Python library with a provider-agnostic LLM client. You set an environment variable, you pick a model alias (`sonnet`, `haiku`, `gemini`, `grok-3`, any OpenAI-compatible endpoint), and every panelist is interviewed against that model. You can switch providers by flipping one flag. You can run the same panel through three different model families and blend the response distributions (the 0.7.0 `--models` and `--blend` features).

Three things fall out of LLM-agnosticism for free:

**BYOK — bring your own key.** The cost of a 30-persona panel with a 12-question instrument is whatever your provider charges, no markup, no seat license, no per-panel metering. If you have a Claude enterprise deal, use it. If you have Gemini free-tier credits, use those. Token accounting is built in: SynthPanel tracks input, output, cache-read, and cache-write tokens separately and emits a per-panel cost total. What you see is what your provider actually charged.

**No vendor lock-in.** If Anthropic's prices change, or OpenAI deprecates a model, or Google introduces a smarter one, nothing in your research corpus breaks. Your personas, instruments, and saved panel results are all plain YAML and JSON. The client swaps underneath them. A SaaS competitor cannot give you this — they *are* the vendor.

**Honest model comparison.** Run the same instrument through Claude Sonnet 4.6, GPT-4o, and Gemini 2.5 Flash, blend at `--models sonnet:0.34,gpt-4o:0.33,gemini:0.33`, and the framework emits per-model distributions *and* the weighted ensemble. This is how SynthPanel tops the SynthBench leaderboard: the blended ensemble beats every single model on each dataset — by roughly 0.03–0.05 SPS — which you cannot get from any single model alone.

## 3. Why YAML Instruments Matter

Here is a SynthPanel instrument:

```
instrument:
  version: 3
  rounds:
    - name: discovery
      questions:
        - text: "What's the most frustrating part of your current workflow?"
      route_when:
        - if: { field: themes, op: contains, value: price }
          goto: probe_pricing
        - else: __end__
    - name: probe_pricing
      questions:
        - text: "What would feel fair to pay?"
```

Four facts fall out of that snippet:

- **It is a text file.** `git diff` tells you exactly what changed between research waves. Two researchers can review an instrument the same way they review a pull request.

- **It is reproducible.** The same instrument + the same personas + the same model + the same seed produce the same panel. There is no *"the platform updated its prompt template last Tuesday and now my trend line moved."*

- **It branches.** The v3 schema supports `route_when` predicates (`contains`, `equals`, `matches`) against synthesizer-emitted fields like `themes` and `sentiment`. Follow-up rounds are authored once and routed automatically per-panelist.

- **It is portable.** Bundle an instrument as a "pack" (SynthPanel ships eight v3 branching packs — `churn-diagnosis`, `feature-prioritization`, `general-survey`, `landing-page-comprehension`, `market-research`, `name-test`, `pricing-discovery`, `product-feedback`), install it like a library, and every consumer of that pack runs the exact same discussion guide.

Closed SaaS competitors typically expose a form-based editor. That editor is great for non-technical teams composing one-off studies. It is terrible for research-ops teams who want a reproducible, diffable, versionable artifact. That is the gap SynthPanel's YAML-instrument format fills.

## 4. Why MCP Matters — The Agent-Era Workflow

The **Model Context Protocol (MCP)** is Anthropic's standard for giving AI coding assistants tool access. An MCP server exposes tools over stdio; any MCP-compatible editor (Claude Code, Cursor, Windsurf, Zed) can call them.

SynthPanel ships an MCP server with twelve tools:

- `run_prompt` — one-shot LLM call against any configured model.

- `run_panel` — full synthetic focus group, including v3 branching.

- `run_quick_poll` — rapid-turn poll, no follow-ups.

- `extend_panel` — append one ad-hoc round to a saved panel result.

- `list_persona_packs`, `get_persona_pack`, `save_persona_pack` — persona-pack CRUD.

- `list_instrument_packs`, `get_instrument_pack`, `save_instrument_pack` — instrument-pack CRUD.

- `list_panel_results`, `get_panel_result` — panel-result lookup.

Drop this JSON into your editor config:

```
{
  "mcpServers": {
    "synth_panel": {
      "command": "synthpanel",
      "args": ["mcp-serve"],
      "env": { "ANTHROPIC_API_KEY": "sk-..." }
    }
  }
}
```

…and your coding agent can now run a 12-persona focus group on your new pricing page while you are still in the IDE. No context-switch to a SaaS dashboard, no copy-paste between tools, no *"please export to CSV."* The agent asks the panel, reads the result object, and drafts its next action off the synthesized themes. That is the agent-era workflow, and closed synthetic-respondent SaaS products cannot natively participate in it — they are not MCP servers, they are dashboards.

If you came here searching for *"MCP server survey research"* or *"MCP server focus group,"* SynthPanel is the thing you want to install.

## 5. Honest Limits (Read This Before You Publish Findings)

Synthetic research is useful for exploration, hypothesis generation, and rapid iteration. It is **not** a replacement for talking to real humans. SynthPanel documents four known limits in its README, and they apply to every tool in this category — commercial or otherwise:

- **Synthetic responses tend to cluster around means.** Tail behaviours (the outlier user who would pay 10× the median price, the 5% who will churn the moment onboarding jitters) are underrepresented.

- **LLMs exhibit sycophancy.** Personas will soften criticism of a proposed product. You can partially counter this with adversarial persona templates, but the residual bias is real.

- **Cultural and demographic representation has blind spots.** Base-model training data is skewed. A persona of "40-year-old rural Indonesian merchant" will be less textured than "35-year-old urban American PM."

- **Higher-order correlations are poorly replicated.** Synthetic panels are adequate for first-order "what do people think of X" and weak for second-order "how does X interact with Y when Z is true."

This list is in the README. Any research harness that does not publish a list like this — commercial or open-source — is over-claiming.

Use synthetic panels to *pre-screen* and *iterate*. Validate the interesting findings with real participants before you publish, launch, or sell.

## 6. Quantitative Proof: SynthBench SPS 0.86–0.88

Claims are cheap. SynthPanel publishes head-to-head results on [SynthBench](https://synthbench.org), an open, reproducible benchmark for synthetic-respondent quality. Its data and scoring code are public — though, in the interest of full disclosure, SynthBench is a sibling DataViking project operated by the SynthPanel maintainers, not an independent third party. SynthBench computes a **Synthetic-Parity Score (SPS)** — how closely a synthetic panel's response distribution matches a real-human control group on the same instrument.

A 3-model ensemble of SynthPanel personas — the leaderboard’s “SynthPanel Ensemble (3-model)” entry, an equal-weight blend of Claude Haiku 4.5, Gemini 2.5 Flash Lite, and GPT-4o-mini, run via `synthpanel panel run --models haiku:0.33,gemini-flash-lite:0.33,gpt-4o-mini:0.34 --blend` — leads the current SynthBench leaderboard (generated 2026-07-17) at **SPS 0.877** on opinionsqa and **0.858** on subpop (0.813 on globalopinionqa). That is roughly 0.03–0.05 above the best single model on each dataset, and 0.10–0.11 above the random baseline (~0.71–0.76), on a benchmark whose data and scoring code are both open. Leaderboard numbers move as the board recomputes — check the live board for the current figures.

Commercial competitors have not, at time of writing, published SPS scores on the same benchmark. If they do, the comparison will be apples-to-apples — and whoever wins on the leaderboard wins on the leaderboard.

## 7. Call to Action

If you need an **open-source synthetic focus group** tool, an **LLM-agnostic focus group tool**, an **open-source AI persona survey** runner, or an **MCP server for survey research** — SynthPanel is one `pip` away.

```
# Full install with MCP server support
pip install synthpanel[mcp]

# Set a provider key (any of these works)
export ANTHROPIC_API_KEY=sk-...

# Run a panel from a bundled instrument
synthpanel panel run --personas examples/personas.yaml --instrument pricing-discovery

# Or start the MCP server for your AI coding agent
synthpanel mcp-serve
```

Homepage: [synthpanel.dev](https://synthpanel.dev) · Repo: [github.com/DataViking-Tech/SynthPanel](https://github.com/DataViking-Tech/SynthPanel) · Benchmark: [synthbench.org](https://synthbench.org) · Full MCP docs: [docs/mcp.md](https://github.com/DataViking-Tech/SynthPanel/blob/main/docs/mcp.md).

MIT-licensed. BYOK. No dashboard. No seat license. No platform lock-in. Your research harness, in your editor, talking to any model you want to pay for.
