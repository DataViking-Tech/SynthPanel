# synthpanel — Run synthetic focus groups with any LLM

 v0.13.0-dev — public beta

# synthpanel

Run synthetic focus groups with any LLM.

Zero-config inside Claude Desktop, Claude Code, Cursor, and other MCP hosts — the host's own model powers every panelist, so you drop the config in and run a panel with **no API key set**. Or bring your own key (Claude, GPT, Gemini, Grok, local) for reproducibility, ensembles, and larger panels. Personas and instruments are plain YAML — run from your terminal, a pipeline, or an AI agent's tool call over MCP (Model Context Protocol, the open standard that lets AI tools call external functions).

$ pip install synthpanel

`[mcp]` extra built in  Add `[mcp]` for Claude Code / Cursor / Windsurf agent integration.

[GitHub repo →](https://github.com/DataViking-Tech/SynthPanel) [PyPI package](https://pypi.org/project/synthpanel/)

## Who is this for?

Three jobs synthpanel does well — pick the one closest to yours.

Startup PM

### No research budget, still need signal

Pressure-test a landing headline, pricing tier, or feature name in 5 minutes with `run_quick_poll` — no recruiting, no calendar tag, no screener. Paste the result into your spec doc and keep moving.

UX researcher

### Faster turnaround, sharper studies

Run a synthetic pre-filter before booking real participants — shortlist the probes that land, kill the questions that don't, and walk into every recruited session with a tighter discussion guide.

AI engineer

### Panels as a tool inside your agent

Embed synthpanel in an agent pipeline via MCP tool calls, or drive it from Python to validate prompts, evals, and routing decisions against simulated audiences at every build.

Who this isn't for

synthpanel is **CLI-first, local-first, BYOK-first** by design. It does **not** ship:

- a hosted web UI or dashboard

- a managed SaaS tier

- SSO, RBAC, or audit-log infrastructure

- SOC 2 (or equivalent) compliance attestation

These are **deliberate non-features**, not a roadmap gap. If you need a hosted GUI or enterprise compliance artifacts for a vendor review, synthpanel isn't your product — and that's intentional.

## See it in action

A `run_quick_poll` against three personas, with the auto-synthesis at the bottom. This is the raw shape of what you get back.

```
# Question
"Would you pay $29/month for a tool that runs synthetic focus groups?"

── Sarah Chen · 34 · Startup PM ───────────────────────────
Honestly? Yes, for a quarter. $29 is under my no-approval-needed
threshold, and if it saves me one botched launch that's already paid
back. I'd want to see at least one real-world validation case first.
verdict: yes  confidence: 0.7

── Marcus Patel · 41 · Senior UX Researcher ───────────────
Not as a replacement for recruited studies, but as a pre-filter — yes.
$29 is cheap enough that I'd expense it personally. My concern is bias:
I need to know how the personas were selected before I trust the
synthesis.
verdict: conditional  confidence: 0.6

── Priya Okafor · 29 · AI Engineer ────────────────────────
I'd pay it to skip building the scaffolding myself, but I'd want an
API, not just a CLI. If it plugs into my agent via MCP I'm in at $29,
probably $99 if the SDK is clean.
verdict: yes  confidence: 0.8

── Synthesis ──────────────────────────────────────────────
3/3 lean toward yes at $29, but each attaches a condition:
  • PM wants a validation case study before committing
  • Researcher wants transparency on persona selection / bias
  • Engineer wants SDK + MCP parity, signals $99 ceiling
Consensus: price is not the blocker; trust + integration depth are.
themes: price-fit, bias-transparency, sdk-parity, mcp-integration
```

Example output, formatted for readability. Real results are returned as structured JSON via CLI or MCP tool call.

## MCP Server

Give your AI coding assistant access to synthetic focus groups. Drop this config into your editor and start running panels from chat.

```
// Claude Code · Cursor · Windsurf · Zed
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

Claude Code Cursor Windsurf Zed Claude Desktop [Full MCP docs →](/mcp)

Requires `pip install synthpanel[mcp]`. 12 tools: run prompts, run panels, manage persona & instrument packs, and more.

## Quick start

```
# 1. install
pip install synthpanel

# 2. add a key — env var (one-shot) or stored (persistent)
export ANTHROPIC_API_KEY="sk-..."             # env var
# or: synthpanel login --provider anthropic --api-key sk-...   # persisted

# 3. one-shot prompt against the default model
synthpanel prompt "What do you think of the name Traitprint?"

# 4. run a full panel and save the result
synthpanel panel run \
  --personas examples/personas.yaml \
  --instrument examples/survey.yaml \
  --save

# 5. render a shareable Markdown report from the saved result
synthpanel report <result-id> -o report.md
```

More commands: `synthpanel pack calibrate` (calibrate a persona pack against a SynthBench baseline), `synthpanel instruments` (manage branching instrument packs), `synthpanel analyze` (statistics on saved results), `synthpanel cost` (spend summary), `synthpanel login` / `whoami` (credential management). Run `synthpanel --help` for the full list.

Every `synthpanel report` output opens with a mandatory synthetic-panel banner — *“Synthetic panel. All responses below were generated by AI personas, not human respondents. Do not cite as user-research data.”* — so the rendered Markdown can’t be mistaken for real-user research. Markdown v1 only; HTML deferred to v2. See the [sp-viz-layer spec](https://github.com/DataViking-Tech/SynthPanel/tree/main/specs/sp-viz-layer).

### [MCP server](/mcp)

→

Drop-in config for Claude Code, Cursor, Windsurf, Zed, and Claude Desktop. 12 tools.

### [PyPI](https://pypi.org/project/synthpanel/)

→

Pip-installable package. `pip install synthpanel`.

### [GitHub](https://github.com/DataViking-Tech/SynthPanel)

→

Source, issues, and roadmap. MIT-licensed.

### [SynthBench](https://synthbench.org)

→

Open benchmark for synthetic survey quality. See the leaderboard at synthbench.org.

### [Recommended models](/recommended-models)

→

SynthBench-validated model picks by use case. Use `--best-model-for` to auto-select.

Powered the [SynthBench](https://synthbench.org) public benchmark — independent, open evaluation of synthetic-respondent quality. Ensemble blending of 3 models hits SPS 0.90 (90% human parity) on the current [leaderboard](https://synthbench.org).

## Further reading

- [SynthPanel vs Synthetic Users vs FocusPanel.ai — when open-source MCP beats the SaaS →](/blog/synthpanel-vs-commercial-alternatives.html) 2026-04-15
