# synthpanel — Run synthetic focus groups with any LLM

 v1.5.7 — public beta

# synthpanel

Run synthetic focus groups with any LLM.

Zero-config inside Claude Desktop, Claude Code, Cursor, and other MCP hosts — the host's own model powers every panelist, so you drop the config in and run a panel with **no API key set**. Or bring your own key (Claude, GPT, Gemini, Grok, local) for reproducibility, ensembles, and larger panels. Personas and instruments are plain YAML — run from your terminal, a pipeline, or an AI agent's tool call over MCP (Model Context Protocol, the open standard that lets AI tools call external functions).

$ pip install synthpanel

Add the `[mcp]` extra — `pip install "synthpanel[mcp]"` — for Claude Code / Cursor / Windsurf agent integration.

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

More commands: `synthpanel pack calibrate` (calibrate a persona pack against a SynthBench baseline), `synthpanel instruments` (manage branching instrument packs), `synthpanel analyze` (statistics on saved results), `synthpanel results` (list and inspect saved runs by ID), `synthpanel cost` (spend summary), `synthpanel login` / `whoami` (credential management). Run `synthpanel --help` for the full list.

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

## Measured against real humans

The [SynthBench](https://synthbench.org) benchmark’s ground truth is real survey respondents — the General Social Survey (NORC), the Pew American Trends Panel (via the OpinionsQA, SubPOP, and GlobalOpinionQA datasets), and the World Values Survey — and every number below is recomputable from public data and open code. GSS is public-use, so for it the real human answer shares are shown directly.

Real General Social Survey question (2024 wave, item `GSS_LIFE`) — GSS is public-use data, so the real human answer shares appear side by side with the synthetic ones:

“In general, do you find life exciting, pretty routine, or dull?”

Exciting humans 36.8% · synthetic 26.7%

Pretty routine humans 56.7% · synthetic 73.3%

Dull humans 5.7% · synthetic 0.0%

Don’t know humans 0.7% · synthetic 0.0%

Green = real humans: GSS 2024, NORC — public-use data, weighted shares (`wtssnrps`). Blue = synthetic: gemini-2.5-flash via the SynthBench harness, run 2026-07-15, 30 sampled responses.

Measured divergence between the two distributions: Jensen–Shannon divergence 0.046 (0 = identical distributions, 1 = disjoint).

This item is one of that run’s closest matches; across all 75 GSS questions the same run’s mean JSD is 0.301. Because NORC releases GSS into the public domain, the full per-question payload — real distribution included — is served openly at [synthbench.org/data/question/gss/GSS_LIFE.json](https://synthbench.org/data/question/gss/GSS_LIFE.json) (no sign-in), alongside 59 more GSS items, and is recomputable from the [NORC GSS release](https://gss.norc.org) with the open [SynthBench harness](https://github.com/DataViking-Tech/SynthBench) (`leaderboard-results/gss_openrouter_google_gemini-2.5-flash_20260715_175807.json`).

Real Pew American Trends Panel question (Wave 96, via SubPOP item `BELIEVE_a_W96`):

“Do you believe in Heaven?”

Pew’s response data is license-gated (CC-BY-NC-SA), so this page shows only the *gap* between the SynthPanel 3-model ensemble (90 sampled responses) and the real Pew respondents — not the survey values themselves:

Yes, I believe in this synthetic within ±13.8 pts

No, I do not believe in this synthetic within ±15.1 pts

Refused synthetic within ±1.3 pts

Bars show |synthetic − human| per answer option, in percentage points (shorter = closer).

Measured divergence from the real Pew respondents’ distribution: Jensen–Shannon divergence 0.023 (0 = identical distributions, 1 = disjoint).

This item is one of the ensemble’s closest matches by JSD; across the full 200-question SubPOP evaluation the same run’s mean JSD is 0.209. View the underlying distributions signed-in at synthbench.org, or recompute these exact numbers from the public [SubPOP dataset](https://huggingface.co/datasets/jjssuh/subpop) and the open [SynthBench harness](https://github.com/DataViking-Tech/SynthBench); the synthetic distribution and per-question JSD are published in the repo (`leaderboard-results/subpop_ensemble_3blend_20260714_171917.json`, run dated 2026-07-14).

Powers the [SynthBench](https://synthbench.org) open benchmark — an open, reproducible evaluation of synthetic-respondent quality (public data + scoring code), operated by the SynthPanel maintainers rather than an independent third party. On the current [leaderboard](https://synthbench.org/data/leaderboard.json) (generated 2026-07-15), the 3-model ensemble scores SPS 0.877 on opinionsqa, 0.831 on subpop, and 0.813 on globalopinionqa — roughly 0.01–0.05 above the best single model on each dataset and 0.07–0.11 above the random baseline (~0.71–0.76). Leaderboard numbers move as the board recomputes.

## Further reading

- [SynthPanel vs Synthetic Users vs FocusPanel.ai — when open-source MCP beats the SaaS →](/blog/synthpanel-vs-commercial-alternatives.html) 2026-04-15
