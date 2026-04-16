# SynthPanel Agent Integration Landscape: Beyond MCP

**Author:** crew/advo (Developer Advocate specialization)
**Date:** 2026-04-15
**Parent bead:** sp-2cw
**Companion:** [docs/agent-discovery-audit.md](agent-discovery-audit.md) (sp-ege)
**Status:** Research complete, recommendations filed as child beads

---

## Executive Summary

SynthPanel's MCP server is better-positioned than we thought. The major agent
frameworks — OpenAI Agents SDK, LlamaIndex, CrewAI, Microsoft Agent Framework
1.0, n8n, Zapier, and VS Code AI Toolkit — **all now support MCP as a
first-class tool integration path**. This means SynthPanel's 12-tool MCP
server already works with every major framework. There is no urgent need to
ship native framework-specific tool wrappers.

The gap is not capability — it's **discoverability and ergonomics**. Nobody
knows SynthPanel works with LangChain/CrewAI/OpenAI Agents because there are
no examples, no docs, no proof. And for programmatic use (scripts, pipelines,
notebooks), spawning an MCP server process is heavier than a direct Python
import.

### Recommendation: Next 3 After MCP

1. **Public Python SDK convenience layer** — `from synth_panel import
   quick_poll, run_panel`. The highest-impact, lowest-effort addition.
   Serves both human developers and any Python-based agent framework
   directly. No MCP overhead.

2. **Agent framework examples + "Works with X" documentation** — Five
   example scripts proving SynthPanel works with LangChain, CrewAI,
   OpenAI Agents SDK, LlamaIndex, and Microsoft Agent Framework via MCP.
   Zero library code — just examples that prove the MCP bridge works.
   Each example becomes a searchable, AEO-indexable content page.

3. **Composio connector** — One listing in the Composio marketplace
   (850+ tools) gives SynthPanel simultaneous presence in LangChain,
   CrewAI, Semantic Kernel, and AutoGen tool catalogs. This is the
   highest-leverage marketplace play.

---

## A. Current Integration Surfaces (Inventory)

SynthPanel ships six integration surfaces today:

| Surface | State | Agent-Reachable? |
|---|---|---|
| **CLI** (`synthpanel` command) | Full-featured, all commands | Yes — any agent can shell out |
| **Python library** (internal functions) | `orchestrator.run_panel_parallel()`, `run_multi_round_panel()`, `ensemble_run()` exist but are internal implementation functions | Partial — importable, but no documented public API; no `__all__` exports at package level; no convenience wrappers |
| **MCP server** (12 tools, 4 resources, 3 prompts) | FastMCP, stdio transport, defaults to haiku | Yes — primary agent interface |
| **Claude Code plugin** (`.claude-plugin/plugin.json`) | Wraps MCP server + ships `/focus-group` skill | Yes — Claude Code only |
| **Claude Code skill** (`skills/focus-group/SKILL.md`) | Structured focus-group workflow | Yes — Claude Code only |
| **Devcontainer** (`.devcontainer/devcontainer.json`) | GitHub Codespaces / dev environments | Partial — dev-only, not agent-facing |

**Key observations:**
- The MCP server is the only general-purpose agent interface.
- The Python library has the functions but no public API surface: no
  `from synth_panel import quick_poll`, no documented entry points,
  no `__init__.py` exports at the package level.
- The Claude Code plugin and skill are well-built but serve one editor only.
- No framework-specific wrappers exist (LangChain, CrewAI, etc.).
- No Docker image exists for ephemeral agent invocation.
- No marketplace/registry presence beyond MCP (Composio, n8n, Zapier).

---

## B. The MCP Convergence Thesis

The most important finding in this research: **MCP has won the agent tool
integration protocol war**. Every major agent framework now supports MCP as a
first-class tool source. This changes the calculus for SynthPanel's
integration strategy.

### Framework-by-framework MCP support (verified April 2026)

| Framework | MCP Support | How It Works | Native Wrapper Needed? |
|---|---|---|---|
| **OpenAI Agents SDK** | Built-in | `MCPServerStdio("synthpanel", ["mcp-serve"])` → agent auto-discovers 12 tools | **No** |
| **LlamaIndex** | `llama-index-tools-mcp` (v0.4.8, Feb 2026) | `BasicMCPClient` → `mcp_tool_spec.to_tool_list()` → agent uses tools | **No** |
| **CrewAI** | `crewai-tools[mcp]` | MCP server as tool source | **No** |
| **Microsoft Agent Framework 1.0** | Built-in MCP clients (April 2026) | MCP client in agent config → auto-discover tools | **No** |
| **LangChain** | `langchain-mcp-adapters` | MCP → LangChain tool bridge | **No** |
| **n8n** | Built-in MCP server support in AI Agent node | Point at MCP server in workflow config | **No** |
| **Zapier** | Zapier MCP (30K+ actions) | Supports custom MCP servers | **No** |
| **VS Code AI Toolkit** | First-class MCP tool type (March 2026) | MCP server in tool catalog | **No** |

**The implication:** Shipping native `SynthPanelTool` wrappers for each
framework is busywork with declining marginal value. The frameworks already
bridge MCP. The right strategy is to:
1. Keep the MCP server excellent.
2. Ship **examples** proving MCP works from each framework (discovery).
3. Build native wrappers only where MCP has gaps (direct Python import).

### Where MCP falls short

MCP is optimized for editor-hosted agents (Claude Code, Cursor, Windsurf)
where a long-lived stdio server process is natural. It's less ergonomic for:

- **Programmatic use in scripts/pipelines/notebooks** — spawning a subprocess
  for `synthpanel mcp-serve` just to call `quick_poll` adds latency and
  complexity. A direct `from synth_panel import quick_poll` is simpler.
- **Serverless/ephemeral environments** — Lambda, Cloud Run, GitHub Actions
  can't easily run a stdio server. A direct function call or HTTP endpoint
  is needed.
- **Agent frameworks without MCP bridges** — diminishing set, but some
  enterprise frameworks (proprietary orchestrators, internal platforms)
  may not support MCP yet.

This is why the **public Python SDK** (recommendation #1) is not redundant
with MCP. It serves the use cases MCP doesn't.

---

## C. Integration Surface Analysis

### C.1 Public Python SDK Convenience Layer

**What:** Export clean public functions at the package level so users can do:

```python
from synth_panel import quick_poll, run_panel, list_instruments

# One-liner: poll 5 personas on a question
results = quick_poll("What do you think of the name Traitprint?")

# Full panel with YAML files
results = run_panel(
    personas="personas.yaml",
    instrument="survey.yaml",
    model="sonnet"
)
```

**Why it matters:**
- Zero overhead: no MCP server process, no subprocess, no stdio protocol.
- Works in Jupyter notebooks, scripts, CI pipelines, Lambda, anywhere.
- Agent frameworks that prefer direct Python calls (LangChain `@tool`
  decorator wrapping a function) can wrap these in 3 lines.
- Enables the LangChain/CrewAI `@tool` pattern without MCP:

```python
from langchain_core.tools import tool
from synth_panel import quick_poll

@tool
def synthetic_focus_group(question: str) -> str:
    """Run a quick synthetic focus group poll on a question."""
    return quick_poll(question).summary
```

**Current state:** The functions exist internally (`orchestrator.py` has
`run_panel_parallel`, `run_multi_round_panel`, `ensemble_run`). But they are
implementation functions — complex signatures, internal types, no convenience
wrappers. The MCP server's tool handlers (`mcp/server.py`) contain the
user-friendly logic but are not importable as a Python API.

**Effort:** 1-2 days. Design the public API surface (5-8 functions), write
thin wrappers around the orchestrator + MCP server logic, add `__all__`
exports, write docstrings, add `examples/sdk_usage.py`.

**Impact:** HIGH. This is the most broadly useful integration surface after
MCP. Every Python developer who discovers SynthPanel via PyPI, pip, or import
gets a zero-friction onramp.

### C.2 Agent Framework Examples + "Works with X" Documentation

**What:** A `examples/integrations/` directory with working scripts:

| File | Framework | Lines of Code |
|---|---|---|
| `langchain_tool.py` | LangChain | ~20 (MCP bridge or direct SDK call) |
| `crewai_tool.py` | CrewAI | ~25 (MCP bridge) |
| `openai_agents.py` | OpenAI Agents SDK | ~15 (built-in MCP client) |
| `llamaindex_tool.py` | LlamaIndex | ~20 (`llama-index-tools-mcp`) |
| `microsoft_agent.py` | Microsoft Agent Framework | ~20 (MCP client) |
| `n8n_workflow.json` | n8n | Config file (no code) |

Plus a README section: "Works with LangChain, CrewAI, OpenAI Agents, LlamaIndex,
Microsoft Agent Framework, n8n, Zapier — any framework that supports MCP."

**Why it matters:**
- The MCP convergence thesis means SynthPanel already works with these
  frameworks — but nobody knows it. Zero examples exist.
- Each example is an independently indexable, searchable content page.
- A developer searching "how to run synthetic focus group from LangChain"
  finds this and discovers SynthPanel.
- sp-ege found SynthPanel invisible to AEO queries. Integration examples
  create exactly the kind of content that LLMs cite.

**Effort:** 1-2 days. Write 5-6 scripts, test each, add README section.

**Impact:** HIGH — especially for AEO/SEO. This is content marketing
disguised as developer documentation.

### C.3 Composio Connector

**What:** Register SynthPanel as a tool in the Composio marketplace (850+
tools, used by LangChain, CrewAI, Semantic Kernel, AutoGen).

**Why it matters:**
- One listing = presence in all major framework tool catalogs simultaneously.
- Composio handles auth, rate limiting, schema validation.
- A developer browsing Composio's tool catalog for "survey" or "focus group"
  finds SynthPanel.
- Composio's connectors are the modern equivalent of an npm package — they
  compound discovery.

**What it requires:**
- Composio uses a connector format (likely OpenAPI spec + auth config).
- SynthPanel would need to expose either its MCP tools via a thin HTTP API
  wrapper, or provide a Python function-based connector.
- Composio supports "bring your own tool" — the connector runs locally.

**Effort:** 2-4 days. Understand Composio's connector format, build adapter,
test with LangChain + CrewAI, submit to Composio's marketplace.

**Impact:** HIGH — marketplace network effect. But requires understanding
Composio's tooling, which has its own learning curve.

### C.4 Claude Code Skills Library Expansion

**What:** Expand beyond the single `/focus-group` skill to a library of
purpose-built skills:

| Skill | File | Purpose |
|---|---|---|
| `/focus-group` | `skills/focus-group/SKILL.md` | Full focus group workflow (exists) |
| `/name-test` | `skills/name-test/SKILL.md` | Quick product/feature name poll |
| `/concept-test` | `skills/concept-test/SKILL.md` | Concept validation with target audience |
| `/survey-prescreen` | `skills/survey-prescreen/SKILL.md` | Pre-screen a survey instrument before real fieldwork |
| `/pricing-probe` | `skills/pricing-probe/SKILL.md` | Pricing sensitivity analysis using bundled v3 instrument |

**Why it matters:**
- Claude Code users get task-specific entry points instead of one generic skill.
- Each skill is a tightly scoped workflow that reduces the user's cognitive
  load: "just type `/name-test`" is better than "set up personas, design
  an instrument, configure the model, run the panel."
- Skills ship with the repo (`.claude-plugin/plugin.json` references them)
  so any Claude Code user who installs the plugin gets them automatically.
- Skills are also discoverable via `.claude/skills/` by Claude Code's
  automatic skill loading — they can be suggested proactively.

**Effort:** 1 day. Each skill is a ~50-line SKILL.md following the existing
pattern. The `/focus-group` skill is a proven template.

**Impact:** MEDIUM-HIGH within the Claude Code ecosystem; zero impact outside
it. Good leverage because Claude Code is SynthPanel's most natural home —
an agent that can invoke MCP tools is the target user.

### C.5 Docker Image

**What:** Publish a Docker image that runs `synthpanel mcp-serve` (and/or
exposes the Python SDK as an HTTP API).

```bash
docker run -e ANTHROPIC_API_KEY=sk-... synthpanel/synthpanel mcp-serve
```

**Why it matters:**
- Agents in serverless environments (Lambda, Cloud Run, GitHub Actions)
  can spin up a SynthPanel container as a tool-call target.
- Devcontainer already exists (`.devcontainer/devcontainer.json`) with a
  base image (`ghcr.io/dataviking-tech/ai-dev-base:edge`). A production
  image is a subset.
- n8n, Zapier, and custom orchestrators that support Docker-based tools
  get a zero-install path.

**Effort:** 1-2 days. Dockerfile based on existing devcontainer. Publish to
GHCR and Docker Hub.

**Impact:** MEDIUM. Enables specific use cases (serverless, ephemeral) but
most developers will pip-install. Worth doing after the top 3.

### C.6 VS Code Extension (GUI)

**What:** A dedicated VS Code extension with GUI panels for persona
management, instrument editing, result visualization.

**Why it matters:**
- Deep integration beyond what MCP config provides.
- Visual feedback loop: see persona cards, preview instrument flows, browse
  results in a panel.
- VS Code Marketplace is a discovery channel.

**Why to defer:**
- MCP config already provides tool integration for Claude Code, Cursor,
  Windsurf, Copilot.
- A GUI extension is weeks of JavaScript/TypeScript work with ongoing
  maintenance.
- VS Code AI Toolkit now supports MCP as a first-class tool type (March
  2026), so the "tools" integration is already handled by MCP.
- The GUI portion (persona editing, result viz) is nice-to-have but does
  not block agent adoption.

**Effort:** 2-4 weeks.

**Impact:** MEDIUM — discovery channel is valuable but the effort/impact
ratio is poor compared to other options.

### C.7 Custom GPT / OpenAI Plugin

**What:** A ChatGPT-hosted wrapper around SynthPanel.

**Why to defer:**
- SynthPanel is local-first (bring-your-own-key, run on your machine).
  A Custom GPT inverts this by requiring a hosted API endpoint.
- OpenAI Agents SDK already supports MCP — the programmatic path is
  covered.
- Custom GPTs have a discoverability problem of their own (the GPT Store
  is noisy).

**Effort:** 1-2 weeks (need hosted endpoint + OAuth).

**Impact:** LOW relative to effort. The audience that uses Custom GPTs is
not the audience that would benefit most from YAML-defined reproducible
research.

### C.8 Webhook / Event-Driven Pattern

**What:** SynthPanel as a service that agents POST to and receive results
asynchronously (webhook callback or polling).

**Why to defer:**
- Adds server complexity (hosting, auth, queuing).
- No demonstrated demand.
- MCP's stdio transport and the Python SDK cover the two main invocation
  patterns (editor-hosted agent and programmatic script).
- Worth revisiting if SynthPanel ever ships a hosted/cloud offering.

**Effort:** 1-2 weeks.

**Impact:** LOW until demand materializes.

### C.9 Native Framework Tool Wrappers

**What:** `pip install synthpanel-langchain`, `pip install synthpanel-crewai`,
etc. Each is a thin Python package that wraps SynthPanel's MCP tools (or
direct SDK) as framework-native tools.

**Why to skip (for now):**
- MCP bridges already handle this. All major frameworks support MCP.
- Each native wrapper is a maintenance burden (API changes in the framework
  require wrapper updates).
- The examples directory (recommendation #2) proves the MCP bridge works
  with zero additional packages.
- If demand appears for a specific framework, build it then.

**Effort:** 1-2 days per framework.

**Impact:** LOW — redundant with MCP bridge. Better to invest that effort
in examples that prove MCP works.

---

## D. Competitor Integration Surface Comparison

| Feature | SynthPanel | Synthetic Users | FocusPanel.ai |
|---|---|---|---|
| CLI | Yes (`synthpanel`) | No | No |
| Python SDK | Internal functions (no public API) | Python + TypeScript SDKs (MIT) | Unknown |
| MCP server | Yes (12 tools, stdio) | No | No |
| Claude Code plugin | Yes | No | No |
| REST API | No | Yes | Yes |
| Custom GPT | No | Unknown | Unknown |
| LangChain/CrewAI wrapper | No (MCP bridge works) | No | No |
| Docker image | No (devcontainer exists) | Unknown | Unknown |
| Composio listing | No | No | No |
| Open source | Yes (MIT) | Yes (MIT) | No (SaaS) |
| Pricing | Free (BYOK) | SaaS | SaaS |

**SynthPanel's differentiation:**
- Only synthetic-research tool with MCP support (agent-native by design).
- Only open-source option with CLI + MCP + Claude Code plugin.
- BYOK (bring your own key) vs SaaS lock-in.
- The public Python SDK would close the one gap where Synthetic Users
  has an advantage (documented importable SDK).

---

## E. The "Agent Reaches for SynthPanel Automatically" Question

The bead asks: *"What patterns would make SynthPanel the default tool that
agents reach for automatically?"*

This is fundamentally an **AEO + registry + content** problem, not a
code problem. An agent's tool-selection is determined by:

1. **What tools are registered** — MCP config, Composio catalog, framework
   tool list. If SynthPanel's MCP server is in the agent's config, it's
   reachable.
2. **What the LLM knows** — training data, retrieval-augmented context,
   indexed documentation. If the LLM has never seen "synthpanel" in its
   training or retrieval corpus, it can't recommend it.
3. **How tools are described** — the MCP server's tool names and descriptions
   determine whether the LLM selects them for a given task.

The sp-ege audit addressed #2 (AEO content, registry listings). This audit
addresses #1 (integration surfaces) and #3 (tool descriptions). Together
they form a complete strategy:

- **Short term:** Fix MCP registry presence (sp-ege beads) + ship
  integration examples (this audit) = agents can find and use SynthPanel.
- **Medium term:** Public Python SDK + Composio listing + AEO content =
  SynthPanel becomes the default answer when an agent (or a human using an
  agent) asks "how do I run a synthetic focus group?"
- **Long term:** Skills library + Docker image + eventual REST API =
  SynthPanel is reachable from every invocation pattern (editor, script,
  serverless, workflow, marketplace).

---

## F. Impact/Effort Matrix

```
                    LOW EFFORT              MEDIUM EFFORT             HIGH EFFORT
              ┌─────────────────────┬──────────────────────┬────────────────────────┐
 HIGH IMPACT  │ 1. Python SDK       │ 3. Composio connector│                        │
              │ 2. Framework        │                      │                        │
              │    examples + docs  │                      │                        │
              ├─────────────────────┼──────────────────────┼────────────────────────┤
 MED IMPACT   │ 4. Skills library   │ 5. Docker image      │ 6. VS Code extension   │
              │    expansion        │                      │                        │
              ├─────────────────────┼──────────────────────┼────────────────────────┤
 LOW IMPACT   │ 9. Native framework │                      │ 7. Custom GPT          │
              │    wrappers (skip)  │                      │ 8. Webhook/async       │
              └─────────────────────┴──────────────────────┴────────────────────────┘
```

---

## G. Methodology and Verification

### Sources consulted

**Agent frameworks (verified MCP support):**
- OpenAI Agents SDK: [openai.github.io/openai-agents-python/mcp](https://openai.github.io/openai-agents-python/mcp/) — built-in MCP client
- LlamaIndex: [pypi.org/project/llama-index-tools-mcp](https://pypi.org/project/llama-index-tools-mcp/) — v0.4.8 bridge, released Feb 2026
- CrewAI: [docs.crewai.com/en/concepts/tools](https://docs.crewai.com/en/concepts/tools) — `crewai-tools[mcp]`
- Microsoft Agent Framework 1.0: [devblogs.microsoft.com/agent-framework](https://devblogs.microsoft.com/agent-framework/microsoft-agent-framework-version-1-0/) — shipped April 2026 with MCP clients
- LangChain: [python.langchain.com/docs/modules/agents/tools/custom_tools](https://python.langchain.com/docs/modules/agents/tools/custom_tools) — `@tool` decorator and `BaseTool`
- n8n: [docs.n8n.io AI Agent node](https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain.agent/tools-agent/) — built-in MCP support
- Zapier MCP: [zapier.com/mcp](https://zapier.com/mcp) — 30K+ actions via MCP

**Marketplaces:**
- Composio: [composio.dev](https://composio.dev/) — 850+ tools, framework-agnostic
- VS Code AI Toolkit MCP: [VS Code extension guides](https://code.visualstudio.com/api/extension-guides/ai/mcp) — MCP as first-class tool type

**Competitors:**
- Synthetic Users: [docs.syntheticusers.com](https://docs.syntheticusers.com/) — Python/TypeScript SDKs (MIT)
- FocusPanel.ai: [focuspanelai.azurewebsites.net](https://focuspanelai.azurewebsites.net/) — SaaS, web-only

**Claude Code integration patterns:**
- Custom slash commands: [code.claude.com/docs/en/slash-commands](https://code.claude.com/docs/en/slash-commands) — `.claude/commands/` and `.claude/skills/` directories

### Internal code audit
- `src/synth_panel/orchestrator.py`: `run_panel_parallel()` (line 501), `run_multi_round_panel()` (line 638), `ensemble_run()` (line 847)
- `src/synth_panel/mcp/server.py`: 12 tool handlers wrapping orchestrator functions
- `.claude-plugin/plugin.json`: MCP server config + skill reference
- `skills/focus-group/SKILL.md`: 5-step workflow skill
- `.devcontainer/devcontainer.json`: `ghcr.io/dataviking-tech/ai-dev-base:edge`

---

## H. Filed Child Beads

Filed 2026-04-15 as hierarchical children of `sp-2cw`, all priority P3 (backlog):

| Bead | Title | Tier |
|---|---|---|
| `sp-2cw.1` | Design & ship public Python SDK convenience layer | **Top 3 — highest impact** |
| `sp-2cw.2` | Write "Works with X" integration examples for 5 agent frameworks | **Top 3 — highest impact** |
| `sp-2cw.3` | Register SynthPanel as a Composio connector | **Top 3 — highest impact** |
| `sp-2cw.4` | Expand Claude Code skills library beyond `/focus-group` | Second tier |
| `sp-2cw.5` | Publish Docker image for ephemeral/serverless agent invocation | Second tier |
| `sp-2cw.6` | Add "Works with" section to README linking integration examples | Second tier (bundle with sp-2cw.2) |
| `sp-2cw.7` | Evaluate VS Code extension: decision doc, not yet build | Decision-gate |

**Explicitly not filed** (recommendation: skip):
- Native LangChain/CrewAI/LlamaIndex tool wrappers → redundant with MCP bridges
- Custom GPT / OpenAI plugin → inverts SynthPanel's local-first architecture
- Webhook/event-driven service → no demonstrated demand
- Native n8n/Zapier nodes → both platforms already support MCP

Rationale for each skipped item is documented in the audit body (Sections C.7–C.9).
