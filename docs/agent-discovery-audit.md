# SynthPanel Agent Discovery & Install Ergonomics Audit

**Author:** crew/advo (Developer Advocate specialization)
**Date:** 2026-04-15
**Parent bead:** sp-ege
**Status:** Research complete, recommendations filed as child backlog beads
**Scope:** Marketing and discovery surface only. No product/MCP-server-code changes
proposed.

---

## Executive Summary

SynthPanel ships a capable product (v0.9.0, MIT-licensed, 12-tool MCP server,
SPS 0.90 SynthBench score) but **its agent-facing surface is under-marketed on
every channel that matters for AI-era discovery**. The MCP server — the most
differentiating agent-era feature — is a one-line footnote on synthpanel.dev, a
section buried at line 335 of a 470-line README, and absent from every major
MCP registry.

LLM-based search ("does a tool for synthetic focus groups exist?") currently
returns Synthetic Users, FocusPanel.ai, Delve.ai, and POPJAM — **not
SynthPanel**. The `"synthpanel" MCP` query surfaces zero indexed pages that
mention the project by name. The project is effectively invisible to the
agent-era discovery pipeline it is architecturally best positioned to serve.

The good news: every gap in this audit is a small, bounded fix. There is no
product rework. Every recommendation below is a copy-paste edit, a PyPI
metadata bump, a single PR to an awesome-list, or a short piece of positioning
content.

### Top 3 Quick Wins (hours)

1. **Lift the MCP story to hero level on synthpanel.dev + README.** Today both
   assets open with "terminal" framing; the `[mcp]` extra is a footnote. Add a
   copy-paste-able Claude Code / Cursor / Windsurf JSON config block as
   section 2, immediately after the install command.
2. **Fix PyPI metadata hygiene in one `pyproject.toml` edit.** Current URLs
   point to a stale repo name (`synth-panel`, hyphenated — survives only via
   GitHub's rename-redirect). Keywords omit `mcp`, `agent`, `claude`,
   `model-context-protocol`, `survey-research`. Classifiers omit
   `Topic :: Software Development :: Libraries :: Python Modules` and
   `Intended Audience :: Information Technology`.
3. **Submit to `punkpeye/awesome-mcp-servers` under Research 🔬.** One PR, no
   code changes. This list has 84.8k stars; the Research category already
   exists; an adjacent `Persona MCP Server` entry is listed on glama.ai,
   proving the category has real demand.

### Top 3 Bigger Plays (days)

1. **AEO positioning content: "SynthPanel vs. Synthetic Users / FocusPanel.ai
   / POPJAM — when open-source MCP beats the SaaS."** One 1500-word article
   that LLMs can cite when asked about synthetic-respondent tools.
   Independently valuable for GEO (generative engine optimization).
2. **Multi-registry submission sweep: glama.ai, PulseMCP, MCPHub, Smithery,
   Klavis.** Glama alone indexes 21,545 servers — we are not one of them.
   Each is a separate free submission and each improves LLM retrieval.
3. **MCP-native README restructure + dedicated `/mcp` site route.** Today
   synthpanel.dev is a single page. A `/mcp` deep page that mirrors the
   reference-server pattern (tools list first, config-by-editor in `<details>`
   blocks, one-click VS Code install badges) gives both humans and crawlers a
   canonical landing spot for agent-era discovery.

---

## A. Audit of Current State

### A.1 synthpanel.dev (live fetch, 2026-04-15)

| Signal | State | Gap |
|---|---|---|
| `<title>` | "synthpanel — Run synthetic focus groups with any LLM" | No "MCP", "agent", or "Claude Code" in the page title |
| Hero tagline | "Run synthetic focus groups with any LLM." | Frames the product as a CLI-first research tool, not agent-native |
| MCP mention | Single 9-word footnote under install command | Not positioned as a feature; no JSON snippet; no "Use with X" block |
| Cards | PyPI · GitHub · SynthBench | No MCP card; no Claude Code / Cursor / Windsurf card |
| Routes | `/` only (single-page site) | No `/mcp`, no `/docs`, no `/tools` — nothing for a crawler to index beyond the homepage |
| `sitemap.xml` | Lists only `https://synthpanel.dev/` | Matches the single-page reality but caps the discoverable surface |
| OG / Twitter cards | Present, brief | Do not mention MCP or agent integration |
| Canonical | `https://synthpanel.dev/` | Correct |

**Verdict:** The site describes a CLI research tool. It does not advertise
that SynthPanel is one of a small number of MCP-capable synthetic-research
servers.

### A.2 README (`README.md`, 470 lines)

| Section | Line | Above/Below Fold |
|---|---|---|
| Title + badges | 1–7 | Above |
| Tagline, install, primary pitch | 10–20 | Above |
| Why | 22–29 | Above |
| Quick Start | 31–53 | Near fold |
| What You Get | 55–79 | Below |
| Defining Personas | 81–107 | Below |
| Defining Instruments | 109–128 | Below |
| Adaptive Research (v3) | 130–234 | Below |
| Examples | 236–243 | Below |
| LLM Provider Support | 245–297 | Below |
| Architecture | 299–334 | Below |
| **MCP Server (Agent Integration)** | **335–367** | **Deep below** |
| Output Formats | 369–379 | Below |
| Budget Control | 381–388 | Below |
| Persona Prompt Templates | 390–407 | Below |
| Methodology Notes | 409–419 | Below |
| Multi-Model Ensemble | 421–438 | Below |
| Versions | 440–450 | Below |
| Contributing | 452–454 | Below |
| **MCP Server Documentation** (link to `docs/mcp.md`) | 456–458 | Below |
| SynthBench | 460–466 | Below |

**Verdict:** The MCP story is the ~7th section visitors encounter. The
opening pitch *does* mention "your agent's MCP tool call" (line 10) — which is
good — but the dedicated MCP Server section is past ten other sections,
including 100+ lines of v3 branching detail that most readers won't finish
before bouncing. Reference Python MCP servers (e.g., `mcp-server-fetch`,
`mcp-server-git`) open with Tools + Install + Config in the top third; we do
the opposite.

### A.3 PyPI metadata (verified via `https://pypi.org/pypi/synthpanel/json`)

| Field | Value | Issue |
|---|---|---|
| `version` | `0.9.0` | Current |
| `summary` | "Run synthetic focus groups and user research panels using AI personas. CLI tool,…" | Truncated; omits "MCP / agent integration" |
| `home_page` | `null` | Legacy field unset — fine, `project_urls.Homepage` covers it |
| `project_urls.Homepage` | `github.com/DataViking-Tech/synth-panel` | **Wrong repo name.** Canonical is `SynthPanel` (no hyphen). GitHub 301-redirects the old name via rename-redirect, but the URL shown to pip users is stale |
| `project_urls.Repository` | Same as above | Same issue |
| `project_urls.Changelog` | Same base URL | Same issue |
| `project_urls.Issues` | Same base URL | Same issue |
| `project_urls` missing | No `Homepage = https://synthpanel.dev`, no `Documentation = https://synthpanel.dev/…`, no `Funding`, no "Use with Claude Code" link | Misses primary marketing channel |
| `keywords` | `llm, research, focus-group, synthetic, personas, user-research, ai` | Missing: `mcp`, `model-context-protocol`, `agent`, `claude`, `claude-code`, `cursor`, `windsurf`, `survey-research`, `synthetic-respondents` |
| `classifiers` | 8 classifiers, only one agent-relevant: `Topic :: Scientific/Engineering :: Artificial Intelligence` | Missing: `Topic :: Software Development :: Libraries :: Python Modules`, `Topic :: Scientific/Engineering :: Information Analysis`, `Intended Audience :: Information Technology`, `Operating System :: OS Independent`, `Environment :: Console` |

**Verdict:** The metadata is functional but gives the search-ranking
algorithms on PyPI (and any downstream scraper — Glama, PulseMCP, etc.)
almost nothing to match an agent-era query against.

### A.4 GitHub repo (`DataViking-Tech/SynthPanel`)

| Signal | State | Gap |
|---|---|---|
| Topics | `python`, `mcp`, `survey-research`, `llm-evaluation`, `persona-simulation`, `synthetic-respondents` | Good — MCP topic is set. Could add: `model-context-protocol`, `claude-code`, `ai-agents`, `focus-group` |
| About | "Run synthetic focus groups and user research panels using AI personas. CLI tool, Python library, any LLM." | Omits "MCP / agent integration" — same as PyPI summary |
| Stars / forks | 0 / 0 | No social proof yet |
| Social preview card | Not set | Default auto-generated GitHub card; a custom card highlighting MCP support would be a visible differentiator |
| Releases | 8 tags through v0.9.0 | Good |
| README MCP section visibility | Below the fold | See A.2 |
| Docs directory | `docs/` exists, indexed | Good; already has `mcp.md`, `adapter-guide.md`, `stability.md` |

### A.5 Docs directory

`docs/mcp.md` is well-written: the 12 tools, 4 resources, 3 prompt templates,
response shape, data storage paths. It is the single best piece of agent-facing
documentation in the repo. **But it is not linked from the homepage, and
synthpanel.dev does not render it.**

---

## B. External Research: How Comparable Projects Get Found

### B.1 MCP registry landscape (2026-04-15)

| Registry | URL | Scale | SynthPanel listed? | Submission |
|---|---|---|---|---|
| `punkpeye/awesome-mcp-servers` | github.com/punkpeye/awesome-mcp-servers | 84.8k ⭐, curated categories incl. **Research 🔬** | **No** | PR |
| `modelcontextprotocol/servers` (Resources section) | github.com/modelcontextprotocol/servers | Official; links out to community registries | **No** | PR |
| glama.ai MCP registry | glama.ai/mcp/servers | **21,545 servers** indexed; categories + language + hosting filters; "claim your server" flow | **No** | "Add Server" button + auto-scrape |
| PulseMCP | pulsemcp.com | Registry + weekly newsletter | **No** | `/submit` form |
| MCPHub | mcphub.com | Registry with user reviews | **No** | Form |
| Smithery | smithery.ai | Hosted MCP runtime + registry | **No** | Platform hosting |
| Klavis AI | klavis.ai | Open-source MCP infra | **No** | Platform integration |

**A direct competitor is already indexed.** On glama.ai, searching "survey"
returns 16 entries including a "Persona MCP Server" ("AI-powered persona
analysis and Focus Group Interview (FGI) system that creates dynamic personas
from survey responses, generates contextual follow-up questions"). This is
significant: it means the category has traction in the registry, a search for
"persona" or "focus group" on glama.ai today surfaces a competitor, and a
SynthPanel listing would compete directly.

### B.2 Reference Python MCP server README pattern

Studied `mcp-server-fetch` and `mcp-server-git` (official reference servers).
Common structure:

1. **Overview** — 1–2 sentences: "A Model Context Protocol server that
   provides X capabilities. This server enables LLMs to Y."
2. **Tools** — numbered list, each with input parameters (type + optionality)
   and return shape. Comes *before* installation — so a reader scanning can
   see capabilities in 10 seconds.
3. **Installation** — `uvx` first (recommended), then `pip`. Python MCP
   packages use the `mcp-server-*` prefix convention.
4. **Configuration** — platform-by-platform in `<details>` tags:
   Claude Desktop → VS Code (with one-click install badges) → Zed → Cursor.
   Each shows a minimal JSON snippet.
5. **Debugging / Development / License.**

**SynthPanel's README inverts this:** we lead with CLI usage, persona YAML,
instrument YAML, v3 branching (100+ lines), architecture — all before we get
to MCP. A reader looking for "can this be called by an agent?" has to scroll
through everything else first.

### B.3 PyPI classifier conventions for agent-era packages

Modern LLM/agent packages commonly stack:
- `Development Status :: X`
- `Intended Audience :: Developers` ✓ (we have this)
- `Intended Audience :: Science/Research` ✓ (we have this)
- `Intended Audience :: Information Technology` — **missing**
- `Topic :: Scientific/Engineering :: Artificial Intelligence` ✓ (we have this)
- `Topic :: Scientific/Engineering :: Information Analysis` — **missing**
- `Topic :: Software Development :: Libraries :: Python Modules` — **missing**
- `Topic :: Office/Business :: Scheduling` — N/A
- `Environment :: Console` — **missing** (we have a CLI)
- `Operating System :: OS Independent` — **missing**

No standardized classifier exists for "MCP server" yet. The community is
discussing `Framework :: MCP` on the Python Packaging Authority tracker but
nothing is accepted. Until it is, the right play is to **overload keywords**
(which PyPI search indexes) rather than wait for a classifier.

### B.4 AEO / GEO audit — do LLMs know SynthPanel exists?

Tested via WebSearch (simulating what modern LLMs retrieve):

| Query | SynthPanel in top 10? | What shows up instead |
|---|---|---|
| `synthetic focus group AI personas tool open source MCP 2026` | **No** | Synthetic Users, FocusPanel.ai, Delve.ai, POPJAM, Maven course, lunar.dev |
| `"synthpanel" MCP model context protocol` | **No** (zero matches for the quoted name) | Only generic MCP explainers |
| `synthpanel Python AI personas focus group pip install` | **No** | Personaut, JasperHG90/persona, DSPy, AI-Persona-Lab, Sybil-Swarm |

**This is the clearest signal in the audit.** A user asking an LLM "is there
an open-source MCP server for synthetic focus groups?" today gets back a
generic "I don't know of one" answer or is pointed at commercial competitors.
The project exists, it's on PyPI, the repo is public — but it's not yet
dense enough in the search-retrieval corpus to be surfaced. Every
recommendation in Section C either (a) increases retrieval density directly
(registry listings, classifier/keyword changes, README changes get crawled) or
(b) produces content that LLMs can cite (positioning article, "Use with X"
guide).

---

## C. Recommendations

Each recommendation below is filed as a child bead against `sp-ege`. Bead IDs
are recorded after filing in the companion section of this document.

### C.1 Quick Wins (hours each)

**Q1. Lift the MCP story above the fold on synthpanel.dev.**
Add a prominent section between the install command and the Quick Start. Use
the reference-server pattern: one-line pitch + JSON config snippet + "Works
with Claude Code · Cursor · Windsurf · Zed" editor badge row. Include a
copy-button on the JSON block matching the existing install copy-button.

**Q2. Lift the MCP story to README section 2.**
Move the "MCP Server (Agent Integration)" section from line 335 to
immediately after Quick Start (before "What You Get"). Mirror reference-server
structure: 12 tools listed with one-line descriptions, config JSON for
Claude Code, link to `docs/mcp.md` for full reference. Keep the old section
content but move it forward.

**Q3. Fix PyPI metadata.**
Edit `pyproject.toml`:
- Change all 4 `project.urls` to canonical casing (`SynthPanel`, no hyphen)
  *and* add `Homepage = "https://synthpanel.dev"`, `Documentation = "https://synthpanel.dev"`, `"MCP Reference" = "https://github.com/DataViking-Tech/SynthPanel/blob/main/docs/mcp.md"`.
- Expand `keywords` to include: `mcp`, `model-context-protocol`, `agent`,
  `claude`, `claude-code`, `cursor`, `windsurf`, `survey-research`,
  `synthetic-respondents`, `market-research`.
- Add classifiers: `Intended Audience :: Information Technology`,
  `Topic :: Scientific/Engineering :: Information Analysis`,
  `Topic :: Software Development :: Libraries :: Python Modules`,
  `Environment :: Console`, `Operating System :: OS Independent`.
- Update `description` to include "MCP server included" phrase.
- Release as v0.9.1 (patch, metadata-only).

**Q4. Add a README "Use with Claude Code / Cursor / Windsurf / Zed" section.**
Copy-paste-able JSON config for each editor, in a top-of-README position. Each
block shows the full `mcpServers` entry a user drops into their editor config.
Pattern: reference-server READMEs use `<details>` tags per editor; adopt that.

**Q5. Add GitHub topics missing from current set.**
Current: 6 topics. Add: `model-context-protocol`, `claude-code`, `ai-agents`,
`focus-group`, `persona`. Five minutes in the repo Settings.

### C.2 Bigger Plays (days each)

**B1. Submit to `punkpeye/awesome-mcp-servers` under Research 🔬.**
Single PR. Pattern for the entry (verbatim from list format):

```
- [DataViking-Tech/SynthPanel](https://github.com/DataViking-Tech/SynthPanel)
  📇 🐍 - Run synthetic focus groups with any LLM. 12 MCP tools: run_panel,
  run_quick_poll, persona/instrument pack management. YAML-defined personas
  and branching research instruments.
```

Where 📇 is the TypeScript-or-similar lang badge — adjust to the Python
equivalent per the list's badge key (🐍 is used in the list for Python).

**B2. Multi-registry submission sweep.**
Same one-liner, different intake forms: glama.ai (Add Server), PulseMCP
(/submit), MCPHub, Smithery (if hosting-compatible; if not, skip), Klavis.
Each listing improves LLM-retrieval odds by a compounding amount because
several LLMs cite those registries directly when asked "what MCP servers
exist for X."

**B3. Write an AEO positioning article.**
Title candidate: *"SynthPanel vs. Synthetic Users vs. FocusPanel.ai — when
open-source MCP beats the SaaS."* Publish on dataviking.tech /blog and
cross-post to Medium. Covers: (a) why LLM-agnostic matters (lock-in,
bring-your-own-key, cost control), (b) why YAML-defined instruments matter
(reproducibility, version control, auditing), (c) why MCP matters (in-editor
invocation, agent-era workflows), (d) honest limitations (sycophancy,
clustering around means — we already document these in the README, lift
them into the article). Target: 1500 words, cite SynthBench SPS 0.90 as the
quantitative proof point.

**B4. Add a dedicated `/mcp` deep page on synthpanel.dev.**
Mirror `docs/mcp.md` as a static HTML page at `synthpanel.dev/mcp` — same
content, indexable, with copy-buttons on every config block. Add a sitemap
entry. Link from the hero card row. This gives LLMs and search engines a
canonical agent-integration landing page that they can cite.

**B5. Add a custom social preview card to the GitHub repo.**
1280×640 PNG showing "SynthPanel · MCP server for synthetic focus groups ·
12 tools · any LLM." Uploaded via repo Settings. Shows up every time the
repo is linked in Slack, Discord, or HN. One afternoon of design work.

**B6. Add README badges: MCP-enabled, discussed-on-X registry, MIT license
version-pinned.**
Pattern studied: `[![MCP](https://img.shields.io/badge/MCP-enabled-purple)]`
as a community convention. Link to `docs/mcp.md` or the Glama listing once
it exists.

---

## D. What Is NOT Recommended

For the record, I considered and rejected:

1. **Renaming the PyPI package to `mcp-server-synthpanel`** — follows the
   reference-server convention but breaks every existing installation,
   documentation link, and the brand. The convention is for servers that
   are *only* MCP. SynthPanel is also a CLI + Python library. Keep the
   current name; surface MCP-ness through the metadata and content layers.
2. **Splitting the MCP server into its own package** — same argument.
   Current packaging (`pip install synthpanel[mcp]`) is the right call.
3. **Chasing a `Framework :: MCP` PyPI classifier** — no upstream consensus
   yet. Revisit in 6 months.
4. **Paying for hosted MCP registry placement (Smithery)** — not needed
   when free registries (Glama, PulseMCP, awesome-list) cover the same
   retrieval surface. Reconsider if traction warrants.
5. **Changing the `synthpanel mcp-serve` CLI entrypoint name** — it's
   non-canonical (`mcp-server-*` is the convention) but changing it is a
   breaking change for any existing user's config. Document, don't rename.

---

## E. Methodology and Verification

- All live-page findings verified via `curl`/WebFetch on 2026-04-15.
- PyPI metadata verified via the JSON API (`/pypi/synthpanel/json`).
- Classifier recommendations cross-checked against currently-accepted values
  at <https://pypi.org/classifiers/>.
- Awesome-list category structure verified by fetching raw `README.md` from
  `punkpeye/awesome-mcp-servers` (84.8k ⭐ as of fetch).
- AEO queries run via web search in April 2026; results are retrieval-layer
  snapshots, not a permanent claim about what every LLM knows.
- The `synth-panel` → `SynthPanel` URL redirect was verified with `curl -I` —
  the hyphenated URL returns `HTTP 301` to the canonical; no broken links,
  but the metadata is stale.

---

## F. Filed Child Beads

Filed 2026-04-15 as hierarchical children of `sp-ege`, all priority P3 (backlog):

| Bead | Title | Category |
|---|---|---|
| `sp-ege.1` | Lift MCP story above the fold on synthpanel.dev | Quick win (site) |
| `sp-ege.2` | Lift MCP section to README position 2 (after Quick Start) | Quick win (README) |
| `sp-ege.3` | Fix PyPI metadata: URLs, keywords, classifiers (v0.9.1 patch) | Quick win (PyPI) |
| `sp-ege.4` | Add "Use with Claude Code / Cursor / Windsurf / Zed" section to README | Quick win (README) |
| `sp-ege.5` | GitHub repo hygiene: topics + social preview + About description | Quick win (GitHub) |
| `sp-ege.6` | Submit SynthPanel to `punkpeye/awesome-mcp-servers` (Research category) | Bigger play (ecosystem) |
| `sp-ege.7` | Multi-registry submission sweep: Glama, PulseMCP, MCPHub, Klavis | Bigger play (ecosystem) |
| `sp-ege.8` | Write AEO positioning article: SynthPanel vs commercial alternatives | Bigger play (content) |
| `sp-ege.9` | Add `/mcp` deep page to synthpanel.dev | Bigger play (site) |

Each bead carries its own acceptance criteria, labels (`discovery`, plus
one of `site`/`readme`/`pypi`/`github`/`registry`/`content`), and may be
picked up independently. The three quick wins (Q1–Q5 in Section C, beads 1–5
here) should ship together as a coordinated discovery push. The four bigger
plays (beads 6–9) are individually valuable and do not block each other.
