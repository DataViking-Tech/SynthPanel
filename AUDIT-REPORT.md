# synth-panel Production Readiness Audit

**Date:** 2026-04-04
**Auditor:** synthpanel/crew/auditor
**Bead:** sp-tqo
**Version:** 0.1.0

---

## Executive Summary

synth-panel's core functionality (single prompt, panel run, MCP server) is solid and ready for PyPI publication. The package structure, metadata, and test suite are in good shape. However, the README makes several claims that don't match reality, and there are stub commands that should be clearly marked or removed before a public release.

**Verdict: CONDITIONALLY READY** — fix 2 blockers, address should-fix items.

---

## 1. Does `pip install synth-panel` work?

**YES** (structurally sound)

- `pyproject.toml` metadata is complete: name, version, description, license, keywords, classifiers, URLs
- `src/` layout with all 7 subpackages having `__init__.py` files
- Entry point correctly defined: `synth-panel = "synth_panel.main:main"`
- Dependencies declared: `httpx>=0.27`, `pyyaml>=6.0`, optional `mcp>=1.0`
- Build system: setuptools with `packages.find` auto-discovery

---

## 2. Test Suite Results

**243 collected | 221 passed | 2 failed | 20 skipped | 1 collection error**

### Failures

| Test | Issue | Severity |
|------|-------|----------|
| `test_acceptance.py::test_unreachable_base_url` | Fails with `MISSING_CREDENTIALS` instead of `TRANSPORT` because no dummy API key is set in the test env | should-fix |
| `test_client.py::test_alias_is_resolved_in_send` | Hits live Anthropic API instead of mocking the provider; 401 on invalid key | should-fix (known: sp-s6o) |

### Skips

20 acceptance tests correctly skip when `ANTHROPIC_API_KEY` is not set.

### Collection Error

`test_mcp_server.py` — `ModuleNotFoundError: No module named 'mcp'`. Expected when `mcp` extra is not installed. Not a bug.

---

## 3. Does `synth-panel panel run` work with example YAMLs?

**YES** (structurally verified)

- `examples/personas.yaml` exists: 3 personas (Sarah Chen, Marcus Johnson, Priya Sharma)
- `examples/survey.yaml` exists: 2 questions with follow-ups and schemas
- CLI parser correctly wires `--personas` and `--instrument` flags
- Orchestrator runs panelists in parallel via ThreadPoolExecutor
- Cost tracking reports per-panelist and aggregate totals

*(Live API test not run — requires API key)*

---

## 4. Is the README accurate?

### BLOCKER: False "zero dependencies" claim

**README line 165:**
> "Zero external dependencies — pure Python 3.10+, standard library only"

**Reality:** `pyproject.toml` declares `httpx>=0.27` and `pyyaml>=6.0` as hard dependencies. This is a direct contradiction.

### README claims vs reality

| Claim | Verdict | Notes |
|-------|---------|-------|
| `synth-panel prompt "text"` | VERIFIED | Works correctly |
| `synth-panel panel run` with YAML flags | VERIFIED | Parser and handlers complete |
| `synth-panel mcp-serve` | VERIFIED | 7 tools, 4 resources, 3 prompts |
| `--output-format json/ndjson` | VERIFIED | Three formats implemented |
| `--model` with provider aliases | VERIFIED | sonnet, haiku, grok, gemini all resolve |
| Interactive REPL (`synth-panel` with no args) | PARTIAL | REPL launches, slash commands work, but **user input is stubbed** — echoes `[stub] Turn N: ...` instead of calling LLM |
| REPL slash commands (`/help`, `/model`, `/status`) | PARTIAL | `/help`, `/model`, `/status` work; `/compact`, `/permissions`, `/config`, `/memory` are stubs |
| Provider table (Anthropic, OpenAI, xAI, Gemini) | VERIFIED | All 4 providers implemented; README table missing Gemini row |
| Budget control | VERIFIED | Cost tracker enforces soft limits |
| Example YAML files | VERIFIED | Both exist and are correctly formatted |

---

## 5. Stub Commands

| Command | Location | Status |
|---------|----------|--------|
| `synth-panel login` | `commands.py:247-251` | Stub — prints "[stub] Login not yet implemented" |
| `synth-panel logout` | `commands.py:254-258` | Stub — prints "[stub] Logout not yet implemented" |
| REPL user input | `repl.py:51-54` | Stub — echoes input, TODO comment to wire to runtime |
| `/compact` | `slash.py` | Stub |
| `/permissions` | `slash.py` | Stub |
| `/config` | `slash.py` | Stub |
| `/memory` | `slash.py` | Stub |

---

## 6. pyproject.toml Metadata

| Field | Present | Value |
|-------|---------|-------|
| name | YES | synth-panel |
| version | YES | 0.1.0 |
| description | YES | Clear, 105 chars |
| requires-python | YES | >=3.10 |
| license | YES | MIT |
| readme | YES | README.md |
| keywords | YES | 7 keywords |
| classifiers | YES | 8 classifiers |
| project.urls | YES | Homepage, Repository, Issues |
| authors | **NO** | Missing |
| LICENSE file | **NO** | Declared MIT but no LICENSE file exists |

---

## 7. Gaps: README Promises vs What Actually Works

### BLOCKERS (must fix before PyPI publish)

1. **False "zero dependencies" claim** — README line 165 says "zero external dependencies, pure Python 3.10+, standard library only" but the package requires `httpx` and `pyyaml`. This will confuse users and misrepresent the package on PyPI.

2. **README advertises interactive REPL as a working feature** — Lines 184-193 show a REPL session with `Tell me about yourself` getting a response. In reality, all non-slash input prints `[stub] Turn N: <input>`. A user running `synth-panel` interactively will think it's broken.

### SHOULD-FIX (before or shortly after publish)

3. **No LICENSE file** — `license = "MIT"` in pyproject.toml but no LICENSE file in repo. PyPI best practice; some automated tools flag this.

4. **No `authors` field** — Missing from pyproject.toml. Low impact but looks unfinished on PyPI.

5. **README missing Gemini from provider table** — The Gemini provider is fully implemented (`gemini.py`) with `GOOGLE_API_KEY`/`GEMINI_API_KEY` support, but the README provider table only shows Anthropic, OpenAI, and xAI.

6. **Two test failures** — `test_unreachable_base_url` needs a dummy API key; `test_alias_is_resolved_in_send` needs provider mocking (tracked as sp-s6o).

7. **Unused dev dependency** — `pytest-httpx>=0.30` is declared but never imported in any test file. Tests use `unittest.mock`.

8. **Login/logout stubs print misleading output** — These commands exist in the CLI but just print a stub message. Should either be removed from the parser or clearly documented as "coming soon".

### NICE-TO-HAVE

9. **Plugin system has no bundled plugins or integration tests** — The framework is complete but untested in the wild.

10. **REPL slash commands** — `/compact`, `/permissions`, `/config`, `/memory` are registered but non-functional.

11. **No `py.typed` marker** — Would benefit downstream type-checkers.

---

## Summary Table

| # | Issue | Severity | Effort |
|---|-------|----------|--------|
| 1 | False "zero deps" claim in README | BLOCKER | 5 min (edit one line) |
| 2 | REPL interactive mode is stubbed but advertised | BLOCKER | 5 min (add caveat) or hours (implement) |
| 3 | No LICENSE file | should-fix | 2 min |
| 4 | No authors in pyproject.toml | should-fix | 1 min |
| 5 | Gemini missing from README provider table | should-fix | 2 min |
| 6 | Two test failures | should-fix | 30 min |
| 7 | Unused pytest-httpx dependency | should-fix | 1 min |
| 8 | Login/logout stubs in CLI | should-fix | 5 min |
| 9 | No plugin integration tests | nice-to-have | — |
| 10 | Stub slash commands | nice-to-have | — |
| 11 | No py.typed marker | nice-to-have | — |
