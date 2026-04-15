# Pre-public-release audit — 2026-04-15

Audit for the synthpanel (PyPI: `synthpanel`; GitHub: currently `DataViking-Tech/synth-panel`) pre-public-flip. Bead: `sp-9m9`.

## Verdict: **READY-TO-FLIP** (after the in-place fixes landed in this PR)

No blockers remain after the hygiene fixes in this PR. Two recommended follow-ups listed below are non-blocking.

## Summary

- Sections audited: **10**
- **Pass**: 8 (secrets, PII, agent-internal files, tests/fixtures, stability, examples/DX, governance, repo metadata)
- **Fix-filed in this PR**: 2 (documentation hygiene, supply-chain/metadata URL hygiene)
- **Blockers (must-fix before public)**: 0

## Section reports

### 1. Secrets & credentials — **PASS**

- `gitleaks detect` (v8.30.1) across **249 commits / 1.90 MB** of content: **no leaks found**.
- `git log --all --diff-filter=A -- "*.env*"`: only `.env.example` was ever added, and its first commit body contains only empty key names (`ANTHROPIC_API_KEY=`, etc.) — no secrets ever checked in.
- No `*.p12`, `*.pem`, `*.key`, `*.pfx` files in history.
- Manual grep for `sk-`, `pk_`, `Bearer `, `ghp_`, `xoxb-`, `BEGIN PRIVATE KEY`, `ssh-rsa AAAA…` across the tree: every hit is legitimate (test dummy keys like `"sk-test"`, doc snippets like `sk-...`, or provider-code `f"Bearer {self._api_key}"`).

**Conclusion:** no real secrets present or ever present in the public history.

### 2. PII / internal references — **PASS**

- `dataviking.tech` appears in 3 intentional places: `security@dataviking.tech` (SECURITY.md), `conduct@dataviking.tech` (CODE_OF_CONDUCT.md), `openclaw@dataviking.tech` (pyproject.toml `[project].authors`). All are legitimate contact points for a public repo.
- No internal infra hostnames, tailscale addresses, RFC1918 addresses, Slack team/channel IDs (T0XXX/CXXX), or internal URLs found.
- Synthpanel's own bead IDs (`sp-*`) are nominal to ship — they're public issue IDs once the repo flips.

### 3. Agent-internal files — **PASS** (after fix)

Tracked files with any Gas-Town / polecat flavor before this PR:

| Path | Disposition |
|---|---|
| `CLAUDE.md` | **Keep** — it's a generic project guide for AI coding assistants (Claude Code, Cursor, etc.), no internal/agent references. Publicly useful. |
| `.claude-plugin/plugin.json` | **Keep** — Claude Code plugin manifest, intentionally shipped. |
| `gastown-rig-settings.example.json` | **REMOVED** in this PR — references internal "Gas Town rig" system the public has no context for. |
| `CODEOWNERS` (root) | **REMOVED** in this PR — conflicted with `.github/CODEOWNERS` (root said `@DataViking-Tech/core`, `.github/` says `@DataViking-Tech/synthpanel`; `.github/` takes precedence and is the one that actually matches the org). |
| `src/synth_panel/cli/commands.py:915` | **Keep** — single code-comment reference to "refinery" as a generic automation example. Not blocking; could be generalized in a future pass. |

Untracked (already in `.gitignore` before this PR): `.claude/`, `.beads/`, `.runtime/`, `CLAUDE.local.md` (the last one added to `.gitignore` in this PR defensively — it's a polecat-only file that happens to be untracked currently).

### 4. Documentation review — **PASS** (after fix)

- **README.md** (463 lines): strong public-facing intro. Clear "what / why / quick-start" in the first 70 lines; runnable `pip install synthpanel` command in the first code fence. Covers v3 branching, providers, MCP, ensemble blending, methodology caveats. Only fix: one `git+https://…/synthpanel.git` URL updated to `synth-panel.git` (line 34).
- **SPEC.md** (794 lines): functional spec — appropriate for public, gives external contributors a clean-room reference.
- **CHANGELOG.md** (was 81 lines, now 131): had a real hole — entries jumped from 0.4.1 straight to 0.8.0 with no 0.5 / 0.6 / 0.7 entries, despite `v0.5.0`, `v0.6.0`, `v0.7.0..v0.7.4`, `v0.8.0` tags all existing. Backfilled minimal entries for 0.5.0, 0.6.0, 0.7.0, and a 0.7.4 patch-series note; added the missing footer link refs. Content cross-references the README Versions table rather than re-narrating release notes.
- **LICENSE**: MIT, © 2026 DataViking Tech. Valid.
- **SECURITY.md**: present, reports to `security@dataviking.tech`, 48h ack / 7d fix target. Good.
- **CODE_OF_CONDUCT.md**: Contributor Covenant 2.1 adoption with enforcement contact. Good.
- **CONTRIBUTING.md**: dev setup, tests, lint, release process, DCO sign-off policy. Updated the `git clone` URL to `synth-panel`.
- **.github/ISSUE_TEMPLATE/**: `bug_report.yml`, `feature_request.yml`, `adapter_proposal.yml`, `config.yml` present. URLs in `config.yml` pointed at the wrong repo name (`synthpanel` vs actual `synth-panel`) — fixed in this PR.
- **.github/pull_request_template.md**: present, semver-label-aware, DCO-aware, adapter section. Good.
- **docs/stability.md**: describes pre-1.0 breakage policy + enumerates public surface. Updated in this PR to add `synth_panel.cost.lookup_pricing_by_provider` (added in 0.8.0) and to reconcile "synth-panel" → "synthpanel" naming for consistency with the package name.
- **docs/mcp.md**, **docs/adapter-guide.md**, **docs/RELEASING.md**: present, accurate, publicly appropriate.

### 5. Governance & supply chain — **PASS** (after fix)

- **`.github/CODEOWNERS`**: `* @DataViking-Tech/synthpanel`. Valid.
- **pyproject.toml `[project]`**: `name`, `description`, `license` (MIT), `authors`, `readme`, `requires-python` (>=3.10), `classifiers`, `keywords` — all present and appropriate. Optional deps (`mcp`, `dev`) declared. Entry point `synthpanel = synth_panel.main:main`. URLs updated in this PR from `synthpanel` → `synth-panel`.
- **CI** (`.github/workflows/ci.yml`): `lint` (ruff check + format), `typecheck` (mypy), `security` (pip-audit), `test` (pytest matrix on Python 3.10–3.14, experimental=true on 3.14, 80% coverage floor). Actions pinned by SHA. Good.
- **Publish** (`.github/workflows/publish.yml` + `publish-test.yml` + `auto-tag.yml`): trusted publishing to PyPI via semver labels on merged PRs.
- **Dependency management**: `renovate.json` present — auto-merge patch, group minor, group GitHub Actions, weekly cadence, `prConcurrentLimit: 5`. Renovate is the counterpart to Dependabot; both would be redundant. No Dependabot config needed.
- **DCO**: enforced via `CONTRIBUTING.md` ("All contributions require a DCO sign-off … `git commit -s`").
- **Pre-commit**: `.pre-commit-config.yaml` present.
- **Branch protection rules on main**: not verified in this audit (requires live GitHub API; network unavailable from this worktree at audit time). Operator should verify: *require PR + 1 approval + all 4 CI checks (lint, typecheck, security, test) passing before merge* before flipping public. Flagged as a **post-flip operator checklist item** below.

### 6. Test fixtures & sample data — **PASS**

- `tests/` directory: 33 Python test files, no `*.json` / `*.csv` / `*.yaml` fixture files — fixtures are embedded Python string literals.
- Spot-checked `tests/test_acceptance.py`'s `_PERSONAS_YAML`: entirely synthetic personas ("Skeptical CTO", "Enthusiastic Intern", "Pragmatic PM") with generic backgrounds. No real human PII.
- No committed real survey data, real demographic snapshots, or real LLM responses.

### 7. API stability claims — **PASS** (after fix)

- `docs/stability.md` correctly frames synthpanel as pre-1.0 with minor-bumps-may-break, patch-bumps-never-break.
- Public surface enumerated: `LLMProvider` base, adapter contract types, CLI commands, MCP tool signatures, YAML instrument / persona formats.
- **Added in this PR:** `synth_panel.cost.lookup_pricing_by_provider(provider_string)` — the public helper added in 0.8.0 (`sp-027`) — was missing from the public-surface list. Now enumerated.
- The 0.8.0 CHANGELOG entry already documents the addition correctly.

### 8. Examples & developer experience — **PASS**

- `examples/` ships a persona pack plus one instrument per format (v1 flat, v2 linear, v3 branching ×2), with a `README.md` that indexes them and gives copy-pasteable run commands.
- `examples/instruments/` keeps three legacy standalone instruments for back-compat.
- **Fresh-venv smoke test** (executed during audit):
  - `python3.14 -m venv /tmp/…`
  - `pip install .` → build succeeded, wheel installed, 922/922 unit tests would pass.
  - `synthpanel --help` → correct help output with all subcommands.
  - `synthpanel instruments list` → all 8 bundled packs listed with descriptions.
  - `synthpanel --version` → `synthpanel 0.8.0`.
- Quickstart in README works for a fresh user with just `pip install synthpanel` + an API key.

### 9. Repo metadata (pre-rename snapshot)

- Current remote: `https://github.com/DataViking-Tech/synth-panel.git` (hyphenated).
- pyproject + all doc URLs now point to `synth-panel` (hyphenated, current).
- After operator executes the `synth-panel` → `synthpanel` rename, a one-line find/replace PR can swap these; GitHub auto-redirects old URLs in the interim so nothing breaks.
- GitHub description / topics / homepage URL not verified from this worktree (network unavailable to `api.github.com` at audit time); flagged in the operator checklist.

### 10. Quality gates — **PASS**

Ran against the head of this PR:

- `ruff check src/ tests/` → all checks passed
- `ruff format --check src/ tests/` → 87 files already formatted
- `mypy src/synth_panel/` → no issues found in 54 source files
- `pytest tests/ --ignore=tests/test_acceptance.py` → **922 passed**, total coverage **86.87%** (floor is 80%)

## Required actions before flip

**None.** All hygiene items are fixed in this PR.

## Recommended actions (non-blocking)

1. **Verify branch protection on `main`** before flip: require PR + 1 approval + all 4 CI checks (lint, typecheck, security, test) green. The merge queue is already pr-mode per the Gas Town rig config, but when the repo is public the UI-level protections should be explicit.
2. **Generalize the `refinery` comment** in `src/synth_panel/cli/commands.py:915` (*"automation (CI, refinery, wrapper scripts) can detect silent-failure scenarios"*) on a future non-urgent pass. Not publicly confusing, just slightly internal-flavored. Filed as a future-cleanup candidate; not worth blocking the flip.

## Post-flip checklist (operator)

1. GitHub Settings → rename `synth-panel` → `synthpanel` (cosmetic; PyPI name already correct).
2. Settings → Visibility → Public.
3. PR with version bump `0.8.0` → `0.9.0` + CHANGELOG entry, `semver:minor` label.
4. Verify `Auto Semver Tag` → `Publish to PyPI` fires cleanly on merge.
5. Update synthbench `pyproject.toml` to pin `synthpanel>=0.9.0`.
6. Add public-launch announcement (blog / Twitter / HN Show HN).
7. Drop-in find/replace of `DataViking-Tech/synth-panel` → `DataViking-Tech/synthpanel` across README/CHANGELOG/pyproject/CONTRIBUTING/ISSUE_TEMPLATE/config.yml once the rename is executed (GitHub redirects mean this is hygiene, not breakage).
8. Fill in GitHub repo **description** (suggested: *"Run synthetic focus groups and user research panels using AI personas. CLI tool, Python library, any LLM."* — matches pyproject `description`) and **topics** (suggested: `llm`, `synthetic-respondents`, `persona-simulation`, `user-research`, `focus-group`, `mcp-server`, `llm-evaluation`).
9. Confirm branch protection rules on main are on.
