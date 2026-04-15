# External Contributor Safety + CI/CD Threat-Model Audit

**Date:** 2026-04-15
**Repo:** `DataViking-Tech/SynthPanel` (now **public**)
**Auditor:** polecat `synthpanel/polecats/dag` (sp-273)
**Trigger:** Public-flip + active PyPI Trusted Publishing → real external attack surface.

---

## Top of page — Operator follow-ups (cannot fix from a polecat PR)

These require repo-admin / founder action. None are blocking, but (1) and (2) are
**HIGH priority** for a public repo with active publishing.

1. **HIGH — Enable secret scanning + push protection.** Currently disabled.
   For a public repo, this is table-stakes.
   ```
   Settings → Code security → Secret scanning: Enable
   Settings → Code security → Push protection: Enable
   ```
2. **HIGH — Enable Dependabot security updates + vulnerability alerts.** Currently disabled.
   ```
   Settings → Code security → Dependabot alerts: Enable
   Settings → Code security → Dependabot security updates: Enable
   ```
   (Renovate handles routine upgrades; Dependabot *security* updates is a
   separate, independent alerting path — enable both.)
3. **MEDIUM — Tighten PR ruleset on `main`.** Current ruleset has
   `required_approving_review_count: 0` and `require_code_owner_review: false`.
   Recommended: set approvals to **1**, enable CODEOWNERS enforcement.
   Without this, any human with write access (or Renovate auto-merge on a
   passing patch bump) can land a change with zero review.
4. **MEDIUM — Add deployment-branch-policy to `pypi` / `pypi-test` environments.**
   Both environments currently have `protection_rules: []` and
   `deployment_branch_policy: null` — i.e. any ref that reaches the job can
   release. Today this is mitigated by workflow triggers (`tag push` +
   `push:main`) plus branch protection, but defence-in-depth: restrict
   `pypi` to `v*.*.*` tags and `pypi-test` to the `main` branch.
5. **MEDIUM — Enable CodeQL + Dependency Review Action on PRs.** No code
   scanning today. Recommended: turn on CodeQL default setup (Actions +
   Python threat model) from Security → Code scanning; optionally add a
   `dependency-review-action` workflow that runs on `pull_request`.
6. **MEDIUM — Delete unused Cloudflare secrets.** `CLOUDFLARE_ACCOUNT_ID`
   and `CLOUDFLARE_API_TOKEN` are stored as repo secrets but are **not
   referenced by any workflow**. If left in place, a future workflow edit
   could inadvertently leak them. Either wire them to a real workflow or
   delete from Settings → Secrets.
7. **MEDIUM — Revisit Renovate `automerge: true` for patch updates.**
   Combined with `required_approving_review_count: 0`, a malicious upstream
   patch release could land with no human review. Options: remove auto-merge
   for patch, OR require ≥ 1 approval (see #3) which Renovate cannot
   self-provide.
8. **LOW — Verify PyPI Trusted Publisher binding matches post-rename repo.**
   Repo was renamed; confirm at pypi.org/manage/project/synthpanel/settings/publishing/
   that owner=`DataViking-Tech`, repo=`SynthPanel`, workflow=`publish.yml`,
   environment=`pypi`. Ditto test.pypi.org for `publish-test.yml` /
   `pypi-test`. Trusted Publishing fails closed if mismatched, so this is
   a correctness check, not a security hole.
9. **LOW — Enable `delete_branch_on_merge`.** Currently `false`. Reduces
   branch sprawl; not a security issue.
10. **LOW — Enable commit signature requirement in ruleset.** Not currently
    required. Optional hardening.

---

## Scope + method

Scanned every file under `.github/`, the four GitHub Actions workflows,
repo secrets, environment protections, branch ruleset, PR/issue templates,
`CODEOWNERS`, `SECURITY.md`, `CONTRIBUTING.md`, `pyproject.toml`, and
`renovate.json`. Remote state inspected via `gh api`.

## Workflow inventory

| File | Trigger(s) | Secrets / token scope | External-PR exposure |
|------|-----------|-----------------------|----------------------|
| `.github/workflows/ci.yml` | `push:main`, `pull_request:main` | `contents: read`, no env secrets | **Safe.** `pull_request` from forks ⇒ read-only `GITHUB_TOKEN`, no secrets. |
| `.github/workflows/publish.yml` | `workflow_call`, `push:tags:v*`, `workflow_dispatch` | `id-token: write` (PyPI OIDC) via `environment: pypi` | **Not reachable from forks.** Tag push requires write; dispatch requires write; `workflow_call` is invoked only by `auto-tag.yml` internally. |
| `.github/workflows/publish-test.yml` | `push:main` | `id-token: write` via `environment: pypi-test` | **Not reachable from forks.** Only fires after a merge to `main`. |
| `.github/workflows/auto-tag.yml` | `pull_request:closed:main` | `contents: write`, `pull-requests: read`, `checks: read` | **Safe for fork PRs.** `pull_request` from forks = read-only token; the `contents: write` requested in `permissions:` is silently downgraded by GitHub for fork-originated events. Tagging only succeeds for PRs from the same repo. |

### ✅ No `pull_request_target` anywhere

The single most dangerous pattern (running attacker PR head code with full
secrets and write token) is not present. Confirmed by grep across
`.github/workflows/`:

```
$ grep -nE 'pull_request_target|workflow_run' .github/workflows/*.yml
# no matches
```

### ✅ All third-party actions pinned to commit SHAs

`actions/checkout@de0fac2…`, `actions/setup-python@a309ff8b…`,
`actions/upload-artifact@043fb46d…`, `pypa/gh-action-pypi-publish@cef22109…`
— all pinned by full 40-char SHA with `# vX.Y.Z` comment. Renovate
`github-actions` group will bump SHAs on schedule (Monday AM).

---

## Section-by-section findings

### 1. `pull_request` vs `pull_request_target`

**Finding: PASS.** No workflow uses `pull_request_target` or `workflow_run`.
All PR-triggered workflows use `pull_request`, which runs with a restricted
`GITHUB_TOKEN` and no repository secrets when the PR originates from a
fork.

### 2. `GITHUB_TOKEN` permissions

**Finding: PASS with one note.**

| Workflow | Top-level `permissions:` | Job-level narrowing | Notes |
|----------|--------------------------|---------------------|-------|
| `ci.yml` | `contents: read` | — | Least-privilege ✓ |
| `publish.yml` | `contents: read`, `id-token: write` | `build-and-publish`: same | OIDC-only ✓ |
| `publish-test.yml` | — | `publish-dev`: `contents: read`, `id-token: write` | Missing top-level block; job-level is sufficient but inconsistent |
| `auto-tag.yml` | — | `auto-tag`: `contents: write`, `pull-requests: read`, `checks: read` | Job-level only; `contents: write` is required to push the tag and create the release. Fork PRs still get read-only token (GitHub enforced). |

**Recommendation (non-blocking):** add a top-level `permissions: { contents: read }`
block to `publish-test.yml` and `auto-tag.yml` for a clearer least-privilege
audit trail. Deferred to operator follow-up; not included in this PR because
`auto-tag.yml` needs write at the job level and the top-level minimum is
defensive-only, not a functional change.

### 3. Workflow injection via untrusted input

**Finding: PASS (with one hardening applied in this PR).**

Searched for dangerous interpolations of attacker-controlled strings
(`github.event.pull_request.title` / `body` / `head.ref`, `github.head_ref`,
`github.event.issue.*`, `github.event.comment.body`) in `run:` blocks.

One hit: `auto-tag.yml:144` uses `${{ github.event.pull_request.title }}`
— correctly passed via an `env:` block and referenced as `"$PR_TITLE"` in
`printf`, which is the safe pattern. ✓

**Hardened in this PR:** `publish.yml` previously interpolated
`${{ inputs.tag }}` directly into a shell `run:` block. Although
`workflow_dispatch`/`workflow_call` are not reachable by external
contributors, moved to `env: INPUT_TAG` + `$INPUT_TAG` as defence in
depth. Regex validation still occurs after the assignment.

### 4. PyPI Trusted Publisher binding scope

**Finding: FLAG — requires operator verification.**

- `publish.yml` declares `environment: pypi` and uses
  `pypa/gh-action-pypi-publish` with OIDC (no API token).
- `publish-test.yml` declares `environment: pypi-test` and publishes to
  `https://test.pypi.org/legacy/`.
- Both environments exist on the repo (verified via `gh api`).
- **Cannot verify from a polecat:** whether the PyPI / TestPyPI side of
  the binding correctly references the post-rename repo (`SynthPanel`, not
  `synth-panel`) and the right workflow filenames. **Operator: confirm
  via pypi.org + test.pypi.org management console.** (Operator follow-up #8 above.)

### 5. Branch protection on `main`

**Finding: PARTIAL — see operator follow-up #3.**

Branch protection is implemented as a **ruleset** (ID 14896580), not the
legacy branch-protection API. Rules in effect:

| Rule | Value | Verdict |
|------|-------|---------|
| `deletion` | enabled | ✓ |
| `non_fast_forward` | enabled | ✓ |
| `required_linear_history` | enabled | ✓ |
| `required_status_checks` | **8 checks** (`lint`, `typecheck`, `security`, `coverage`, `test (3.10/3.11/3.12/3.13)`) | ✓ |
| `strict_required_status_checks_policy` | `false` | — (acceptable) |
| `pull_request.required_approving_review_count` | **0** | ✗ **WEAK** |
| `pull_request.require_code_owner_review` | `false` | ✗ **WEAK** |
| `pull_request.required_review_thread_resolution` | `true` | ✓ |
| `pull_request.dismiss_stale_reviews_on_push` | `true` | ✓ |
| `pull_request.allowed_merge_methods` | `merge`, `squash`, `rebase` | ✓ (linear-history rule blocks non-FF merges anyway) |
| `bypass_actors` | `[]` | ✓ |
| `current_user_can_bypass` | `never` | ✓ |

The ruleset blocks force-pushes, deletions, and non-linear history, and
requires all 8 CI checks to pass. What's missing:

1. **Zero required approvals.** Any maintainer can merge their own PR
   unreviewed; Renovate can auto-merge its own patch PRs.
2. **CODEOWNERS not enforced.** The file exists but the rule doesn't
   reference it, so required reviews from listed owners aren't guaranteed.

Cannot fix from a polecat (ruleset edit requires admin). See operator #3.

### 6. CODEOWNERS coverage

**Finding: PARTIAL.**

`.github/CODEOWNERS` is present:

```
* @DataViking-Tech/synthpanel
renovate.json @DataViking-Tech/synthpanel
```

Coverage check:
- `*` covers everything (workflows, `pyproject.toml`, `src/`, secrets-touching code). ✓
- `renovate.json` is explicitly scoped — redundant given `*` but harmless.
- **Team validity** (whether `@DataViking-Tech/synthpanel` exists and has
  write access): cannot verify from a polecat's token. Operator should
  confirm from org settings. (Folded into follow-up #3.)
- **Not enforced** by branch ruleset (see §5 above).

### 7. Dependency safety

**Finding: PASS.**

- **Actions:** all pinned to commit SHAs with version comments; Renovate's
  `github-actions` group bumps them weekly (Monday AM).
- **`pyproject.toml`:** runtime deps are two well-known packages with
  lower-bound ranges (`httpx>=0.27`, `pyyaml>=6.0`). No bare `*`, no
  URL-based deps, no git deps. Optional `mcp>=1.0` for the MCP extra. ✓
- **Dev deps:** pytest, ruff, mypy, pip-audit — all reputable, lower-bound
  ranges. ✓
- **No Dependabot config.** Renovate covers routine upgrades. Dependabot
  *security* updates is a separate alerting path and should be enabled
  regardless (see operator #2).
- **`pip-audit`** runs as a required CI check (`security` job in `ci.yml`). ✓

### 8. Secret hygiene

**Finding: PARTIAL.**

Repo secrets (`gh secret list`):
- `CLOUDFLARE_ACCOUNT_ID` (added 2026-04-04)
- `CLOUDFLARE_API_TOKEN` (added 2026-04-04)

Grep across `.github/workflows/*.yml` shows **no references** to either
secret. They appear to be leftovers from an earlier deployment target.
Operator follow-up #6.

No hardcoded test API keys found (`grep -E '(sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|xoxb-[0-9]{10,})'` across repo — 0 hits).

Publish workflows use PyPI Trusted Publishing (OIDC), no static tokens. ✓

### 9. CodeQL / security scanning

**Finding: FAIL — operator follow-up #5.**

- CodeQL default setup: `state: not-configured`.
- No `.github/workflows/codeql.yml` present.
- No `dependency-review-action` usage on PRs.
- `SECURITY.md` is present and routes reports to `security@dataviking.tech`. ✓

Not fixed in this PR to avoid coordinating a new required check. Operator
should enable CodeQL default setup from the Security tab (no workflow
file needed; GitHub manages it).

### 10. Issue + PR templates

**Finding: PASS.**

- `.github/ISSUE_TEMPLATE/` contains `bug_report.yml`, `feature_request.yml`,
  `adapter_proposal.yml`, `config.yml`. ✓
- `.github/pull_request_template.md` covers intent, test plan, semver
  choice, and a contributor checklist (tests, ruff, mypy, CHANGELOG,
  DCO sign-off). ✓
- `CONTRIBUTING.md` covers setup, tests, linting, MCP server, adapter
  authoring, submission flow, DCO. ✓
- `CODE_OF_CONDUCT.md` present. ✓
- `SECURITY.md` present with responsible-disclosure channel. ✓

### 11. Auto-merge / Renovate interaction

**Finding: MEDIUM RISK — operator follow-ups #3 + #7.**

- `renovate.json` sets `automerge: true` for `patch` update types.
- Repo setting `allow_auto_merge: true`.
- Ruleset on `main` requires 0 approvals.
- Net effect: a passing-CI Renovate patch PR lands with zero human review.

Small runtime dependency surface (2 packages) limits blast radius, but
best practice is to require at least 1 human approval before any merge
(see #3), which also prevents Renovate auto-merge from bypassing review.

### 12. Verdict

**Status: NEEDS-FIX** (weighted by repo being public + actively publishing
to PyPI).

**Top 3 risks (ranked):**

1. **Secret scanning + push protection are OFF.** A maintainer accidentally
   committing a key (API token, personal PAT) would not be caught at push
   time. High probability × high impact for a public repo.
2. **Zero-approval merges + Renovate auto-merge.** A malicious upstream
   patch release or a careless maintainer self-merge can land without any
   second pair of eyes, and the chain flows straight to
   `publish-test.yml` → TestPyPI on every `main` push.
3. **Unused Cloudflare secrets in the repo.** They're currently dormant
   (no workflow reads them), but any future workflow edit — intentional or
   via a compromised Action — could exfiltrate them.

**Top 3 immediate fixes (in scope for *this* PR — already applied):**

1. Moved `${{ inputs.tag }}` interpolation in `publish.yml` behind an
   `env:` block (`INPUT_TAG` → `$INPUT_TAG`). Defense-in-depth against any
   future removal of the post-assignment regex validation.
2. Added `persist-credentials: false` to every `actions/checkout` in
   `ci.yml` and to the checkouts in `publish.yml` + `publish-test.yml`.
   This prevents the `GITHUB_TOKEN` from being written to disk, so it
   cannot be read by any later step (including a future malicious
   dependency install). Not applied to `auto-tag.yml` because that
   workflow needs persisted credentials to `git push` the tag and
   `gh release create`.
3. Produced this audit document itself — the deliverable and operator
   checklist that turns the remaining findings into trackable work.

**Operator follow-ups** — see the numbered list at the top of this
document. None are blocked by this PR; all should be tracked as
individual beads.

---

## Appendix A — Commands used for remote inspection

```bash
gh repo view --json name,nameWithOwner,visibility,url,defaultBranchRef
gh api repos/DataViking-Tech/SynthPanel/rulesets
gh api repos/DataViking-Tech/SynthPanel/rulesets/14896580
gh api repos/DataViking-Tech/SynthPanel/branches/main/protection   # returns 404: uses ruleset, not legacy protection
gh secret list --repo DataViking-Tech/SynthPanel
gh api repos/DataViking-Tech/SynthPanel/environments
gh api repos/DataViking-Tech/SynthPanel/code-scanning/default-setup
gh api repos/DataViking-Tech/SynthPanel/vulnerability-alerts
gh api repos/DataViking-Tech/SynthPanel/automated-security-fixes
gh api repos/DataViking-Tech/SynthPanel \
  --jq '{allow_auto_merge,allow_merge_commit,allow_squash_merge,allow_rebase_merge,
         delete_branch_on_merge,allow_update_branch,
         squash_merge_commit_title,squash_merge_commit_message,
         has_vulnerability_alerts:.security_and_analysis}'
```

## Appendix B — Diff of in-PR hardening

```
.github/workflows/ci.yml          | 4 ++++  (4× persist-credentials: false)
.github/workflows/publish.yml     | 5 +++--  (env INPUT_TAG; persist-credentials: false)
.github/workflows/publish-test.yml| 2 ++  (persist-credentials: false)
```

No functional behaviour changed. `auto-tag.yml` intentionally left as-is;
it needs the persisted token for `git push` + `gh release create`.

---

*End of audit. File this as `sp-273` deliverable. Operator follow-ups
should be opened as individual beads referencing the numbered list at
the top.*
