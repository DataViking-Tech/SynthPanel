# Python Ecosystem Content Distribution — Landscape Audit

Objective scan of 11 tools' patterns for distributing user-authored content.

## 1. pre-commit (Git-based with manifest)

**Model:** Decentralized; hook providers create Git repos with `.pre-commit-hooks.yaml` manifest. Users reference via Git URLs pinned to tags. No central registry.

**Workflow:** Author creates Git repo + manifest + language-specific install → tags release → users reference by URL + rev.

**Review:** None. Repositories are author-maintained. Official `pre-commit-hooks` repo is reference/trusted implementation only.

**Upgrades:** Version pinning via tags (`rev: v6.0.0`). Breaking changes = author's responsibility via release notes.

**Failures:** Abandoned repos discoverable but no central deprecation flag. No namespace collision (Git URLs are namespaces). No moderation.

## 2. cookiecutter (GitHub-topic discovery)

**Model:** GitHub search for `cookiecutter` topic. Install via `cookiecutter gh:user/repo` or local paths.

**Workflow:** Author creates repo + `cookiecutter.json` + publishes + tags repo with topic.

**Review:** None. Visibility relies on GitHub search ranking + community sharing.

**Upgrades:** Git tags. Templates typically used once per project; fork/customize is common.

**Failures:** Abandoned templates clutter search indefinitely. No namespace isolation (`cookiecutter-django` could be anyone). Community reputation only differentiator.

## 3. dbt Packages (Hub registry)

**Model:** Centralized `hub.getdbt.com` recommended; fallback to Git URL or tarball. Specified in `packages.yml` with semver.

**Workflow:** Author creates package with `dbt_project.yml` + publishes to Git → submits to Hub (submission process *undocumented*) → users add to `packages.yml` + `dbt deps`.

**Review:** Undocumented. dbt Labs hosts Hub "as courtesy" with caveat: "does not certify integrity, operability, effectiveness, or security." No published criteria.

**Upgrades:** Semver pinning. `dbt deps` resolves. Breaking changes are author's responsibility.

**Failures:** Opaque submission creates friction. No documented security review. Git fallback fragments ecosystem visibility. Packages exist in two places (Hub + Git).

## 4. Jupyter Extensions (fragmented)

**Model:** `jupyter-contrib-extensions` consolidated repo + PyPI packages following `jupyter_*` naming. Install via pip or `jupyter nbextensions` commands.

**Workflow:** Contribute to consolidated repo OR publish to PyPI. Extensions register via Python entry points.

**Review:** Consolidated repo: GitHub PR + community review. Standalone PyPI: no review.

**Upgrades:** PyPI versioning + pip pinning. Entry points auto-discover; upgrades transparent.

**Failures:** Fragmented discovery (no central like dbt). Authors choose consolidated-visibility vs independent-autonomy. PyPI namespace collision possible. Abandoned extensions in consolidated repo create debt; removal disruptive.

## 5. Home Assistant + HACS (dual)

**Model:** (1) Official integrations bundled in `home-assistant/core`. (2) Community via HACS (Home Assistant Community Store), aggregating repos from curated `hacs/default` list.

**Workflow:** Official = contribute to core with rigorous review. Community = repo with `manifest.json` + PR to `hacs/default` to register URL.

**Review:** Official = intensive core-team review. Community = PR to register. Manifest enforces consistent format. HACS sandboxes + version-pins.

**Upgrades:** HACS tracks GitHub releases + semver. Users pin or auto-update within major. Compatibility flags prevent broken installs.

**Failures:** `hacs/default` list is centrally maintained, bottlenecks registration. Abandoned integrations persist unless removed. Rejected core contributions fork and maintain independently → fragmentation. Security review burden on HACS maintainers.

## 6. OpenSearch plugins (undocumented)

**Model:** Undocumented. Plugins as binary artifacts (JAR/JS bundles). Install via `opensearch-plugin install` with URL or local path. No central registry documented.

**Workflow:** Inferred — build → host on URL → users install via plugin CLI.

**Review:** None documented.

**Upgrades:** Plugin versioning = author's responsibility. No centralized version mgmt documented.

**Failures:** No registry → discoverability problem. No vetting documented. Users must know plugin URL or find via informal channels. Security is user responsibility.

## 7. Sphinx themes/extensions (PyPI namespace)

**Model:** PyPI packages following `sphinx-*` naming. `pip install sphinx-theme-foo`. Entry points auto-register at import time.

**Workflow:** Python package with Sphinx entry points → publish to PyPI → pip install → auto-register.

**Review:** PyPI's publish-and-pray. No Sphinx-specific review.

**Upgrades:** PyPI semver + pip pinning. Namespace packages (PEP 420) available but not widely adopted.

**Failures:** PyPI flat namespace enables typosquatting. No central curated registry. Abandoned projects persist. Entry-point conflicts resolve first-in-wins non-deterministically.

## 8. pytest plugins (entry-point auto-discovery)

**Model:** PyPI packages with `pytest11` entry point. pytest auto-discovers installed plugins at startup.

**Workflow:** Python package + `pytest11` entry point → PyPI → pip install → auto-load.

**Review:** PyPI only. No pytest-specific review. Docs maintain informational list of 1300+ known plugins.

**Upgrades:** PyPI semver + pip pinning. Auto-discovery loads new versions on upgrade unless pinned.

**Failures:** PyPI namespace collision → typosquatting (`pytest-super-plugin-xyz` vs `pytest-super-pluginxyz`). Entry-point name collisions non-deterministic (first wins). Auto-discovery masks breaking changes. High supply-chain attack surface.

## 9. conda-forge (community channel)

**Model:** Centralized GitHub-based feedstock repos. Authors contribute to `conda-forge/staged-recipes` (new) or dedicated feedstock (updates). CI builds + publishes to conda-forge channel.

**Workflow:** Recipe PR → community maintainer review → automated CI builds → publish. Feedstocks owned by maintainers.

**Review:** GitHub PR review by community + automated CI. Requires proper metadata, license compliance, dependency resolution, multi-platform builds. Transparent; no formal security review.

**Upgrades:** Semver with recipe pinning. Conda-forge bots automate version bumps. CI detects breaking changes. `conda lock` for reproducibility.

**Failures:** PR review can bottleneck when maintainers overloaded. Unresponsive feedstock maintainers orphan packages. `staged-recipes` acts as new-package gatekeeper reducing squatting. No formal appeals governance.

## 10. Jenkins plugins (centralized registry)

**Model:** Central `plugins.jenkins.io` registry + GitHub hosting. JAR distributions queried via Jenkins Update Center UI.

**Workflow:** Develop plugin → submit to registry (process undocumented publicly) → indexed in Update Center → users install via admin UI.

**Review:** Undocumented. 2000+ plugins suggest a pathway exists but is not publicly documented. Registry categories imply some curation.

**Upgrades:** Semver + compatibility metadata. Users pin or auto-update.

**Failures:** Opaque submission + governance. Abandoned plugins persist indefinitely. No enforced security standards or code-review transparency. Unclear vulnerability disclosure.

## 11. PyPI (namespace baseline)

**Model:** Flat namespace, first-come-first-claimed. Case-insensitive; `-` and `_` normalize the same (collision edge cases). World-discoverable unless yanked/archived.

**Workflow:** `setup.py`/`pyproject.toml` → twine upload → users `pip install`.

**Review:** None. Publish-and-pray. 2FA required for uploads. Email verification for registration. Compromised accounts can upload malicious versions; token revocation + recovery exist.

**Upgrades:** Semver convention (not enforced). Users pin. "Yanked" mechanism marks versions broken without removing. No automated security updates.

**Failures:** Typosquatting documented (`texst` vs `test`). Character-normalization collisions (`my_package` == `my-package`). Abandoned projects persist forever. Supply-chain attacks undetected without external tooling. Flat namespace has no scoping like npm's `@user/package`.

## Pattern Clusters & Failure-Mode Lessons

### Centralization spectrum

- **Decentralized** (pre-commit, cookiecutter): zero infrastructure, maximum discovery friction, author owns everything
- **Centralized** (dbt Hub, HACS, conda-forge): curated registry, submission bottleneck, governance-critical; transparency varies (conda-forge good, dbt Hub poor)
- **Hybrid** (pytest, Sphinx, Jupyter): PyPI infrastructure + entry-point auto-discovery for frictionless installation, inherits PyPI's flat-namespace vulnerabilities

### Governance failure patterns

- **Undocumented processes** (OpenSearch, Jenkins, dbt Hub): opacity creates friction and maintainer burnout
- **Abandoned package persistence**: all models allow persistence without mechanism for cleanup (1300+ pytest plugins, many unmaintained)
- **Moderation scaling**: community-driven (conda-forge) scales better than single-maintainer (dbt Hub) with clear governance + conflict resolution

### Namespace & supply chain

- **Flat namespaces** (PyPI, Sphinx): typosquatting + character-normalization collisions
- **Scoped namespaces** (npm model, not standard in Python): per-user safety; PEP 420 partial solution not widely adopted
- **Entry-point auto-discovery**: reduces user config, masks plugin load-order non-determinism, makes security audits harder

### Security & trust gaps

- **No universal code review**: only Home Assistant (official) + conda-forge (transparent PR) document security-focused review. dbt/pytest/Sphinx/Jupyter/OpenSearch delegate to authors or omit.
- **Dependency confidence**: PyPI doesn't verify package contents at publish. Transitively trusted; compromised upstreams propagate silently.
- **Breaking-change communication**: most systems lack structured compatibility metadata. Conda-forge's bot partially mitigates.

### Lessons for plugin-registry design

1. **Publish-to-pair (pre-commit, cookiecutter)** — minimal infra, fragments discovery; suitable for small specialized ecosystems.
2. **Centralized manifest registry + distributed hosting (HACS)** — balances autonomy and curation. Requires clear manifest schema and registration process.
3. **Entry-point auto-discovery (pytest, Sphinx)** — reduces friction, masks safety issues. Pair with explicit allow-listing or plugin-load logging.
4. **Transparent community governance (conda-forge)** — scales better than opaque processes. Requires documented review criteria, appeals, moderation tooling.
5. **Avoid PyPI flat namespace for plugin namespacing.** Use explicit scoping (prefix convention) or PEP 420 namespace packages with documented authoring guidelines.
6. **Document submission, review, maintenance explicitly.** Undocumented processes invite governance failures. Transparency builds community trust even with strict processes.
