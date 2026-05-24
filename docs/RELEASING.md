# Releasing synthpanel

## Release flow (production PyPI)

Releases follow semver and are fully automated via GitHub Actions:

1. **Open a PR** against `main` with your changes.
2. **Add a semver label** to the PR: `semver:patch`, `semver:minor`, or `semver:major`.
   - Use `semver:skip` to merge without creating a release.
   - **Release PRs are enforced.** If the PR title starts with `chore(release):`
     or `release:`, `auto-tag.yml` fails loudly when no semver label is present
     instead of silently skipping. Add a label and re-run the workflow.
3. **Merge the PR.** On merge, `auto-tag.yml` runs:
   - Reads the semver label to determine the bump type.
   - Computes the next version from the latest `v*.*.*` tag.
   - Creates and pushes a new git tag (e.g. `v0.5.0`).
   - Creates a GitHub Release with auto-generated notes.
   - Triggers `publish.yml` which builds and publishes to [PyPI](https://pypi.org/project/synthpanel/).

### Manual publish

You can re-publish an existing tag via workflow dispatch:

1. Go to **Actions → Publish to PyPI → Run workflow**.
2. Enter the tag (e.g. `v0.4.0`). The tag must already exist in the repo.

## Post-release verification — keep the four "latest" surfaces aligned

A v1.5.3 dogfood found the public surfaces disagreeing about the current
version (PyPI said `1.5.3`, GitHub `releases/latest` said `v1.0.3`,
synthpanel.dev said `v1.4.0`). `auto-tag.yml` now creates the GitHub Release
and syncs every version artifact on each bump, but **verify all four surfaces
after every release** so a silent gap can't reopen (issue #524):

```bash
VER="1.5.3"   # the version you just released

# 1. PyPI — the canonical artifact.
curl -s https://pypi.org/pypi/synthpanel/json | jq -r '.info.version'   # → $VER

# 2. GitHub releases/latest — must equal the PyPI version.
gh api repos/DataViking-Tech/SynthPanel/releases/latest --jq '.tag_name' # → v$VER

# 3. Tags vs releases — every release tag should have a Release (no gap).
comm -23 \
  <(git tag --list 'v1.*' --sort=v:refname) \
  <(gh release list -R DataViking-Tech/SynthPanel -L 100 --json tagName --jq '.[].tagName' | sort -V)
# (empty output = aligned)

# 4. synthpanel.dev — the live landing hero must show the new version.
curl -s https://synthpanel.dev/ | grep -o 'v[0-9.]* — public beta'      # → v$VER — public beta
```

If `releases/latest` lags PyPI (e.g. older tags were cut before the GitHub
Release step existed), backfill the missing releases — mark only the newest as
latest:

```bash
gh release create vX.Y.Z -R DataViking-Tech/SynthPanel \
  --verify-tag --generate-notes --title "vX.Y.Z" --latest=false   # backfilled intermediate
gh release create vNEWEST -R DataViking-Tech/SynthPanel \
  --verify-tag --generate-notes --title "vNEWEST" --latest          # current latest
```

> **Social card.** `site/og-image.png` / `site/github-social-preview.png`
> embed the version too. `site/generate-og-image.py` reads it from
> `src/synth_panel/__version__.py` (no longer hardcoded); rerun
> `python3 site/generate-og-image.py` and commit the PNGs if the card text
> drifts from the released version.

## Dev builds (TestPyPI)

Every merge to `main` automatically publishes a dev build to [TestPyPI](https://test.pypi.org/project/synthpanel/):

- Version format: `{base_version}.dev{run_number}` (e.g. `0.4.0.dev42`).
- Workflow: `publish-test.yml`.
- No labels or manual steps required — it runs on every push to `main`.

### Install a dev build

```bash
# Latest dev build from TestPyPI
pip install -i https://test.pypi.org/simple/ synthpanel

# Specific dev version
pip install -i https://test.pypi.org/simple/ synthpanel==0.4.0.dev42
```

> **Note:** TestPyPI may not have all dependencies. If installation fails due to
> missing deps, install them from real PyPI first, then install synthpanel from
> TestPyPI:
>
> ```bash
> pip install httpx pyyaml
> pip install -i https://test.pypi.org/simple/ --no-deps synthpanel==0.4.0.dev42
> ```

## Install a release

```bash
# Latest stable release from PyPI
pip install synthpanel

# Specific version
pip install synthpanel==0.4.0
```

## GitHub environments

| Environment | Purpose | Trusted publisher workflow |
|-------------|---------|---------------------------|
| `pypi` | Production PyPI releases | `publish.yml` |
| `pypi-test` | Dev builds on TestPyPI | `publish-test.yml` |

Both environments and their trusted publishers are configured in the GitHub repo settings.
