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
