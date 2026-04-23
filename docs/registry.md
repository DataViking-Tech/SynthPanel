# Pack Registry

synthpanel ships with a small set of bundled persona packs, but most interesting
packs are written by the community. The **pack registry** is how you discover,
pull, and publish those packs without leaving the CLI.

The registry itself is a JSON index (`default.json`) hosted at
[`DataViking-Tech/synthpanel-registry`](https://github.com/DataViking-Tech/synthpanel-registry).
Anyone can add their pack to it via a pull request. synthpanel caches the index
locally for 24 hours and falls back to a stale copy when the network is
unreachable, so `pack import` keeps working offline.

## Importing a Remote Pack

`pack import` accepts three kinds of source:

| Source form | Example |
|---|---|
| `gh:user/repo[@ref][:path]` | `gh:dataviking-tech/example-pack` |
| `https://github.com/.../blob/...` | `https://github.com/dv/pk/blob/main/synthpanel-pack.yaml` |
| `https://raw.githubusercontent.com/...` | `https://raw.githubusercontent.com/dv/pk/main/synthpanel-pack.yaml` |

### `gh:` URIs

```bash
# Default ref (main), default path (synthpanel-pack.yaml)
synthpanel pack import gh:dataviking-tech/example-pack

# Pin to a tag or commit SHA
synthpanel pack import gh:dataviking-tech/example-pack@v0.2.0
synthpanel pack import gh:dataviking-tech/example-pack@3a7f1c0

# Non-default YAML path inside the repo
synthpanel pack import gh:dataviking-tech/example-pack:packs/contrarian.yaml

# Ref + path together
synthpanel pack import gh:dataviking-tech/example-pack@v0.2.0:packs/contrarian.yaml
```

When no `:path` is given, the resolver tries `synthpanel-pack.yaml` at the repo
root first. If that file is missing, it inspects the root via the GitHub
Contents API and — if and only if there is exactly one `*.yaml`/`*.yml` file
there — imports that one. Two or more root-level YAML files is a `ValueError`
asking you to pick one explicitly.

### Publishable filename

The publishable default is **`synthpanel-pack.yaml`** (namespaced so it will not
collide with unrelated `pack.yaml` files in the same repo). Authors who drop a
single file at the repo root should use exactly that name.

### HTTPS forms

Paste any GitHub `blob/` URL directly; synthpanel rewrites it to the
corresponding `raw.githubusercontent.com` URL and fetches it. Pre-rewritten
`raw.githubusercontent.com/...` URLs are accepted as-is. No magic for other
hosts — if the URL starts with `https://` but is not a GitHub address, the
import fails with a clear error.

## Verified vs Unverified

By default, `pack import` will **only import packs that are listed in the
registry**. This is a soft trust check: inclusion requires a PR to
`synthpanel-registry`, which gets basic sanity review before merge.

A pack is considered registered when the registry contains an entry whose
`repo` matches `user/repo` from your `gh:` source and whose `ref` matches the
ref you pulled. HTTP(S) URLs are always treated as unregistered (there is no
stable `user/repo@ref` to match on).

### `--unverified`

To import a pack that is not in the registry, pass `--unverified`:

```bash
synthpanel pack import gh:some-user/not-yet-submitted-pack --unverified
```

The import prints a one-time warning block with the source URL, the SHA-256
checksum of the YAML body as it was fetched, and the id synthpanel saved it
under. Keep that checksum if you care about pinning — there is no TOFU cache
yet; `--unverified` only gates the registry check.

If you pass `--unverified` for a source that **is** already in the registry,
the flag is ignored and you get a friendly `Note: ... --unverified flag
unnecessary.` message on stderr — the import proceeds normally.

## Pack IDs and Collisions

synthpanel needs a unique `pack_id` to save the pack under. It resolves the id
in this priority order:

1. `--id <value>` if you passed it explicitly
2. The `id:` field inside the pack YAML, if present
3. A slug derived from the source — the repo name for `gh:` URIs, or the path
   stem for raw URLs

Two collision cases are enforced before anything touches disk:

| Collision | Result |
|---|---|
| id matches a **bundled pack** (ships inside the synthpanel wheel) | Hard error — bundled ids are reserved. Re-run with `--id <new-id>` to import under a different name. |
| id matches an **existing user-saved pack** | Hard error. Re-run with `--force` to overwrite, or `--id <new-id>` to keep both. |

Example — collision with an existing user pack:

```
$ synthpanel pack import gh:alice/panel-pack
Error: pack id 'panel-pack' already exists as a user-saved pack.
  Re-run with --force to overwrite, or --id <new-id> for a new copy.
```

`--name` is independent of `--id`: it sets the human-readable display name and
defaults to the YAML `name:` field, falling back to the pack id.

## Cache and Offline Behavior

The registry cache lives at
`$SYNTH_PANEL_DATA_DIR/registry-cache.json` (default
`~/.synthpanel/registry-cache.json`). Fresh fetches use `If-None-Match` against
the cached `ETag`, so repeated calls are cheap.

| Env var | Effect |
|---|---|
| `SYNTHPANEL_REGISTRY_URL` | Override the registry URL (useful for forks, air-gapped setups, tests). |
| `SYNTHPANEL_REGISTRY_REFRESH=1` | Force a network fetch even if the cached copy is fresh. |
| `SYNTHPANEL_REGISTRY_OFFLINE=1` | Skip the network; use the cached copy, or an empty registry if none exists. |
| `SYNTH_PANEL_DATA_DIR` | Override the on-disk cache location. |

When the cache is older than 24 hours and the fetch fails, synthpanel keeps
using the stale copy — operations never block on the network. When there is
no cache at all and the fetch fails, you get an empty registry; registered-only
imports will then fail with the standard `not in the synthpanel registry`
error, and you can fall back to `--unverified` if you know what you are
importing.

## Authoring a Pack

Any valid persona pack YAML can be published. Minimum shape:

```yaml
# synthpanel-pack.yaml  (at the repo root — the default path)
name: Contrarian Stress Pack
version: "1"          # optional; defaults to "1" when omitted
description: >
  Five personas tuned to push back on pricing and feature claims.
personas:
  - name: Jamie
    age: 41
    occupation: Operations lead
    background: >
      12 years in logistics. Has seen three "AI-powered" tools fail to
      deliver. Defaults to skepticism until shown hard numbers.
    personality_traits: [skeptical, quantitative, blunt]
  - name: ...
```

Notes:

- `name` is required; `version` is optional and must be a string when present
  (a single YAML `1` without quotes will be rejected).
- `personas` must be a non-empty list; each persona needs at least `name`.
- `personality_traits` accepts either a list or a comma-separated string —
  both are normalized to lowercased list items at import time.
- If you include an `id:` at the top level, it is used as the default pack id
  on import (still overridable with `--id`).

See [`examples/personas.yaml`](../examples/personas.yaml) in this repo for a
complete working example.

### Filename convention

Put the pack at `synthpanel-pack.yaml` in the repo root if the repo is a
dedicated pack repo. If the repo contains multiple packs, put them under a
subdirectory and document the `gh:user/repo:packs/<name>.yaml` form in your
README — synthpanel will not guess across directories.

## Contributing a Pack to the Registry

The registry is a separate repository:
[`DataViking-Tech/synthpanel-registry`](https://github.com/DataViking-Tech/synthpanel-registry).

The contribution flow:

1. Publish your pack in its own GitHub repo (public). Tag a release if you
   want users to be able to pin to `@v…`.
2. Open a PR against `synthpanel-registry` adding a new entry to
   `default.json`. See that repo's
   [`CONTRIBUTING.md`](https://github.com/DataViking-Tech/synthpanel-registry/blob/main/CONTRIBUTING.md)
   for the entry schema and review bar.
3. Once merged, `synthpanel pack import gh:<you>/<repo>` works without
   `--unverified` for anyone whose cache refreshes — within 24h for most
   users, immediately with `SYNTHPANEL_REGISTRY_REFRESH=1`.

The registry review is intentionally light. It checks the entry is
well-formed, the repo is reachable, the YAML parses as a persona pack, and
the author field is filled in. It does not vouch for the personas themselves —
users are still responsible for reviewing a pack before trusting its output
in their own research.

## Troubleshooting

**`Error: pack not found at https://raw.githubusercontent.com/... (HTTP 404).`**
The repo, ref, or path does not exist — or the repo is private. `GITHUB_TOKEN`
auth for private repos is planned; for now, download the YAML and use the
local-file form of `pack import`.

**`Error: pack '<source>' is not in the synthpanel registry.`**
Either submit an entry to the registry (see above), or re-run with
`--unverified` if you are importing something you trust.

**`multiple yaml files at root of user/repo@ref: [...]`**
The fallback resolver only accepts a single root-level YAML. Either rename
your pack file to `synthpanel-pack.yaml`, or import with an explicit path:
`gh:user/repo:path/to/pack.yaml`.

**Stale data after you submitted a registry PR.**
Your local cache is up to 24h old. Force a refresh:
`SYNTHPANEL_REGISTRY_REFRESH=1 synthpanel pack import gh:you/your-pack`.
