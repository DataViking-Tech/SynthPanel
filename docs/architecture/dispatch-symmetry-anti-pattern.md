
# Dispatch-Symmetry Anti-Pattern

**Status:** active

**Authors:** midgard mayor (drafted), jotunheim mayor (PR via synthpanel), yggdrasil mayor (cross-coverage)

**Empirical anchor:** 7 instances cataloged 2026-05-09 / 2026-05-10 across boardroom, dataviking-infra, synthpanel, traitprint_cloud, dataviking-site (see Inventory below).

## TL;DR

When a feature is added at one entrypoint of a multi-entrypoint system, it is rarely propagated to every other entrypoint that should expose the same behavior. The result is **silent degradation**: the feature works on the path it was tested on, and silently does the wrong thing on every other path. Loud errors don't trigger; users see attachments dropped, rounds skipped, headers missing, attachments fetched-but-empty. The catch happens manually, after the broken path is in production.

The fix is mechanical: **wire the feature everywhere AND add a CI test that audits dispatch-path symmetry**. The harder discipline is checking for this smell at design time, before each new feature ships.

## What it looks like

You have N dispatch paths exposing similar surface area:

- Multiple HTTP route handlers in a single Worker
- Multiple repos in an org all serving user content
- Multiple instrument versions (v1, v2, v3) parsed by the same engine
- A library entry point and an MCP server entry point and a CLI entry point
- Same model accessible via direct API and via OpenRouter

A new feature lands on **one** of these. It works. Tests pass on that path. It ships.

The other paths silently behave differently:

- Don't apply the feature → users get the old behavior with no error
- Apply the feature partially → users get a degraded version
- Apply the feature with a stale shape → users get a corrupted result

No exception is raised. No 500. No log line saying "feature X disabled here." Just a quieter, wronger output.

## Inventory (7 instances, 2026-05-09 → 2026-05-10)

### Cache-Control across boardroom Worker routes (bo-sq2)

`/assets/catalyst.css` set `Cache-Control: public, max-age=31536000, immutable`. `/sessions` (HTML) and `/health` (JSON) sent NO `Cache-Control` at all. Browsers + CF edge defaulted to heuristic caching → updated HTML invisible to users after redeploys. Caught by the operator visiting the site and reporting "nothing changed."

### Cache-Control across DV properties (dvi-25f)

Same pattern, repo-wide: boardroom Worker, mtg-api Worker, dataviking-site (CF Pages), mtg-frontend (CF Pages), synthpanel.dev (CF Pages) — every one had a different combination of missing Cache-Control on HTML, wrong directive on hashed assets, or wrong duration on unhashed assets. Different surface, identical root: nobody was thinking about cache headers as a property of the dispatch surface, only of the immediate response.

### Image content blocks via OpenRouter→Anthropic (synthpanel hq-m333)

synthpanel emits Anthropic-shaped `image` content blocks. Direct Anthropic API receives and renders them. OpenRouter route to the same Anthropic models: image blocks are dropped at the proxy layer, model receives only text. **Persona output silently fabricated visual reasoning** ("squints at imaginary screen") instead of refusing. 100% delivery failure masked by persona roleplay until a deterministic probe with explicit "say NO IMAGE RECEIVED if you can't see it" instruction made it visible.

### `html` attachments via MCP (synthpanel hq-aaca)

CLI path renders html attachments correctly into the user message. MCP path: ~50% of personas saw the html, ~50% said "you didn't attach anything." Same instrument YAML, different dispatch path, different delivery rate. Inconsistent rather than zero — even worse to debug.

### Attachment resolver in v1 single-round (synthpanel hq-ilke)

v3 multi-round path resolves bank-ref attachments before sending to the model. v1 single-round path silently drops them. v1 callers got an attachment-less response with no error.

### Multi-round termination via MCP (synthpanel hq-fjdx)

CLI v3 multi-round runs all N rounds. MCP v3 multi-round terminates after round 1 and falls through positionally on the rest. Caller gets an "ensemble complete" event and a partial result; never sees the rest of the rounds attempted.

### `fetch_mode: screenshot` without playwright (synthpanel hq-gjh7)

When an instrument requests `fetch_mode: screenshot` but playwright extras aren't installed, the URL fetch silently degrades to `html_text` mode instead of failing loudly. Caller gets text where they specified pixels; no warning surfaces in the run log.

## Common shape

Each instance has the same anatomy:

1. **Multiple dispatch paths** to the same logical capability.
2. **A feature added at one path**, with tests that exercise that path.
3. **Other paths inherit a stale or empty version** of the feature.
4. **No loud failure mode** — degraded responses, dropped data, wrong headers, wrong durations.
5. **Discovery is manual**, usually by a careful user or a deterministic probe; never by routine testing.

What unites these is *not* shared code (in some cases the paths don't share an implementation at all). What unites them is **shared expectations from the caller**: a user calling `/sessions` expects the same Cache-Control discipline as `/assets/catalyst.css`; a model invoked via OpenRouter expects the same content blocks as via direct API; an instrument with attachments expects the resolver to fire whether on v1 or v3.

## Why this is hard

- **Tests live with paths, not with features.** When you add Cache-Control to the `_assets` handler, you write a test for the `_assets` handler. The test passes. The feature is "shipped." Nothing nudges you to ask "where else should this live?"

- **Mocks hide the asymmetry.** If two dispatch paths call into the same lower layer in tests but different layers in production, the mock makes them look symmetric.

- **Type systems don't help here.** All the surfaces typecheck. The problem is semantic, not structural.

- **Loud errors would help, but adding them is itself an asymmetric task.** "Raise on missing playwright" needs to be added to the screenshot fetcher; "log when attachments dropped" needs to be added to the v1 path. You're back at the original problem one level down.

## Detection: dispatch-path symmetry tests

The pattern that's worked best across these 7 instances is a CI test class: **for each feature exposed at multiple dispatch paths, write one test that probes ALL paths with the same input and asserts they produce the same observable behavior.**

Examples:

```python
# synthpanel hq-83ye: dispatch-path symmetry for multi-round v3
@pytest.mark.parametrize("dispatcher", [cli_dispatcher, mcp_dispatcher])
def test_v3_multi_round_runs_all_rounds(dispatcher):
    instrument = load_fixture("five_round_v3.yaml")
    result = dispatcher.run(instrument, personas=[fake_persona()])
    assert len(result.responses) == 5  # not 1
```

```bash
# scripts/audit-cache-headers.sh: cross-property symmetry audit
for url in $URLS; do
  curl -sI "$url" | check-cache-policy.sh
done
```

```python
# Hypothetical: probe both Anthropic-direct and OpenRouter-Anthropic for image delivery
@pytest.mark.parametrize("client", [direct_anthropic_client, openrouter_anthropic_client])
def test_image_content_block_reaches_model(client):
    resp = client.run_with_image(test_png)
    assert "image" in resp.text.lower() or assert_describes_pixels(resp)
```

Three properties of these tests:

- **Parametrized over dispatchers**, not over inputs. The variation is the path, not the data.
- **Asserts the SAME observable**. If symmetric paths produce different output for the same input, that's the bug.
- **Lives in CI**, runs on every PR. Adding a third dispatcher requires extending the parametrize set.

## Detection: at design time

Every PR that adds a feature should answer: *"What other dispatch paths in this system expose the same surface area? Does this feature need to be added there too?"*

Practical prompts during code review:

1. List every entrypoint that touches the same data shape this feature operates on. (HTTP routes, CLI commands, MCP handlers, library functions.)
2. For each, decide explicitly: feature applies / doesn't apply / TBD. Write the decision somewhere durable (PR description, code comment, follow-up bead).
3. If "feature applies" elsewhere and isn't being added in this PR, file a follow-up bead before merge. Don't trust "I'll get to it later."

## Remediation: when you find one

When the smell catches you in production:

1. **Reproduce on every dispatch path** before fixing one. The first reported path is rarely the only one affected — yggdrasil's `fetch_mode: screenshot` issue surfaced after similar issues in image content blocks and html attachments because nobody had checked the third dispatch path.
2. **Fix everywhere at once if possible**. Partial fixes leave the next user holding the same shape of bug on a different path.
3. **Add the dispatch-path symmetry CI test in the same PR**. Without that, the next addition will re-create the asymmetry.
4. **Catalog the instance** (this doc, or successor) so future code review can recognize the pattern faster.

## Why this is not "just write more tests"

The temptation is to read this and conclude "we need better test coverage." But every one of the 7 cataloged instances had test coverage on the working dispatch path. The tests passed. The feature shipped.

What's missing isn't *more* tests; it's *parametrized-over-dispatchers* tests, plus a code-review prompt that names the question explicitly. Both are cheap if you do them at design time. Both are very expensive if you do them after a user reports "nothing changed."

## See also

- `feedback_sibling_merge_cascade` (cross-town mayor memory) — the merge-time analog: code paths that touched shared files re-conflicting under parallel polecat execution. Different mechanism, same systemic shape.
- `docs/cache-control-policy.md` (dataviking-infra) — the cross-property cache policy that fell out of dvi-25f.
- `docs/known-patterns/openrouter-byok-visual-review.md` (synthpanel) — empirical workaround for the OR→Anthropic image-block drop while hq-m333 was still open.
- synthpanel `hq-83ye` — the first CI symmetry test of this kind in the org; reference implementation for follow-on coverage.
