# Attachments: showing panelists images, web pages, and HTML snippets

When you want a synthetic panel to react to something visual or external
— a screenshot, a live landing page, a marketing snippet — you give the
instrument an **attachment bank** and reference its entries from
individual questions.

A copy-pasteable starter lives at
[`examples/instruments/with-attachments.yaml`](../../examples/instruments/with-attachments.yaml).
Run it with:

```bash
althing panel run \
  --personas examples/personas.yaml \
  --instrument examples/instruments/with-attachments.yaml
```

## The four attachment types

| Type | Source | Use when |
|------|--------|----------|
| `image` | `base64` data, `url`, or uploaded `file_id` | You want panelists to react to a screenshot, mockup, or photo. |
| `document` | `base64` data, `url`, or `file_id` | You want panelists to read a PDF (spec, brief, contract). |
| `url` | A live URL fetched at run time | You want panelists to react to a real, currently-live page. |
| `html` | Inline string | You have a literal HTML/markup fragment — pricing block, email body, copy variant. |

## Shape

The bank is a top-level mapping keyed by attachment id. Each id must
match `^[a-z][a-z0-9_-]{0,63}$`. Questions reference ids by name:

```yaml
instrument:
  version: 3
  attachments:
    hero_screenshot:
      type: image
      media_type: image/png
      source:
        type: base64
        data: iVBORw0KGgoAAAANSUhEUgAAAAEA...
    landing_page:
      type: url
      url: https://example.com/pricing
      fetch_mode: markdown
    pricing_snippet:
      type: html
      text: |
        <div class="pricing">
          <h2>Starter — $9/mo</h2>
        </div>
  rounds:
    - name: visual_reaction
      questions:
        - text: "What stands out about the screenshot?"
          attachments: [hero_screenshot]
        - text: "Does the live pricing page land?"
          attachments: [landing_page]
        - text: "Is the Starter tier markup fairly priced?"
          attachments: [pricing_snippet]
      route_when:
        - else: __end__
```

## Notes

- **`fetch_mode` for `url`** — one of `auto`, `html_text`, `screenshot`,
  or `markdown`. `markdown` is usually the cleanest signal for
  content-heavy pages; `screenshot` is the right choice when layout or
  visual hierarchy is the thing under test.
- **`image` and `screenshot` attachments require a vision-capable model.**
  Text-only models (e.g. `claude-3.5-haiku`) are rejected fast with an
  explicit error — point `--models` at a multimodal model such as
  `claude-haiku-4.5`, `gpt-4o-mini`, or `gemini-2.0-flash`. (`markdown` /
  `html_text` modes return text and work with any model.)
- **`fetch_mode: screenshot` needs the `visual` extra.** Install with
  `pip install 'althing[visual]'`, then fetch the browser once with
  `python -m playwright install chromium`. Without it, screenshot mode
  can't render the page. Text-only modes have no extra dependency.
- **Failed URL fetches are a hard error by default (sy-550).** If a
  `type: url` attachment can't be fetched or yields no usable content
  (perimeter-denied / HTTP error / timeout / empty extraction), the
  affected question fails loudly — naming the URL and reason — and counts
  in the run's failure rate, instead of silently sending the persona an
  empty page (which makes personas answer blind while the run still
  reports 0% failures). Pass `--allow-empty-attachments` to `panel run`
  to opt into best-effort behaviour, where a failed fetch becomes a
  placeholder note and the run continues. Either way, a per-attachment
  `attachment_fetch_status` (`ok` / `failed` + reason) is recorded on
  each response in the saved result for auditability.
- **Loopback / private addresses are SSRF-blocked.** URLs that resolve to
  `localhost`, `127.0.0.1`, or RFC-1918 private ranges (`10.x`,
  `172.16–31.x`, `192.168.x`, link-local, etc.) are rejected by the fetch
  perimeter — so **a local preview server cannot be used as a `type: url`
  source** (e.g. `http://localhost:4321/` fails with `perimeter denied …
  loopback`). For local or not-yet-published content, embed it inline as
  a `type: html` (or `type: document`) attachment instead of pointing at a
  local URL.
- **Inline HTML** — YAML's `|` (literal block scalar) preserves newlines
  and indentation, which is what you want for HTML fragments.
- **Image media types** — `image/png`, `image/jpeg`, `image/gif`,
  `image/webp`. Anything else fails at parse time.
- **Document media types** — currently only `application/pdf`.
- **Attachment ids are validated at parse time** — typos in attachment
  references fail fast with `attachment ref '…' does not resolve to a
  top-level attachment`.

## See also

- [`examples/instruments/with-attachments.yaml`](../../examples/instruments/with-attachments.yaml)
  — the runnable starter this page describes.
- [Instrument regression testing in CI](./instrument-regression-testing.md)
  — locks panel behavior, including attachment-driven runs.
