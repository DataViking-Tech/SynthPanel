
# OpenRouter BYOK: visual review of HTML/UI content

**Status:** Working pattern as of v1.0.2. Use this when:

- You want a synthetic panel to react to the visual presentation of a web UI, dashboard, or rendered HTML document.
- Your LLM provider is OpenRouter (BYOK), and direct provider creds aren't available.
- You're hitting the documented attachment limitations: image attachments fail
  ~100% of the time on OpenRouter's Anthropic routes (see `hq-m333`); `html`
  attachments deliver ~50% of the time (see `hq-aaca`).

This pattern bypasses both attachment paths by inlining the source HTML
directly into the question text via template-variable substitution. Delivery
to the model is deterministic — every persona receives the content as part
of the user message, no transport-layer surprises.

## When to use a different pattern

- **Direct Anthropic / OpenAI / Gemini creds** — use the documented `image`
  attachment path. Image content blocks work natively when synthpanel talks
  to the upstream provider directly.
- **Document review (PDF)** — use the `document` attachment type with
  `media_type: application/pdf`. Same delivery caveats as image attachments
  via OpenRouter; works directly with Anthropic Files API.
- **Live page review without source access** — use the `url` attachment type
  with `fetch_mode: screenshot` or `markdown`. Same OpenRouter delivery
  caveat; the URL fetch happens server-side regardless.

## The pattern

Three files: an instrument with template placeholders, a vars file with the
HTML payload, and a synthesis script that takes the panel output and
produces a single executive summary via one extra LLM call.

### 1. Capture the HTML

For a server-rendered page or static HTML you control, just read the file.
For an SPA or page rendered behind auth, screenshot it via headless
chromium first, then dump the rendered HTML:

```python
# render-and-dump.py
import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 920, "height": 1100})
        await page.goto("https://your-app.example.com/some-page")
        # If auth-gated, do auth here. For internal tools, run wrangler dev or
        # equivalent to bypass auth before this step.
        html = await page.content()
        with open("captured.html", "w") as f:
            f.write(html)
        await browser.close()

asyncio.run(main())
```

For HTML you already have on disk, skip this step.

### 2. Write the instrument with template placeholders

```yaml
# instrument-vars.yaml
name: my-visual-review
version: 1
description: Visual review of <thing> via inline HTML template substitution.
author: <you>

instrument:
  version: 3
  rounds:
    - name: first_impressions
      questions:
        - text: |
            Below is the complete HTML+CSS source of <thing>. Read it
            carefully, mentally render it, then react.

            ---
            {ack_html}
            ---

            Reconstruct the visual experience: layout, hierarchy, color,
            density. What does this page communicate, and to whom?
        - text: |
            On a scale of 1-10, how visually pleasant would the page above
            be when rendered? Brief reasoning anchored to specific CSS
            choices (font stack, colors, spacing, border-radius).

    - name: comprehension
      questions:
        - text: |
            In the page above, the buttons at the bottom are labeled
            Approve, Reject, and Re-queue. Without explanation, what would
            each one do in your mental model? Are any ambiguous?

    - name: concrete_changes
      questions:
        - text: |
            Give three concrete CSS-level changes (specific properties +
            values, not vague directions) that would noticeably improve
            the visual quality of these screens. Order by impact-per-effort,
            highest first.
```

Use `{varname}` placeholders inside the question text wherever the HTML
should land. Multiple placeholders per question are fine (e.g., showing two
pages side-by-side).

### 3. Build the vars file

```yaml
# vars.yaml
ack_html: |
  <!doctype html>
  <html>
  <head>...</head>
  <body>
    <h1>...</h1>
    ...
  </body>
  </html>

list_html: |
  <!doctype html>
  ...
```

The literal-block scalar (`|`) preserves whitespace and newlines exactly,
which keeps the HTML readable in the prompt.

For programmatic generation, write the YAML via Python so you don't have
to hand-quote large HTML payloads:

```python
import yaml
ack = open("captured.html").read()
list_ = open("list-page.html").read()
yaml.safe_dump(
    {"ack_html": ack, "list_html": list_},
    open("vars.yaml", "w"),
    default_style="|",
    sort_keys=False,
    allow_unicode=True,
)
```

### 4. Run the panel

```bash
synthpanel --output-format json panel run \
  --personas developer.yaml \
  --personas-merge enterprise-buyer.yaml \
  --instrument instrument-vars.yaml \
  --vars-file vars.yaml \
  --models openrouter/anthropic/claude-sonnet-4.5 \
  --max-cost 5.0 \
  --no-synthesis \
  > panel-output.log 2>&1
```

Notes:

- **`--output-format json`** — required for stable parsing. Place this
  flag before `panel run`, not after.
- **`--models openrouter/anthropic/claude-sonnet-4.5`** — Sonnet 4.5 reasons
  about CSS properties, font choices, and brand fit competently. Haiku is
  too underpowered for fine-grained visual reasoning even with text-mode
  delivery; expect generic responses.
- **`--no-synthesis`** — the bundled synthesis prompt is tuned for survey
  data, not free-text design feedback. Skip it; do a custom synthesis call
  yourself (next step).
- **`--save`** — currently a no-op for this path (see `hq-0pnq`). The full
  panel result is in stdout via `--output-format json`; capture stdout
  instead.

### 5. Synthesize manually

The bundled synthesis prompt produces summary statistics, which isn't what
you want for visual review. Pull the responses out of the NDJSON log and
fire one more LLM call with a prompt tuned to your task:

```python
# synthesize.py
import os, sys, json, urllib.request

key = os.environ["OPENROUTER_API_KEY"]
with open("panel-output.log") as f:
    for line in f:
        if line.startswith("{") and '"Ensemble complete"' in line:
            data = json.loads(line)
            break

results = data["per_model_results"]["openrouter/anthropic/claude-sonnet-4.5"]["results"]
payload = []
for r in results:
    for resp in r["responses"]:
        payload.append({
            "persona": r["persona"],
            "question_first_30w": " ".join(resp["question"].split()[:30]),
            "response": resp["response"],
        })

prompt = (
    "Below are responses from synthetic personas reviewing the visual "
    "presentation of <thing>. Synthesize into a tight executive summary. "
    "Focus on: 1) overall verdict + 2-3 consistent praises and criticisms, "
    "2) any specific UI element flagged by multiple personas, 3) top-N "
    "concrete improvements (with property:value pairs where given), "
    "ordered by frequency × persona-type weight.\n\n"
    "Be terse. Quote directly only when a phrase captures something multiple "
    "personas echoed.\n\n"
    "Responses (JSON):\n" + json.dumps(payload)
)

body = json.dumps({
    "model": "anthropic/claude-sonnet-4.5",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 2500,
}).encode()

req = urllib.request.Request(
    "https://openrouter.ai/api/v1/chat/completions",
    data=body,
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=300) as r:
    d = json.loads(r.read())
    print(d["choices"][0]["message"]["content"])
```

Run it:

```bash
OPENROUTER_API_KEY=$(... your key fetch ...) python3 synthesize.py
```

## Empirical anchor

Reference run: 30 personas (15 developers + 15 enterprise buyers) × 7
questions × Sonnet 4.5 BYOK. **Zero refusals out of 210 responses.**
Persona-level reasoning quality was high — concrete CSS property
suggestions, brand-fit assessments anchored to specific design tokens,
divergence between developer and enterprise-buyer perspectives surfaced
naturally.

Total cost: ~$3.50 panel + ~$0.39 synthesis call ≈ **$4 for a complete
actionable UI review** producing a multi-page exec summary the operator
acted on directly.

## What this pattern does NOT cover

- **Visual rendering quality** — the model reasons about HTML+CSS source,
  not the rendered pixels. It can score visual hierarchy from `font-size`
  and `font-weight` declarations but won't catch a font-loading failure or
  a layout shift caused by a JS-rendered widget.
- **Interactive flows** — single static snapshots only. For multi-step
  flows, decompose into separate snapshot questions or use the `url`
  attachment with `fetch_mode: screenshot` once `hq-m333` lands.
- **Image-only content** — if the page is mostly imagery (e.g., a marketing
  hero with text rendered into a JPEG), the HTML source won't carry the
  semantic content. Use the `image` attachment path with direct provider
  creds, or capture text via OCR and include separately.

## See also

- `hq-m333` — transport-layer investigation for OpenRouter image attachments
- `hq-aaca` — `html` attachment delivery flake via OpenRouter
- `hq-0pnq` — `--save` flag silent no-op
- `feedback_async_wisp_consult_default` (cross-town mayor memory) — the
  consult cadence that produced this pattern
