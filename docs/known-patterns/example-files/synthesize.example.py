import json
import os
import urllib.error
import urllib.request

key = os.environ["OPENROUTER_API_KEY"]
with open("/tmp/sp-synthesis-input.json") as f:
    payload_text = f.read()

prompt = (
    "Below are 210 responses from 30 synthetic personas (15 developers + 15 "
    "enterprise buyers) reviewing the visual presentation of the DataViking "
    "Boardroom internal UI — an LLM-generated implementation spec ACK page and "
    "a sessions list dashboard.\n\n"
    "Synthesize their feedback into a tight executive summary for Wesley "
    "(founder/operator). Focus on:\n\n"
    "1. **Visual quality verdict** — overall rating, with 2-3 things personas "
    "consistently praised AND 2-3 things they consistently criticized.\n"
    "2. **Re-queue button verdict** — was its purpose understood? Recommend "
    "rename or keep.\n"
    "3. **Pre-block markdown rendering verdict** — should literal `# Context` "
    "stay or render as H1?\n"
    "4. **Brand-fit verdict** — does this feel like a sharp AI startup tool, "
    "or drift toward enterprise CRM / consumer SaaS?\n"
    "5. **Top 5 concrete CSS changes** — extract the most-mentioned specific "
    "improvements (with property:value pairs where given). Order by "
    "frequency × persona-type weight.\n"
    "6. **One sentence each on developer vs enterprise-buyer divergence.**\n\n"
    "Be terse. Quote directly only when a phrase captures something multiple "
    "personas echoed. No filler. This is for someone who runs companies.\n\n"
    "Responses (JSON array):\n" + payload_text
)

body = json.dumps(
    {
        "model": "anthropic/claude-sonnet-4.5",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2500,
    }
).encode()

req = urllib.request.Request(
    "https://openrouter.ai/api/v1/chat/completions",
    data=body,
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
        print(d["choices"][0]["message"]["content"])
        print(f"\n[usage: {d.get('usage')}]")
except urllib.error.HTTPError as e:
    print("ERROR", e.code, e.read().decode())
