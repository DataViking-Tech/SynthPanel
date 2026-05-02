---
description: Run a one-question synthetic poll across AI personas and synthesize the themes
allowed-tools:
  - mcp__synth_panel__run_quick_poll
  - mcp__synth_panel__list_persona_packs
  - mcp__synth_panel__get_persona_pack
argument-hint: <question to ask the panel>
---

Run a quick synthetic poll using the SynthPanel MCP server. One question, a
handful of AI personas, one synthesized writeup.

Question: $ARGUMENTS

## What to do

1. **Resolve the question.** If `$ARGUMENTS` is empty, ask the user for the
   question before continuing. Do not invent one.

2. **Pick personas.** Default to the built-in zero-config persona set by
   omitting the `personas` argument. Only build a custom list when the user
   names a specific audience (e.g. "three enterprise SRE personas"). In that
   case, either:
   - Call `mcp__synth_panel__list_persona_packs` and `get_persona_pack` to
     pull an installed pack, or
   - Hand-author 3–5 dicts with `name`, `age`, `occupation`, `background`,
     and `personality_traits`.

   Keep panels small — `run_quick_poll` caps at 3 personas when running in
   sampling mode (no API key).

3. **Run the poll.** Call `mcp__synth_panel__run_quick_poll` with:
   - `question`: the user's question
   - `personas`: omit for the default set, or pass your custom list
   - `synthesis`: leave as `true` (default) — the synthesis pass is the
     point of this command

   Do not pass `model` unless the user explicitly asked for a specific one.
   The MCP server picks a sensible default (haiku in BYOK mode, or the
   host's own model in sampling mode).

4. **Report the result.** Present the output in this shape:

   ```
   Question: <the question>
   Panel: <n> personas (<mode: BYOK model-name | sampling>)

   ## Themes
   <bulleted list from the synthesis>

   ## Per-persona responses
   - <persona name>: <one-line summary of their response>
   - ...

   ## Notable divergences
   <any disagreements the synthesis flagged>

   Cost: <$X.XX from the response's `cost` field, or "n/a" in sampling mode>
   ```

   Keep each persona summary to one line. The full transcripts are in the
   raw tool output — don't re-print them.

## Guardrails

- **Synthetic panels are exploratory, not validating.** End the report with
  a single-line caveat: "Synthetic panel — directional signal only, not a
  substitute for real user research."
- **Don't loop.** If the user wants a second question, that's another
  `/synthpanel-poll` invocation, not a follow-up inside this one. For
  multi-question structured studies, suggest the `/focus-group` skill
  instead.
- **Report errors honestly.** If the tool returns an `error` field
  (missing API key, persona cap exceeded, etc.), show it verbatim and
  suggest the fix — don't silently fall back.
