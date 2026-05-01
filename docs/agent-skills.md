# Agent Skills & Slash Commands

synthpanel ships six Claude Code-native artifacts that drive the MCP
server: one slash command (`/synthpanel-poll`) and five skills
(`focus-group`, `name-test`, `concept-test`, `survey-prescreen`,
`pricing-probe`). They live at the repository root in
[`commands/`](../commands/) and [`skills/`](../skills/), and Claude Code
discovers them from the local filesystem rather than from the installed
Python package — so `pip install synthpanel[mcp]` alone doesn't expose
them. This page documents how to install them.

## What ships

| Artifact | File | Type | Triggers |
|---|---|---|---|
| `/synthpanel-poll <question>` | [`commands/synthpanel-poll.md`](../commands/synthpanel-poll.md) | Slash command | One-question quick poll |
| `focus-group` | [`skills/focus-group/SKILL.md`](../skills/focus-group/SKILL.md) | Skill | Full focus-group workflow |
| `name-test` | [`skills/name-test/SKILL.md`](../skills/name-test/SKILL.md) | Skill | 1–3 candidate name comparison |
| `concept-test` | [`skills/concept-test/SKILL.md`](../skills/concept-test/SKILL.md) | Skill | Concept / value-prop validation |
| `survey-prescreen` | [`skills/survey-prescreen/SKILL.md`](../skills/survey-prescreen/SKILL.md) | Skill | Pre-screen a survey instrument |
| `pricing-probe` | [`skills/pricing-probe/SKILL.md`](../skills/pricing-probe/SKILL.md) | Skill | Pricing sensitivity probe |

All six call into the synthpanel MCP server, so the
[MCP server](mcp.md) must be configured first — none of these work
standalone.

## Install paths

There are three ways to install the slash command and skills, ordered
by ease.

### 1. Claude Code plugin (easiest, Claude Code only)

If you use Claude Code, install the bundled plugin and you get
everything in one shot — no file copying:

```
/plugin install synthpanel
```

The plugin manifest at [`.claude-plugin/plugin.json`](../.claude-plugin/plugin.json)
registers the MCP server and points Claude Code at all six artifacts
under `commands/` and `skills/`. Restart Claude Code after install and
`/synthpanel-poll` plus the five skills are available.

### 2. Manual copy into Claude Code (any project, no plugin)

Claude Code looks for slash commands in `~/.claude/commands/` (user
scope, every project) or `<project>/.claude/commands/` (project scope,
one project only). Skills use the same pattern under `~/.claude/skills/`
or `<project>/.claude/skills/`. To install without the plugin, copy
the files in.

From a clone of this repo:

```bash
# User-scope install — every project gets the artifacts
mkdir -p ~/.claude/commands ~/.claude/skills
cp commands/synthpanel-poll.md ~/.claude/commands/
cp -r skills/focus-group skills/name-test skills/concept-test \
      skills/survey-prescreen skills/pricing-probe ~/.claude/skills/

# OR project-scope install — only this project gets the artifacts
mkdir -p .claude/commands .claude/skills
cp commands/synthpanel-poll.md .claude/commands/
cp -r skills/focus-group skills/name-test skills/concept-test \
      skills/survey-prescreen skills/pricing-probe .claude/skills/
```

Restart Claude Code. `/synthpanel-poll "your question"` and the five
skills (`focus-group`, `name-test`, etc.) are now available.

If you don't have a clone, fetch the files directly from GitHub:

```bash
SP_RAW=https://raw.githubusercontent.com/DataViking-Tech/SynthPanel/main

mkdir -p ~/.claude/commands ~/.claude/skills/focus-group \
         ~/.claude/skills/name-test ~/.claude/skills/concept-test \
         ~/.claude/skills/survey-prescreen ~/.claude/skills/pricing-probe

curl -fsSL $SP_RAW/commands/synthpanel-poll.md \
     -o ~/.claude/commands/synthpanel-poll.md

for s in focus-group name-test concept-test survey-prescreen pricing-probe; do
  curl -fsSL $SP_RAW/skills/$s/SKILL.md \
       -o ~/.claude/skills/$s/SKILL.md
done
```

### 3. Other MCP hosts (Cursor, Windsurf, Copilot, etc.)

`/synthpanel-poll` and the skills use Claude Code's slash-command and
skill conventions, which other hosts don't share. On those hosts:

- Configure the [MCP server](mcp.md#editor-configuration) so the host
  can call the synthpanel tools (`run_panel`, `run_quick_poll`,
  `run_prompt`, etc.) directly.
- Treat the contents of `commands/synthpanel-poll.md` and each
  `skills/*/SKILL.md` as **prompt templates** — paste the workflow
  body into the host's chat or your own prompt library and let the
  agent follow it manually.

The MCP tools themselves are host-agnostic; only the
slash-command/skill packaging is Claude Code-specific.

## Verifying the install

After installing, in a Claude Code session:

```
/synthpanel-poll "Should we name this thing 'Traitprint' or 'Personagram'?"
```

Claude Code routes the slash command, which calls
`mcp__synth_panel__run_quick_poll`, which calls the synthpanel MCP
server. If the MCP server is misconfigured you'll get a tool-call
error pointing at the missing piece (typically a missing API key —
see [Sampling Mode](mcp.md#sampling-mode) for the zero-config
fallback).

Skills are auto-discovered by Claude Code from the YAML frontmatter
in each `SKILL.md`. Ask the agent in plain language ("run a focus
group on …", "test these three names") and the relevant skill loads
into context. You can also invoke a skill explicitly by name (e.g.
`/focus-group`) when Claude Code surfaces it in the slash-command
auto-complete.

## Updating

When synthpanel ships a new version of a skill or slash command:

- **Plugin install:** re-run `/plugin install synthpanel` (or
  whatever Claude Code's plugin-update flow is at that time).
- **Manual copy:** repeat the copy step from §2 above. Files
  overwrite cleanly.

There's no version pinning between the MCP server and the artifacts —
the artifacts are pure prompt content with no compiled dependency on
the server. A newer skill against an older server (or vice versa)
works as long as the named MCP tools still exist.

## See also

- [MCP server reference](mcp.md) — server config, tool list, sampling
  mode.
- [`commands/synthpanel-poll.md`](../commands/synthpanel-poll.md) —
  the actual slash command source.
- [`skills/`](../skills/) — the five skill workflows.
- [`.claude-plugin/plugin.json`](../.claude-plugin/plugin.json) —
  what `/plugin install synthpanel` registers.
