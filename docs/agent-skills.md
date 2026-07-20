# Agent Skills & Slash Commands

althing ships seven Claude Code-native artifacts that drive the MCP
server: one slash command (`/althing-poll`), one top-level director
skill (`althing`), and five workflow skills (`focus-group`,
`name-test`, `concept-test`, `survey-prescreen`, `pricing-probe`). They
live at the repository root in [`commands/`](../commands/) and
[`skills/`](../skills/), and Claude Code discovers them from the local
filesystem rather than from the installed Python package — so
`pip install althing[mcp]` alone doesn't expose them. This page
documents how to install them.

The skill files follow the open
[Agent Skills Discovery](https://agentskills.io/) format, so
[Hermes Agent](https://hermes-agent.nousresearch.com/), Codex CLI,
Cursor, Goose, OpenHands, and other skills-compatible hosts can load
them with the same files. The `althing` director skill in particular
includes Hermes-specific frontmatter (`metadata.hermes.*`,
`required_environment_variables`) so Hermes can install and configure
it without ad-hoc spelunking.

## What ships

| Artifact | File | Type | Triggers |
|---|---|---|---|
| `/althing-poll <question>` | [`commands/althing-poll.md`](../commands/althing-poll.md) | Slash command | One-question quick poll |
| `althing` | [`skills/althing/SKILL.md`](../skills/althing/SKILL.md) | Skill (director) | "run a synthetic panel", "test positioning", "compare names", "pricing probe", "poll N personas", "prioritization poll" |
| `focus-group` | [`skills/focus-group/SKILL.md`](../skills/focus-group/SKILL.md) | Skill | Full focus-group workflow |
| `name-test` | [`skills/name-test/SKILL.md`](../skills/name-test/SKILL.md) | Skill | 1–3 candidate name comparison |
| `concept-test` | [`skills/concept-test/SKILL.md`](../skills/concept-test/SKILL.md) | Skill | Concept / value-prop validation |
| `survey-prescreen` | [`skills/survey-prescreen/SKILL.md`](../skills/survey-prescreen/SKILL.md) | Skill | Pre-screen a survey instrument |
| `pricing-probe` | [`skills/pricing-probe/SKILL.md`](../skills/pricing-probe/SKILL.md) | Skill | Pricing sensitivity probe |

All seven call into the althing MCP server, so the
[MCP server](mcp.md) must be configured first — none of these work
standalone.

## Install paths

There are three ways to install the slash command and skills, ordered
by ease.

### 1. Claude Code plugin (easiest, Claude Code only)

If you use Claude Code, install the bundled plugin and you get
everything in one shot — no file copying:

```
/plugin install althing
```

The plugin manifest at [`.claude-plugin/plugin.json`](../.claude-plugin/plugin.json)
registers the MCP server and points Claude Code at all six artifacts
under `commands/` and `skills/`. Restart Claude Code after install and
`/althing-poll` plus the five skills are available.

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
cp commands/althing-poll.md ~/.claude/commands/
cp -r skills/althing skills/focus-group skills/name-test \
      skills/concept-test skills/survey-prescreen skills/pricing-probe \
      ~/.claude/skills/

# OR project-scope install — only this project gets the artifacts
mkdir -p .claude/commands .claude/skills
cp commands/althing-poll.md .claude/commands/
cp -r skills/althing skills/focus-group skills/name-test \
      skills/concept-test skills/survey-prescreen skills/pricing-probe \
      .claude/skills/
```

Restart Claude Code. `/althing-poll "your question"` and the six
skills (`althing`, `focus-group`, `name-test`, etc.) are now
available.

If you don't have a clone, fetch the files directly from GitHub:

```bash
SP_RAW=https://raw.githubusercontent.com/DataViking-Tech/Althing/main

mkdir -p ~/.claude/commands \
         ~/.claude/skills/althing ~/.claude/skills/focus-group \
         ~/.claude/skills/name-test ~/.claude/skills/concept-test \
         ~/.claude/skills/survey-prescreen ~/.claude/skills/pricing-probe

curl -fsSL $SP_RAW/commands/althing-poll.md \
     -o ~/.claude/commands/althing-poll.md

for s in althing focus-group name-test concept-test survey-prescreen pricing-probe; do
  curl -fsSL $SP_RAW/skills/$s/SKILL.md \
       -o ~/.claude/skills/$s/SKILL.md
done
```

### 3. Hermes Agent (skills-compatible host)

Hermes consumes the same SKILL.md format (Agent Skills Discovery,
agentskills.io). The `althing` director skill carries Hermes-specific
frontmatter (`metadata.hermes.tags`, `metadata.hermes.config`,
`required_environment_variables`) so it installs and configures
cleanly:

```bash
SP_RAW=https://raw.githubusercontent.com/DataViking-Tech/Althing/main
HERMES_SKILLS=~/.hermes/skills   # adjust to your Hermes install path

mkdir -p $HERMES_SKILLS/althing $HERMES_SKILLS/focus-group \
         $HERMES_SKILLS/name-test $HERMES_SKILLS/concept-test \
         $HERMES_SKILLS/survey-prescreen $HERMES_SKILLS/pricing-probe

for s in althing focus-group name-test concept-test survey-prescreen pricing-probe; do
  curl -fsSL $SP_RAW/skills/$s/SKILL.md \
       -o $HERMES_SKILLS/$s/SKILL.md
done
```

Hermes will prompt for the configured environment variables (Anthropic /
OpenAI / Google / xAI keys) on first activation, then route Althing
triggers ("run a synthetic panel", "test positioning", "compare names",
"pricing probe", "poll 30 personas") to the `althing` director,
which in turn delegates to the matching workflow skill. Also configure
the [MCP server](mcp.md#editor-configuration) so the tools resolve.

The index of skills is also published at
[`/.well-known/agent-skills/index.json`](https://althing.dev/.well-known/agent-skills/index.json)
on althing.dev, per Agent Skills Discovery RFC v0.2.0, so
discovery-aware hosts can find them without prior knowledge.

### 4. Other MCP hosts (Cursor, Windsurf, Copilot, etc.)

`/althing-poll` is Claude Code-specific. On other hosts:

- Configure the [MCP server](mcp.md#editor-configuration) so the host
  can call the althing tools (`run_panel`, `run_quick_poll`,
  `run_prompt`, etc.) directly.
- For hosts that consume SKILL.md (Hermes, Codex CLI, Cursor, Goose,
  OpenHands, OpenCode, …), drop the `skills/althing` directory plus
  the workflow skills into that host's skills path.
- For hosts that don't consume SKILL.md, treat the contents of
  `commands/althing-poll.md` and each `skills/*/SKILL.md` as
  **prompt templates** — paste the workflow body into the host's chat
  or your own prompt library and let the agent follow it manually.

The MCP tools themselves are host-agnostic; only the
slash-command/skill packaging is host-specific.

## Verifying the install

After installing, in a Claude Code session:

```
/althing-poll "Should we name this thing 'Traitprint' or 'Personagram'?"
```

Claude Code routes the slash command, which calls
`mcp__althing__run_quick_poll`, which calls the althing MCP
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

When althing ships a new version of a skill or slash command:

- **Plugin install:** re-run `/plugin install althing` (or
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
- [`commands/althing-poll.md`](../commands/althing-poll.md) —
  the actual slash command source.
- [`skills/althing/SKILL.md`](../skills/althing/SKILL.md) — the
  top-level director skill (Hermes-compatible).
- [`skills/`](../skills/) — the five workflow skills.
- [`.claude-plugin/plugin.json`](../.claude-plugin/plugin.json) —
  what `/plugin install althing` registers.
