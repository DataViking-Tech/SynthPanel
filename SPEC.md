# Synth-Panel Foundation Specification

**Version**: 1.0.0
**Date**: 2026-04-03
**Purpose**: Functional specification for a synthetic focus group CLI tool ("synth-panel") that orchestrates multiple LLM-powered personas to generate structured qualitative feedback.

**Audience**: Implementers who will build synth-panel from scratch. This document describes behavioral contracts and data flows only -- no implementation details from any reference codebase.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [LLM Client Abstraction](#2-llm-client-abstraction)
3. [Agent Runtime / Session Loop](#3-agent-runtime--session-loop)
4. [Multi-Agent Orchestration](#4-multi-agent-orchestration)
5. [Structured Output](#5-structured-output)
6. [Session Persistence](#6-session-persistence)
7. [Cost and Budget Tracking](#7-cost-and-budget-tracking)
8. [CLI Framework](#8-cli-framework)
9. [Plugin and Extension System](#9-plugin-and-extension-system)
10. [Cross-Cutting Concerns](#10-cross-cutting-concerns)

---

## 1. System Overview

Synth-panel is a CLI tool that simulates a focus group by:

- Accepting a stimulus (brand name, product concept, survey question, etc.)
- Spawning multiple LLM-backed "panelist" agents, each with a distinct persona
- Collecting structured responses from each panelist
- Aggregating results into a report

The system requires eight foundational capabilities described in this specification. Each component is designed so that it can be built, tested, and replaced independently.

### Design Principles

- **Provider agnostic**: The system must work with multiple LLM providers without changes to business logic.
- **Cost-aware**: Every LLM call is tracked and can be budget-gated. Focus group simulations can be expensive; the operator must have control.
- **Structured by default**: Panelist responses must conform to declared schemas. Free-text is the exception, not the rule.
- **Resumable**: Long-running panels can be interrupted and resumed without data loss.
- **Extensible**: New persona types, output formats, and provider integrations are added without modifying core code.

---

## 2. LLM Client Abstraction

### Purpose

Provide a single interface for sending prompts to any supported LLM provider. The rest of the system never constructs HTTP requests or parses provider-specific response formats.

### Interface Contract

The client abstraction exposes two operations:

1. **Send message (blocking)**: Accept a request, return a complete response.
2. **Stream message**: Accept a request, return an iterator/stream of incremental events.

Both operations accept the same request structure and share the same error type.

### Data Model

**Completion Request**:
- Model identifier (string)
- Maximum output tokens (unsigned integer)
- Conversation messages (ordered list of input messages)
- System prompt (optional string)
- Tool definitions (optional list of tool schemas)
- Tool choice constraint (optional: auto / any / specific tool name)
- Stream flag (boolean, defaults to false)

**Input Message**:
- Role: "user" or "assistant"
- Content blocks (ordered list), where each block is one of:
  - Text block (string content)
  - Tool invocation block (id, tool name, JSON input)
  - Tool result block (tool invocation id, result content blocks, error flag)

**Completion Response**:
- Unique response identifier
- Model identifier (which model actually served the request)
- Role (always "assistant")
- Output content blocks, where each block is one of:
  - Text (string content)
  - Tool invocation (id, tool name, JSON input)
  - Thinking (internal reasoning text, optional cryptographic signature)
- Stop reason (end of turn, tool use, max tokens, etc.)
- Token usage counters (see below)

**Token Usage**:
- Input tokens consumed
- Output tokens generated
- Cache write tokens (tokens written to prompt cache)
- Cache read tokens (tokens served from prompt cache)

**Stream Events** (emitted in order during streaming):
1. Message start (contains partial response metadata)
2. Content block start (index + block type)
3. Content block delta (incremental text, JSON fragments, or thinking deltas)
4. Content block stop (index)
5. Message delta (stop reason, final usage)
6. Message stop (terminal event)

The stream parser must handle:
- Chunked delivery (a single event split across multiple network chunks)
- Ping/keepalive frames (silently discarded)
- End-of-stream sentinel values (silently discarded)
- Multi-line data fields within a single SSE frame

### Provider Resolution

The system must support at least three provider families:

| Provider Family | Model Prefix | Auth Mechanism |
|----------------|-------------|----------------|
| Anthropic | claude-* | API key or OAuth bearer token via environment variable |
| xAI | grok-* | API key via environment variable |
| OpenAI-compatible | (configurable) | API key via environment variable |

**Model alias resolution**: Short aliases (e.g., "opus", "sonnet", "haiku", "grok") must resolve to canonical model identifiers before provider detection.

**Provider detection order**:
1. If the canonical model name matches a known prefix, use that provider.
2. If no prefix matches, check which provider credentials are available in the environment, in priority order.

Each provider must be configurable via:
- An API key environment variable name
- A base URL environment variable name
- A default base URL

### Error Handling

Errors fall into these categories:

| Category | Retryable | Examples |
|----------|-----------|---------|
| Missing credentials | No | API key env var unset or empty |
| Authentication failure | No | Invalid key, expired OAuth token |
| HTTP transport errors | Yes | Connection refused, timeout, request error |
| API errors (server-side) | Conditional | 429 rate limit (yes), 400 bad request (no), 500 server error (yes) |
| Deserialization errors | No | Malformed JSON response |
| Retries exhausted | No | All retry attempts failed |

**Retry policy**:
- Initial backoff: 200ms
- Maximum backoff: 2 seconds
- Maximum retry attempts: 2 (configurable)
- Backoff strategy: exponential with jitter
- Only retryable errors trigger retries

### Extension Points

- New providers are added by implementing the two-operation interface (send + stream) and registering a model prefix pattern.
- Base URLs are overridable via environment variables, enabling proxy or mock server use.
- Prompt caching is an optional layer that wraps the client. It is provider-specific (only some providers support it) and tracks cache hit/miss statistics per session.

---

## 3. Agent Runtime / Session Loop

### Purpose

Manage the lifecycle of a single agent conversation: accept user input, send it to the LLM, execute any tool calls the LLM requests, and loop until the LLM produces a final text response or a budget/iteration limit is reached.

### Interface Contract

The runtime exposes a single primary operation:

**Run Turn**: Given user input text, execute a complete conversational turn. A "turn" may involve multiple iterations if the LLM requests tool use.

Returns a turn summary containing:
- All assistant messages produced during the turn
- All tool results produced during the turn
- Number of iterations within the turn
- Cumulative token usage
- Whether automatic compaction was triggered

### Data Model

**Conversation Message**:
- Role: system, user, assistant, or tool
- Content blocks (text, tool invocations, tool results)
- Token usage (optional, present on assistant messages)

**Session** (the full state of one agent's conversation):
- Unique session identifier
- Version number
- Creation and last-update timestamps
- Ordered list of conversation messages
- Compaction metadata (how many times compacted, summary text)
- Fork metadata (parent session ID, branch name) for sessions derived from another

### Lifecycle

A single turn proceeds as follows:

1. **Accept user input**: Push user text as a new message onto the session.
2. **Build API request**: Combine system prompt + all session messages.
3. **Call LLM**: Stream or send the request.
4. **Process response**: Parse assistant output into content blocks.
5. **Record usage**: Update cumulative token counters.
6. **Check for tool calls**: If the assistant response contains tool invocation blocks, continue to step 7. Otherwise, the turn is complete.
7. **For each tool call**:
   a. Run pre-tool-use hooks (may deny/cancel the tool).
   b. Check permission policy (may prompt the operator).
   c. If allowed, execute the tool and capture output.
   d. Run post-tool-use hooks (may modify output or flag errors).
   e. Push tool result message onto the session.
8. **Check iteration limit**: If not exceeded, go to step 2. Otherwise, error.
9. **Check auto-compaction**: If cumulative input tokens exceed a threshold (default 100K), compact older messages into a summary.
10. **Return turn summary**.

### Error Handling

- LLM API errors propagate upward after retries are exhausted.
- Tool execution errors are captured as error-flagged tool results and fed back to the LLM (the agent gets a chance to recover).
- Iteration limit exceeded is a hard error.
- Session persistence errors (if auto-persist is enabled) roll back the last message push.

### Extension Points

- **Tool executor**: Any component implementing "execute(tool_name, input) -> output" can be plugged in.
- **Permission policy**: Controls which tools are allowed, denied, or require operator confirmation. Policies are composable.
- **Hook runner**: Pre- and post-tool-use hooks intercept tool execution (see Plugin System).
- **Session tracer**: Optional telemetry integration for recording turn-level metrics.

---

## 4. Multi-Agent Orchestration

### Purpose

Manage the lifecycle of multiple independent agent sessions working in parallel or in a coordinated workflow. For synth-panel, this means spawning one agent per panelist persona and coordinating their execution.

### Interface Contract

**Worker Registry**: A thread-safe registry that tracks the state of spawned workers.

Operations:
- **Create worker**: Given a working context and trust configuration, create a new worker entry and return its identifier.
- **Observe worker**: Feed screen/output text from the worker process to the registry. The registry uses pattern matching to detect state transitions (ready for input, running, blocked, etc.).
- **Send prompt to worker**: Deliver a prompt to a worker that is in the "ready" state.
- **Await ready**: Poll a worker's readiness status.
- **Restart worker**: Reset a worker to its initial state.
- **Terminate worker**: Mark a worker as finished.

**Team Registry**: Groups workers into named teams.

Operations:
- Create a team with a name and a list of task identifiers
- List all teams
- Get/delete a team (soft delete with status change)

**Scheduled Execution Registry**: Manages recurring/scheduled agent invocations.

Operations:
- Create a scheduled entry with a cron expression, prompt, and optional description
- List entries (optionally filtered to enabled-only)
- Disable an entry without removing it
- Record a run (increment counter, update last-run timestamp)
- Delete an entry (hard delete)

### Data Model

**Worker**:
- Unique worker identifier
- Working directory
- Current status (see lifecycle)
- Trust configuration (auto-resolve flag, trust-gate-cleared flag)
- Prompt delivery state (attempt count, last prompt, replay prompt)
- Error state (failure kind + message)
- Event log (ordered sequence of state transitions with timestamps)

**Worker Status Lifecycle**:
```
Spawning --> TrustRequired --> [trust resolved] --> Spawning
Spawning --> ReadyForPrompt --> PromptAccepted --> Running --> Finished
                                     |
                                     v
                              [prompt misdelivery detected]
                                     |
                                     v
                              Blocked (or ReadyForPrompt if auto-recover enabled)
```

**Worker Failure Kinds**:
- Trust gate: the worker is blocked on a trust/permission prompt
- Prompt delivery: the prompt was delivered to the wrong target (e.g., a shell instead of the agent)
- Protocol: generic communication failure

**Trust Resolution**:
- Workers can be configured with trusted directory roots. If the worker's working directory falls under a trusted root, trust prompts are auto-resolved.
- Otherwise, trust must be resolved manually via the registry API.

**Prompt Misdelivery Detection and Recovery**:
- The system detects when a prompt lands in a shell (by observing shell error patterns like "command not found").
- If auto-recovery is enabled, the prompt is saved for replay and the worker returns to ready state.
- If auto-recovery is disabled, the worker enters a blocked state.

### Error Handling

- Worker not found: error returned from all operations taking a worker ID.
- Invalid state transitions: operations validate preconditions (e.g., send_prompt requires ReadyForPrompt status).
- Thread safety: the registry uses interior mutability with mutex protection. Lock poisoning is treated as a panic.

### Extension Points

- Detection heuristics (trust prompts, ready cues, running cues, shell prompts) are configurable pattern lists.
- The registry is an in-memory data structure. For synth-panel, this is sufficient -- persistent orchestration state is handled by the session persistence layer.

---

## 5. Structured Output

### Purpose

Ensure that LLM responses conform to a declared schema, enabling reliable machine processing of panelist feedback.

### Interface Contract

The structured output system provides:

1. **Schema declaration**: Define the expected shape of a response as a JSON Schema.
2. **Response formatting**: When structured output mode is enabled, the engine wraps LLM output in valid JSON matching the declared schema.
3. **Validation with retry**: If the LLM produces output that fails schema validation, the system retries up to a configurable limit.

### Data Model

**Structured Output Configuration**:
- Enabled flag (boolean)
- Retry limit (integer, default 2)
- Schema definition (JSON Schema object)

**Tool Definition** (also used for structured output via tool-use forcing):
- Tool name (string)
- Description (optional string)
- Input schema (JSON Schema object defining required parameters and types)

**Tool Choice Constraint**:
- Auto: the LLM decides whether to use tools
- Any: the LLM must use at least one tool
- Specific: the LLM must use a named tool

### Behavioral Requirements

For synth-panel, the primary pattern is:

1. Define a "respond" tool whose input schema matches the desired panelist response format.
2. Set tool_choice to "specific" with the respond tool's name.
3. The LLM is forced to produce a tool invocation whose input is valid JSON conforming to the schema.
4. The runtime extracts the JSON input from the tool invocation block as the structured response.

If the LLM produces malformed JSON:
- The system catches the deserialization error.
- It retries the request (up to the configured retry limit).
- On final failure, it falls back to a minimal valid response with an error flag.

### Error Handling

- Schema validation failure triggers retry (up to limit).
- Deserialization errors (malformed JSON) trigger retry.
- After retries exhausted, the system produces a fallback payload containing the error description.

### Extension Points

- Schemas are passed as data, not compiled into the system. Any valid JSON Schema can be used.
- The tool-use forcing pattern can be used for any structured extraction, not just panelist responses.

---

## 6. Session Persistence

### Purpose

Save and restore agent conversation state so that sessions survive process restarts, can be resumed later, or can be forked into branches.

### Interface Contract

**Save session**: Serialize a session to a file at a specified path.
**Load session**: Deserialize a session from a file path, returning a fully hydrated session object.
**Append message**: Persist a single new message incrementally (without rewriting the full session).
**Fork session**: Create a new session that inherits all messages from a parent, with its own unique ID and optional branch name.

### Data Model

**Persisted Session Format**:

The system supports two serialization formats:

1. **Single JSON object**: A complete snapshot containing version, session ID, timestamps, messages array, and optional compaction/fork metadata.
2. **JSONL (newline-delimited JSON)**: Each line is a typed record:
   - `session_meta` record: version, session ID, timestamps, fork info
   - `message` records: one per conversation message
   - `compaction` records: compaction metadata

The loader auto-detects the format by attempting JSON parse first, then falling back to JSONL.

**File Rotation**:
- When a session file exceeds a size threshold (default 256KB), the existing file is rotated (renamed with a numeric suffix).
- A maximum number of rotated files are kept (default 3). Older rotations are deleted.
- Writes are atomic (write to temp file, then rename) to prevent corruption on crash.

**Session Store** (simplified variant for lightweight use):
- Session ID (string)
- Ordered list of message texts
- Cumulative input/output token counts
- Serialized as a single JSON file in a configurable directory

### Lifecycle

```
[New Session] --> push_message --> push_message --> ... --> save_to_path
                                                             |
                                                             v
                                                      [File on disk]
                                                             |
                                                             v
                                                      load_from_path --> [Resumed Session]
                                                             |
                                                             v
                                                         fork() --> [New Session with parent link]
```

**Compaction**: When a session accumulates too many messages, older messages can be replaced with a summary. The compaction metadata records: how many compactions have occurred, how many messages were removed, and the summary text.

### Error Handling

- I/O errors during save/load propagate as session errors.
- Malformed JSON/JSONL produces a format error with line number context.
- Missing required fields produce descriptive format errors.
- Failed atomic writes leave the previous version intact.

### Extension Points

- The persistence path is configurable per session.
- The serialization format is extensible (new record types in JSONL can be added without breaking existing readers).
- The session store directory is configurable.

---

## 7. Cost and Budget Tracking

### Purpose

Track token consumption and estimated dollar costs across all LLM calls. Enforce budget limits to prevent runaway spending during focus group simulations.

### Interface Contract

**Usage Tracker**: Accumulates token usage across multiple turns.

Operations:
- Record a turn's usage (input, output, cache write, cache read tokens)
- Query current turn usage
- Query cumulative usage across all turns
- Query turn count
- Reconstruct tracker state from a persisted session (by replaying usage from each message)

**Cost Estimation**: Convert token counts to estimated USD costs.

Operations:
- Estimate cost for a given token usage with default pricing
- Estimate cost for a given token usage with model-specific pricing
- Format a cost as a USD string (4 decimal places)
- Produce human-readable summary lines showing token breakdown and cost

**Budget Enforcement** (in the query engine / session loop):
- Maximum token budget (configurable, default 2000 tokens for lightweight use)
- When projected cumulative usage exceeds the budget, the turn completes with a "max_budget_reached" stop reason instead of continuing.

### Data Model

**Token Usage** (per-turn snapshot):
- Input tokens (unsigned integer)
- Output tokens (unsigned integer)
- Cache creation input tokens (unsigned integer)
- Cache read input tokens (unsigned integer)
- Derived: total tokens = sum of all four

**Model Pricing** (per-million-token rates in USD):
- Input cost per million tokens
- Output cost per million tokens
- Cache creation cost per million tokens
- Cache read cost per million tokens

**Known Model Pricing Tiers**:

| Model Family | Input | Output | Cache Write | Cache Read |
|-------------|-------|--------|-------------|------------|
| Haiku tier | $1.00 | $5.00 | $1.25 | $0.10 |
| Sonnet tier (default) | $15.00 | $75.00 | $18.75 | $1.50 |
| Opus tier | $15.00 | $75.00 | $18.75 | $1.50 |

When a model does not match any known tier, the default (Sonnet) pricing is used and the output is annotated with "pricing=estimated-default".

**Cost Estimate** (per-turn or cumulative):
- Input cost USD
- Output cost USD
- Cache creation cost USD
- Cache read cost USD
- Derived: total cost = sum of all four

### Summary Output Format

Cost summary produces two lines:
```
<label>: total_tokens=N input=N output=N cache_write=N cache_read=N estimated_cost=$X.XXXX [model=...] [pricing=estimated-default]
  cost breakdown: input=$X.XXXX output=$X.XXXX cache_write=$X.XXXX cache_read=$X.XXXX
```

### Error Handling

- Budget exceeded is a soft stop (the current turn completes, but further turns are prevented).
- Unknown model pricing produces a warning annotation but does not fail.
- Token counts are unsigned; overflow is not expected in practice.

### Extension Points

- New model pricing tiers are added to the pricing lookup table.
- Budget enforcement thresholds are configurable per session or per command invocation.
- The tracker can be initialized from a persisted session to resume accurate cost accounting.

---

## 8. CLI Framework

### Purpose

Provide the command-line interface through which operators configure and run focus group simulations.

### Interface Contract

**Top-Level Arguments** (global, apply to all commands):
- `--model <model>`: LLM model to use (default: a sensible default like the best available model)
- `--permission-mode <mode>`: Control what the agent is allowed to do. Values: read-only, workspace-write, full-access (default: full-access)
- `--config <path>`: Path to a configuration file
- `--output-format <format>`: Output format. Values: text, json, ndjson (default: text)

**Subcommands**:
- `prompt <text...>`: Run a single non-interactive prompt and exit. All remaining arguments are joined as the prompt text.
- `login`: Start an authentication flow (e.g., OAuth).
- `logout`: Clear saved authentication credentials.
- (Additional subcommands as needed for synth-panel: `panel run`, `panel resume`, etc.)

**Interactive Mode** (when no subcommand is given):
- Enter a REPL loop.
- The prompt character is a stylized indicator (e.g., a right-pointing arrow).
- User input is submitted on Enter. Multi-line input via Shift+Enter or Ctrl+J.
- Ctrl+C cancels the current input. Ctrl+D (or equivalent) exits.

**Slash Commands** (available in interactive mode):
- `/help`: List available commands with summaries.
- `/status`: Show current session state (turn count, model, permission mode, output format, last usage).
- `/compact`: Compact session history into a summary, freeing context window space.
- `/model [name]`: Show or switch the active model.
- `/permissions [mode]`: Show or switch the permission mode.
- `/config [section]`: Inspect configuration.
- `/memory`: Show loaded instruction/memory files.
- `/clear [--confirm]`: Start a fresh session (requires --confirm flag).
- Unknown slash commands produce an error message but do not crash.

### Session State (maintained during interactive mode)

- Turn counter
- Compacted message count
- Last model used
- Last turn's usage summary (input/output tokens)

### Output Formatting

| Format | Behavior |
|--------|----------|
| text | Human-readable: streaming markdown for responses, token usage footer |
| json | Single JSON object per response: `{"message": "...", "usage": {"input_tokens": N, "output_tokens": N}}` |
| ndjson | One JSON object per line per event: `{"type": "message", "text": "...", "usage": {...}}` |

### Streaming UX

During response streaming:
- Show a spinner while waiting for the first token.
- Switch to streaming text output once tokens arrive.
- Show a separate spinner for each tool execution.
- After the response completes, show token usage.

### Error Handling

- Invalid arguments produce a usage error and exit with a non-zero code.
- Authentication errors produce a descriptive message directing the user to `login`.
- LLM errors during interactive mode are displayed and the REPL continues (does not crash).

### Extension Points

- New subcommands are added by defining a new variant and handler.
- Slash commands are registered in a static table with a name, parser, and handler function. New commands require only a new entry.
- Output formatting is dispatched on the format enum; new formats require a new variant and renderer.

---

## 9. Plugin and Extension System

### Purpose

Allow third-party extensions to modify agent behavior by intercepting tool use, adding new tools, and running lifecycle scripts.

### Interface Contract

**Plugin Manager**: Discovers, installs, enables/disables, and uninstalls plugins.

Operations:
- Install a plugin from a source directory (copies/links the plugin into the config directory)
- List installed plugins
- Enable/disable a plugin
- Uninstall a plugin
- Build a plugin registry (snapshot of all enabled plugins and their capabilities)

**Plugin Registry**: Read-only snapshot of enabled plugins for runtime use.

Operations:
- List all enabled plugins with metadata
- Aggregate hooks from all enabled plugins into a single hook set

**Hook Runner**: Executes hook commands at defined interception points.

Operations:
- Run pre-tool-use hooks (before a tool executes)
- Run post-tool-use hooks (after a tool executes successfully)
- Run post-tool-use-failure hooks (after a tool execution fails)

### Data Model

**Plugin Manifest** (declared in a well-known file within the plugin directory):
- Name (string)
- Version (semver string)
- Description (string)
- Required permissions (list: read, write, execute)
- Default enabled flag
- Hook declarations (mapping from hook event to list of command strings)
- Lifecycle declarations (init and shutdown command lists)
- Tool declarations (list of tool manifests)
- Command declarations (list of command manifests)

**Plugin Metadata** (runtime representation):
- Unique plugin identifier
- Name, version, description
- Kind: builtin, bundled, or external
- Source path
- Default enabled flag
- Root directory

**Plugin Hooks**:
- Pre-tool-use commands (list of shell command strings)
- Post-tool-use commands (list of shell command strings)
- Post-tool-use-failure commands (list of shell command strings)

When multiple plugins are enabled, hooks are aggregated: all pre-tool-use commands from all plugins run in order, all post-tool-use commands run in order, etc.

**Hook Execution Protocol**:

Each hook command is a shell command that receives context via:
- **Environment variables**: HOOK_EVENT (event name), HOOK_TOOL_NAME, HOOK_TOOL_INPUT, HOOK_TOOL_OUTPUT (if post-hook), HOOK_TOOL_IS_ERROR ("0" or "1")
- **Standard input**: A JSON payload containing all of the above plus parsed tool input

**Hook command exit codes**:
- 0: Allow (tool execution proceeds; stdout is captured as a message)
- 2: Deny (tool execution is blocked; stdout is used as the denial reason)
- Any other non-zero: Failure (hook itself failed; subsequent hooks in the chain are skipped)
- Terminated by signal: Failure

**Hook Run Result**:
- Denied flag
- Failed flag
- Messages (list of strings captured from hook stdout)

Hook evaluation is short-circuiting: if any hook in the chain denies or fails, subsequent hooks do not run.

**Lifecycle Commands**:
- Init commands run when the plugin is first loaded.
- Shutdown commands run when the session ends.

### Error Handling

- Plugin manifest not found or malformed: installation fails with a descriptive error.
- Hook command fails to start (e.g., executable not found): treated as a hook failure, not a crash.
- Hook command timeout: not currently specified; hooks run synchronously.
- Plugin not found for enable/disable/uninstall: error returned.

### Extension Points

- New hook events can be added by extending the hook event enum and adding corresponding command lists to the manifest format.
- Plugin tools and commands are declared in the manifest and integrated into the runtime's tool/command registry.
- The plugin discovery mechanism (config directory layout) is a convention, not hardcoded. Alternative registries could be implemented.

---

## 10. Cross-Cutting Concerns

### Configuration Hierarchy

Configuration flows from multiple sources, in priority order (highest wins):
1. CLI arguments
2. Environment variables
3. Configuration file
4. Defaults

### Telemetry and Observability

The runtime supports an optional session tracer that records:
- Turn start/completion/failure events
- Tool execution start/finish events
- Iteration counts within turns
- Token usage per turn

Telemetry is opt-in and does not affect behavior.

### Thread Safety

All registries (worker, team, cron) use interior mutability with mutex guards. They are safe to share across threads. Lock poisoning is treated as unrecoverable.

### Naming Conventions for Synth-Panel

When implementing these foundations for synth-panel, use domain-appropriate names:

| Foundation Concept | Synth-Panel Term |
|-------------------|-----------------|
| Worker | Panelist |
| Team | Panel |
| Session | Interview |
| Turn | Exchange |
| Tool | Instrument |
| Hook | Interceptor |
| Prompt | Stimulus |
| Response | Reaction |

### Minimum Viable Subset

For a first working version of synth-panel, the following components are required in this order:

1. **LLM Client Abstraction** -- needed for everything
2. **Structured Output** -- panelist responses must be schema-conformant
3. **Cost and Budget Tracking** -- prevent runaway costs
4. **Session Persistence** -- save panel results
5. **Agent Runtime / Session Loop** -- single-panelist execution
6. **CLI Framework** -- operator interface
7. **Multi-Agent Orchestration** -- parallel panelist execution
8. **Plugin and Extension System** -- can be deferred to v2

---

## 11. Acceptance Tests (Live Validation)

Each component must be validated against a live LLM API (Claude or OpenAI-compatible). Tests should be runnable scripts in `tests/` that exercise real API calls and assert behavioral correctness.

### LLM Client Abstraction
- Send a simple prompt ("Say hello"), receive a text response containing at least one word.
- Stream the same prompt, collect all stream events, verify: message_start → content_block_start → content_block_delta(s) → content_block_stop → message_delta → message_stop.
- Send with an invalid API key, verify authentication error (not retried).
- Send with an unreachable base URL, verify transport error with retry attempts.
- Model alias resolution: "sonnet" resolves to a valid canonical model ID.

### Structured Output
- Define a schema: `{"name": string, "sentiment": "positive"|"negative"|"neutral", "confidence": number}`.
- Force the LLM to respond via tool-use with that schema. Verify the response is valid JSON matching the schema.
- Send a deliberately ambiguous prompt and verify retry on malformed output (if it occurs).

### Cost/Budget Tracking
- After a successful LLM call, verify token usage has non-zero input and output counts.
- Verify cost estimation produces a non-zero dollar amount.
- Set a budget of 100 tokens. Make a call. Verify budget enforcement triggers on a subsequent call.

### Session Persistence
- Run a turn, save the session, load it back, verify message count matches.
- Verify JSONL append mode adds a single line (not full rewrite).
- Fork a session, verify the fork has a parent link and all original messages.

### Agent Runtime
- Run a single turn with a prompt like "What is 2+2?". Verify the turn completes with a text response.
- Verify cumulative usage is tracked across multiple turns.
- Verify auto-compaction triggers after exceeding the token threshold (can be set low for testing, e.g., 1000 tokens).

### CLI Framework
- `synth-panel prompt "Say hello"` exits 0 and prints a response.
- `synth-panel prompt "Say hello" --output-format json` outputs valid JSON with message and usage fields.
- `synth-panel --help` prints usage information.
- Invalid subcommand exits non-zero with an error message.

### Integration (end-to-end)
- Define 3 personas in YAML (e.g., "skeptical CTO", "enthusiastic intern", "pragmatic PM").
- Define a survey instrument asking "What do you think of the name 'Traitprint' for a career matching app?"
- Run `synth-panel run --personas personas.yaml --instrument survey.yaml --model sonnet`.
- Verify: 3 structured responses returned, each conforming to the schema, each with different content reflecting the persona, total cost printed.

---

*End of specification.*
