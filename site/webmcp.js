// WebMCP browser registration for synthpanel.dev.
//
// Implements the WebMCP browser API surface
// (https://webmachinelearning.github.io/webmcp/): when the page is
// loaded inside a browser/agent that exposes
// ``navigator.modelContext.provideContext``, this module advertises
// the four primary synthpanel tools so the host agent can discover
// them without an MCP stdio connection.
//
// Tool surface mirrors ``site/.well-known/mcp/server-card.json`` and
// the canonical Python MCP server in ``src/synth_panel/mcp/server.py``.
// Names, descriptions, and input schemas track the four headline
// tools: ``run_panel``, ``run_quick_poll``, ``run_prompt``, and
// ``extend_panel``.
//
// There is no public synthpanel HTTP API at synthpanel.dev — the site
// is a static Cloudflare Pages deployment. Each tool's ``execute``
// callback therefore returns a typed error envelope
// (``{ ok: false, error: 'browser-bridge-not-available' }``) so host
// agents see a stable, machine-readable response rather than an
// unhandled exception. When a future bridge endpoint ships, only the
// ``execute`` callbacks need to change; the schemas stay stable.

const TOOLS = [
  {
    name: "run_prompt",
    description:
      "Send a single prompt to an LLM and get a response. No personas required. The simplest tool — ask a quick research question without constructing personas or running a full panel.",
    inputSchema: {
      type: "object",
      properties: {
        prompt: {
          type: "string",
          description: "The question or prompt to send.",
        },
        model: {
          type: "string",
          description:
            "LLM model to use (e.g. 'sonnet', 'haiku', 'gpt-4o', 'gemini-2.5-flash'). Defaults to haiku.",
        },
        temperature: {
          type: "number",
          description: "Sampling temperature (0.0-1.0). Controls randomness.",
        },
        top_p: {
          type: "number",
          description:
            "Nucleus sampling threshold (0.0-1.0). Alternative to temperature.",
        },
        use_sampling: {
          type: "boolean",
          description:
            "Explicit mode override. true forces host sampling, false forces BYOK.",
        },
      },
      required: ["prompt"],
    },
  },
  {
    name: "run_panel",
    description:
      "Run a full synthetic focus group panel. Each persona answers all questions independently in parallel; a synthesis step then aggregates findings into themes, agreements, disagreements, and recommendations. Supports v1/v2 flat instruments and v3 branching instruments.",
    inputSchema: {
      type: "object",
      properties: {
        questions: {
          type: "array",
          description:
            "Flat list of question dicts (v1-equivalent). Each should have a 'text' key.",
          items: { type: "object" },
        },
        personas: {
          type: "array",
          description:
            "Inline persona definitions. Each persona object accepts 'name' (required), plus optional 'age', 'occupation', 'background', 'personality_traits'.",
          items: { type: "object" },
        },
        pack_id: {
          type: "string",
          description:
            "ID of a saved persona pack. Merged with inline personas (inline first).",
        },
        instrument: {
          type: "object",
          description:
            "Raw instrument body (the value under the top-level 'instrument:' key in YAML). Takes precedence over 'questions'.",
        },
        instrument_pack: {
          type: "string",
          description:
            "Name of an installed instrument pack. Takes precedence over 'instrument' and 'questions'.",
        },
        model: {
          type: "string",
          description: "LLM model to use. Defaults to haiku.",
        },
        response_schema: {
          type: "object",
          description:
            "Optional JSON Schema for structured output per panelist response.",
        },
        synthesis: {
          type: "boolean",
          description:
            "Whether to run synthesis after collecting responses. Defaults to true.",
        },
        synthesis_model: {
          type: "string",
          description: "Model to use for synthesis. Defaults to panelist model.",
        },
        synthesis_prompt: {
          type: "string",
          description: "Custom synthesis prompt. Replaces the default.",
        },
        temperature: { type: "number" },
        top_p: { type: "number" },
        persona_models: {
          type: "object",
          description:
            "Per-persona model overrides: maps persona name to model alias.",
        },
        extract_schema: {
          description:
            "Schema for post-hoc structured extraction. Built-in name ('sentiment', 'themes', 'rating') or inline JSON Schema.",
        },
        models: {
          type: "array",
          description:
            "List of model names for multi-model ensemble (length >= 2 enables ensemble mode).",
          items: { type: "string" },
        },
        synthesis_temperature: { type: "number" },
        variants: {
          type: "integer",
          description: "Number of variant runs for sampling-based aggregation.",
        },
        use_sampling: { type: "boolean" },
        decision_being_informed: {
          type: "string",
          description:
            "v1.0.0 contract field — the decision this panel will inform (12-280 chars, single line).",
        },
      },
    },
  },
  {
    name: "run_quick_poll",
    description:
      "Quick single-question poll across personas. A simplified version of run_panel for quick feedback on one question. Includes synthesis by default. Defaults to a built-in diverse persona pack when none is provided.",
    inputSchema: {
      type: "object",
      properties: {
        question: {
          type: "string",
          description: "The question to ask all personas.",
        },
        personas: {
          type: "array",
          description:
            "List of persona definitions. Optional — when omitted, a built-in pack of diverse personas is used.",
          items: { type: "object" },
        },
        model: {
          type: "string",
          description: "LLM model to use. Defaults to haiku.",
        },
        response_schema: { type: "object" },
        synthesis: { type: "boolean" },
        synthesis_model: { type: "string" },
        synthesis_prompt: { type: "string" },
        temperature: { type: "number" },
        top_p: { type: "number" },
        use_sampling: { type: "boolean" },
        decision_being_informed: {
          type: "string",
          description:
            "v1.0.0 contract field — the decision this poll will inform (12-280 chars, single line).",
        },
      },
      required: ["question"],
    },
  },
  {
    name: "extend_panel",
    description:
      "Append one ad-hoc round to a saved panel result. Reuses each panelist's saved session so the follow-up sees full conversational context. NOT a re-entry into the v3 DAG — for adaptive branching, run a fresh run_panel against a v3 instrument instead.",
    inputSchema: {
      type: "object",
      properties: {
        result_id: {
          type: "string",
          description: "ID of a previously saved panel result.",
        },
        questions: {
          type: "array",
          description:
            "One or more questions for the ad-hoc round. They run as a single round, in order, against the same personas as the original run.",
          items: { type: "object" },
        },
        model: {
          type: "string",
          description: "LLM model to use for the new round. Defaults to haiku.",
        },
        synthesis: { type: "boolean" },
        synthesis_model: { type: "string" },
        synthesis_prompt: { type: "string" },
        decision_being_informed: {
          type: "string",
          description:
            "v1.0.0 contract field — the decision this extension informs (12-280 chars, single line).",
        },
      },
      required: ["result_id", "questions"],
    },
  },
];

const BRIDGE_UNAVAILABLE = Object.freeze({
  ok: false,
  error: "browser-bridge-not-available",
  message:
    "synthpanel.dev does not yet expose a public HTTP bridge for these tools. Run the synthpanel MCP server locally (`pip install synthpanel[mcp]` then add the stdio config to your MCP host) for live tool execution.",
  docs: "https://synthpanel.dev/mcp",
});

function makeExecute(name) {
  return async ({ input } = {}) => {
    return { ...BRIDGE_UNAVAILABLE, tool: name, input: input ?? null };
  };
}

export async function registerWebMCP() {
  const ctx =
    typeof navigator !== "undefined" ? navigator.modelContext : undefined;
  if (!ctx || typeof ctx.provideContext !== "function") {
    return { registered: false, reason: "webmcp-unsupported" };
  }
  const tools = TOOLS.map((tool) => ({
    name: tool.name,
    description: tool.description,
    inputSchema: tool.inputSchema,
    execute: makeExecute(tool.name),
  }));
  try {
    await ctx.provideContext({ tools });
    return { registered: true, tools: tools.map((t) => t.name) };
  } catch (err) {
    return {
      registered: false,
      reason: "provideContext-failed",
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

export const webmcpTools = TOOLS;
