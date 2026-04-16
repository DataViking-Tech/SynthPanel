# SynthPanel + Agent Frameworks — Works With X

SynthPanel ships an [MCP](https://modelcontextprotocol.io) server with 12 tools
(`run_quick_poll`, `run_panel`, `extend_panel`, and more). Because every major
agent framework now supports MCP as a first-class tool source, **SynthPanel
works with all of them today — no framework-specific wrapper package
required**. The examples in this directory prove that claim end-to-end.

Each example is 15-30 lines, runs a single `run_quick_poll` demo in under 30
seconds on a fresh install, and deliberately does **not** depend on any
SynthPanel-specific library beyond the MCP server that already ships with
`pip install synthpanel[mcp]`.

> **Prerequisites for every example:** `pip install synthpanel[mcp]` and an
> LLM API key set in your environment (`ANTHROPIC_API_KEY` by default — see
> the [Provider Configuration](../../README.md#llm-provider-support) table
> for other providers). The `synthpanel` CLI must be on your `PATH`.

| File | Framework | Bridge | Install |
|------|-----------|--------|---------|
| [openai_agents.py](openai_agents.py) | OpenAI Agents SDK | Built-in `MCPServerStdio` | `pip install openai-agents synthpanel[mcp]` |
| [llamaindex_tool.py](llamaindex_tool.py) | LlamaIndex | `llama-index-tools-mcp` | `pip install llama-index-tools-mcp llama-index-llms-anthropic synthpanel[mcp]` |
| [crewai_tool.py](crewai_tool.py) | CrewAI | `crewai-tools[mcp]` | `pip install "crewai-tools[mcp]" crewai synthpanel[mcp]` |
| [langchain_tool.py](langchain_tool.py) | LangChain / LangGraph | `langchain-mcp-adapters` | `pip install langchain-mcp-adapters langgraph langchain-anthropic synthpanel[mcp]` |
| [microsoft_agent.py](microsoft_agent.py) | Microsoft Agent Framework 1.0 | Built-in `MCPStdioTool` | `pip install agent-framework synthpanel[mcp]` |
| [n8n_workflow.json](n8n_workflow.json) | n8n | Built-in MCP Client tool | Import into n8n + install `synthpanel[mcp]` on the runner |

---

## OpenAI Agents SDK

OpenAI's Agents SDK has native MCP support via `MCPServerStdio`. Wrap
`synthpanel mcp-serve` once and every tool becomes callable.

```bash
pip install openai-agents synthpanel[mcp]
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-...   # SynthPanel's default provider
python openai_agents.py
```

See [openai_agents.py](openai_agents.py).

## LlamaIndex

`llama-index-tools-mcp` (v0.4.8+, Feb 2026) exposes any MCP server as
LlamaIndex `FunctionTool` objects. Hand them to a `FunctionAgent` and you're
done.

```bash
pip install llama-index-tools-mcp llama-index-llms-anthropic synthpanel[mcp]
export ANTHROPIC_API_KEY=sk-...
python llamaindex_tool.py
```

See [llamaindex_tool.py](llamaindex_tool.py).

## CrewAI

`crewai-tools[mcp]` provides `MCPServerAdapter`, a context manager that makes
every MCP tool look like a native CrewAI tool. Pass the list straight into the
`Agent(tools=...)` argument.

```bash
pip install "crewai-tools[mcp]" crewai synthpanel[mcp]
export ANTHROPIC_API_KEY=sk-...
python crewai_tool.py
```

See [crewai_tool.py](crewai_tool.py).

## LangChain / LangGraph

`langchain-mcp-adapters` bridges MCP tools into LangChain's `BaseTool`
interface. The example below hands them to a LangGraph prebuilt ReAct agent,
but the same `tools` list works with any LangChain agent executor.

```bash
pip install langchain-mcp-adapters langgraph langchain-anthropic synthpanel[mcp]
export ANTHROPIC_API_KEY=sk-...
python langchain_tool.py
```

See [langchain_tool.py](langchain_tool.py).

## Microsoft Agent Framework 1.0

Microsoft shipped built-in MCP clients in Agent Framework 1.0 (April 2026).
`MCPStdioTool` is the stdio-transport equivalent of OpenAI's
`MCPServerStdio`.

```bash
pip install agent-framework synthpanel[mcp]
export OPENAI_API_KEY=sk-...        # Agent Framework's default provider
export ANTHROPIC_API_KEY=sk-...     # SynthPanel's default provider
python microsoft_agent.py
```

See [microsoft_agent.py](microsoft_agent.py).

## n8n

n8n's AI Agent node has a built-in **MCP Client Tool** subnode. Import
[n8n_workflow.json](n8n_workflow.json) via **Workflows → Import from File**,
attach an Anthropic credential, and make sure `synthpanel` is on the n8n
runner's `PATH`. The workflow fires `run_quick_poll` via the agent.

```bash
# on the machine running n8n:
pip install synthpanel[mcp]
export ANTHROPIC_API_KEY=sk-...
```

---

## Why this directory exists

Every major agent framework now speaks MCP — but that fact is invisible until
someone searches for "how do I run a synthetic focus group from
&lt;framework&gt;?" and finds a working script. Each example is a
self-contained, indexable answer to that question and doubles as verification
that SynthPanel's MCP bridge works end-to-end.

If your framework is not listed here and it supports MCP, the pattern is the
same in every case: point its MCP client at `synthpanel mcp-serve` over stdio
and all 12 tools become available. If you wire up a new one, a PR adding a
sibling example here is very welcome.

For the full integration-surface analysis (what we shipped, what we skipped,
and why), see [`docs/agent-integration-landscape.md`](../../docs/agent-integration-landscape.md).
