"""SynthPanel + LlamaIndex.

Expose SynthPanel's MCP tools to a LlamaIndex FunctionAgent via the
llama-index-tools-mcp bridge. All 12 tools become callable with no wrapper code.

Install:
    pip install llama-index-tools-mcp llama-index-llms-anthropic synthpanel[mcp]

Run:
    export ANTHROPIC_API_KEY=sk-...
    python llamaindex_tool.py
"""

import asyncio

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.anthropic import Anthropic
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec


async def main() -> None:
    mcp_client = BasicMCPClient("synthpanel", args=["mcp-serve"])
    tool_spec = McpToolSpec(client=mcp_client)
    tools = await tool_spec.to_tool_list_async()

    agent = FunctionAgent(
        tools=tools,
        llm=Anthropic(model="claude-haiku-4-5"),
        system_prompt="You run synthetic focus groups via the synthpanel tools.",
    )
    response = await agent.run(
        "Run a quick_poll with three designer personas on: 'Would you pay $19/mo for an AI design critique tool?'"
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
