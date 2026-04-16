"""SynthPanel + LangChain / LangGraph.

Bridge SynthPanel's MCP tools into LangChain via langchain-mcp-adapters, then
hand them to a LangGraph ReAct agent. No custom tool wrappers required.

Install:
    pip install langchain-mcp-adapters langgraph langchain-anthropic synthpanel[mcp]

Run:
    export ANTHROPIC_API_KEY=sk-...
    python langchain_tool.py
"""

import asyncio

from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent


async def main() -> None:
    client = MultiServerMCPClient(
        {
            "synthpanel": {
                "command": "synthpanel",
                "args": ["mcp-serve"],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
    agent = create_react_agent(ChatAnthropic(model="claude-haiku-4-5"), tools)
    result = await agent.ainvoke(
        {
            "messages": [
                (
                    "user",
                    "Run run_quick_poll with three small-business-owner personas on: "
                    "'Would a $49/mo AI bookkeeper replace your spreadsheet?'",
                )
            ]
        }
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
