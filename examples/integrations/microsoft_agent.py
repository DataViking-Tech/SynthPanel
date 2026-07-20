"""Althing + Microsoft Agent Framework 1.0.

Connect the Microsoft Agent Framework's built-in MCP client to Althing's
stdio server. The framework shipped MCP support in version 1.0 (April 2026).

Install:
    pip install agent-framework althing[mcp]

Run:
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-...
    python microsoft_agent.py
"""

import asyncio

from agent_framework import ChatAgent, MCPStdioTool
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    async with MCPStdioTool(
        name="althing",
        command="althing",
        args=["mcp-serve"],
    ) as althing:
        agent = ChatAgent(
            chat_client=OpenAIChatClient(model_id="gpt-4.1-mini"),
            instructions="Run synthetic focus groups via the althing tools.",
            tools=althing,
        )
        result = await agent.run(
            "Use run_quick_poll with three data-scientist personas on: "
            "'Would you trust an AI code-review agent on your production PRs?'"
        )
        print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
