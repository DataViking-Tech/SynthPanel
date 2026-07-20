"""Althing + OpenAI Agents SDK.

Run a synthetic focus group from an OpenAI agent via Althing's MCP server.
The Agents SDK has built-in MCPServerStdio support, so no wrapper code is needed:
the agent auto-discovers all 12 Althing tools.

Install:
    pip install openai-agents althing[mcp]

Run:
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-...   # Althing's default provider
    python openai_agents.py
"""

import asyncio

from agents import Agent, Runner
from agents.mcp import MCPServerStdio


async def main() -> None:
    async with MCPServerStdio(
        name="althing",
        params={"command": "althing", "args": ["mcp-serve"]},
    ) as althing:
        agent = Agent(
            name="Researcher",
            instructions="You run synthetic focus groups via the althing tools.",
            mcp_servers=[althing],
        )
        result = await Runner.run(
            agent,
            "Use run_quick_poll to ask three PM personas: 'What would make you pay for a synthetic research tool?'",
        )
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
