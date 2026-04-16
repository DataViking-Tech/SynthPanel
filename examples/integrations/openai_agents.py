"""SynthPanel + OpenAI Agents SDK.

Run a synthetic focus group from an OpenAI agent via SynthPanel's MCP server.
The Agents SDK has built-in MCPServerStdio support, so no wrapper code is needed:
the agent auto-discovers all 12 SynthPanel tools.

Install:
    pip install openai-agents synthpanel[mcp]

Run:
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-...   # SynthPanel's default provider
    python openai_agents.py
"""

import asyncio

from agents import Agent, Runner
from agents.mcp import MCPServerStdio


async def main() -> None:
    async with MCPServerStdio(
        name="synthpanel",
        params={"command": "synthpanel", "args": ["mcp-serve"]},
    ) as synthpanel:
        agent = Agent(
            name="Researcher",
            instructions="You run synthetic focus groups via the synthpanel tools.",
            mcp_servers=[synthpanel],
        )
        result = await Runner.run(
            agent,
            "Use run_quick_poll to ask three PM personas: 'What would make you pay for a synthetic research tool?'",
        )
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
