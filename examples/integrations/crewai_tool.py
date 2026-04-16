"""SynthPanel + CrewAI.

Attach SynthPanel's MCP server to a CrewAI Agent via MCPServerAdapter. The
agent gets all 12 synthpanel tools as native CrewAI tools — no wrapper code.

Install:
    pip install "crewai-tools[mcp]" crewai synthpanel[mcp]

Run:
    export ANTHROPIC_API_KEY=sk-...
    python crewai_tool.py
"""

from crewai import Agent, Crew, Task
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters

server_params = StdioServerParameters(command="synthpanel", args=["mcp-serve"])

with MCPServerAdapter(server_params) as synthpanel_tools:
    researcher = Agent(
        role="Synthetic Research Lead",
        goal="Answer product questions via synthetic focus groups.",
        backstory="You use the synthpanel tools to poll AI personas.",
        tools=synthpanel_tools,
    )
    task = Task(
        description=(
            "Use run_quick_poll with three startup-founder personas to answer: "
            "'What would make you switch from Notion to a focused writing tool?' "
            "Return the synthesis summary."
        ),
        expected_output="A synthesis summary of the poll results.",
        agent=researcher,
    )
    crew = Crew(agents=[researcher], tasks=[task])
    print(crew.kickoff())
