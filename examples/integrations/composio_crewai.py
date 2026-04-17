"""SynthPanel + Composio + CrewAI.

Expose SynthPanel actions to a CrewAI agent via Composio's CrewAI
provider. The same `synthpanel_toolkit` works across Composio providers;
only the provider object changes from the LangChain example.

Install:
    pip install composio composio_crewai crewai synthpanel

Run:
    export ANTHROPIC_API_KEY=sk-...
    export COMPOSIO_API_KEY=...
    python composio_crewai.py
"""

from composio import Composio
from composio_crewai import CrewAIProvider
from crewai import Agent, Crew, Task

from synth_panel.integrations.composio import synthpanel_toolkit


def main() -> None:
    composio = Composio(provider=CrewAIProvider())
    toolkit = synthpanel_toolkit(composio)

    session = composio.create(
        user_id="researcher_1",
        experimental={"custom_toolkits": [toolkit]},
    )
    tools = session.tools()

    researcher = Agent(
        role="Market Researcher",
        goal="Surface honest panelist reactions to product concepts.",
        backstory="You probe AI personas via SynthPanel and distil the findings.",
        tools=tools,
    )
    task = Task(
        description=(
            "Run quick_poll against the general-consumer pack with the question: "
            "'Would a $49/mo AI bookkeeper replace your spreadsheet?' "
            "Return the synthesis recommendation."
        ),
        expected_output="A one-sentence recommendation grounded in the panel synthesis.",
        agent=researcher,
    )

    crew = Crew(agents=[researcher], tasks=[task])
    print(crew.kickoff())


if __name__ == "__main__":
    main()
