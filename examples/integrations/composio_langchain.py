"""SynthPanel + Composio + LangChain.

Register SynthPanel as a Composio toolkit, then hand the resulting tools
to a LangChain agent. No MCP bridge, no subprocess — Composio calls the
SynthPanel SDK in-process.

Install:
    pip install composio composio_langchain langchain langchain-anthropic synthpanel

Run:
    export ANTHROPIC_API_KEY=sk-...
    export COMPOSIO_API_KEY=...            # required by Composio
    python composio_langchain.py
"""

from composio import Composio
from composio_langchain import LangchainProvider
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from synth_panel.integrations.composio import synthpanel_toolkit


def main() -> None:
    composio = Composio(provider=LangchainProvider())
    toolkit = synthpanel_toolkit(composio)

    session = composio.create(
        user_id="researcher_1",
        experimental={"custom_toolkits": [toolkit]},
    )
    tools = session.tools()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a market research assistant. Use SynthPanel tools to poll personas."),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    llm = ChatAnthropic(model="claude-haiku-4-5")
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=tools)

    result = executor.invoke(
        {
            "input": (
                "Run quick_poll with the general-consumer persona pack on: "
                "'Would a $49/mo AI bookkeeper replace your spreadsheet?' "
                "Summarise the synthesis recommendation in one sentence."
            )
        }
    )
    print(result["output"])


if __name__ == "__main__":
    main()
