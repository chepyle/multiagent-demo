# from https://multiagentbook.com/labs/usecases/?usecase=mcp-tool-agent
# pip install -U autogen-ext[mcp] json-schema-to-pydantic>=0.2.2
# uv tool install mcp-server-fetch
# verify it in path by running uv tool update-shell
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_core import CancellationToken
from autogen_core.models import ModelInfo, ModelFamily
from autogen_agentchat.ui import Console
import os

async def main() -> None:
    # Setup server params for local filesystem access
    fetch_mcp_server = StdioServerParams(command="uvx", args=["mcp-server-fetch"])
    tools = await mcp_server_tools(fetch_mcp_server)

    # Create an agent that can use the fetch tool.    
    model_client=OpenAIChatCompletionClient(
        model="openai/gpt-4o-mini",
        api_key=os.environ['OPENROUTER_API_KEY'],
        base_url="https://openrouter.ai/api/v1",
        model_info=ModelInfo(vision=False, function_calling=True, json_output=True, family=ModelFamily.GPT_4O),
    )
    agent = AssistantAgent(name="fetcher", model_client=model_client, tools=tools, reflect_on_tool_use=True)  # type: ignore 

    termination = MaxMessageTermination(
        max_messages=5) | TextMentionTermination("TERMINATE")

    team = RoundRobinGroupChat([agent], termination_condition=termination)
    #  team.dump_component().model_dump()
 
    await Console(team.run_stream(task="Summarize the content of https://www.fiercepharma.com/", cancellation_token=CancellationToken()))
    
if __name__ == "__main__":
    asyncio.run(main())