import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_agentchat.conditions import TextMentionTermination
from autogen_core.models import ModelInfo
from autogen_agentchat.ui import Console
from autogen_core.models import ModelFamily
from autogen_core.tools import FunctionTool

import os

async def generate_md(content:str, file_name:str):
    """
    Asynchronously generates a markdown file with the given content.
    Args:
        content (str): The content to be written to the markdown file.
        file_name (str): The name of the markdown file to be created.
    Writes:
        A markdown file at the specified file path containing the provided content.
    """
    with open(file_name, "w") as f:
        f.write(content)

generate_md_tool = FunctionTool(
    generate_md, description="A tool that generates a markdown file from an input string."
)


agent = AssistantAgent(
    name="AssistantAgent",
    tools=[generate_md_tool],
    model_client=OpenAIChatCompletionClient(
        model="meta-llama/llama-4-scout",
        api_key=os.environ['OPENROUTER_API_KEY'],
        base_url="https://openrouter.ai/api/v1",
        model_info=ModelInfo(vision=False, function_calling=True, json_output=True, family=ModelFamily.UNKNOWN),
    ),
    description="Utilize surfer agent to research and generate a report based on a given topic",
    system_message="You are a helpful assistant. Your task is to research and synthesize data extracted into a high quality literature review including CORRECT references. You MUST write a final report that is formatted as a literature review with CORRECT references.  Your response should end with the word 'TERMINATE'",
)

surfer = MultimodalWebSurfer(
    "WebSurfer",
    model_client=OpenAIChatCompletionClient(
        model="openai/gpt-4o-mini",
        api_key=os.environ['OPENROUTER_API_KEY'],
        base_url="https://openrouter.ai/api/v1",
        model_info=ModelInfo(vision=False, function_calling=True, json_output=True, family=ModelFamily.GPT_4O),
    ),
)

agent_team = SelectorGroupChat([agent, surfer], 
                               model_client=OpenAIChatCompletionClient(
                                   model="meta-llama/llama-4-scout",
                                   api_key=os.environ['OPENROUTER_API_KEY'],
                                   base_url="https://openrouter.ai/api/v1",
                                   model_info=ModelInfo(vision=False, function_calling=True, json_output=True, family=ModelFamily.UNKNOWN),
                               ),                               
                               termination_condition=TextMentionTermination("TERMINATE"))

config = agent_team.dump_component()

with open("/workspaces/multiagent-demo/src/ag_code_example_config.json", "w") as file:
    file.write(config.model_dump_json())

async def main():
    await Console(agent_team.run_stream(task = 
        """Search the web and summarize the current state of GLP-1 drug development.
          Summarize the findings with references and images in a markdown formatted report and save to disk.
          """
        ))
    await websurfer_agent.close()

asyncio.run(main())

# is there a ValueError from the MultomodalWebSurfer? See: https://github.com/microsoft/autogen/issues/6084