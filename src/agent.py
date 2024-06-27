from llama_index.core.agent import ReActAgent
from models import llm
from tools import tools

# Create a ReActAgent instance using the provided tools and LLM
agent = ReActAgent.from_tools(
    tools,  # List of tools to be used by the agent
    verbose=False,  # Set verbose to False for less detailed logging
    llm=llm,  # The language model instance to be used by the agent
)
