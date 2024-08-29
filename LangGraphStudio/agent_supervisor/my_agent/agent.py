import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add the parent directory to the sys.path

from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file

from typing import TypedDict, Literal, Sequence, Annotated
import functools
import operator

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI  # Import the language model
from my_agent.utils.nodes import agent_node, create_agent, create_supervisor_chain
from my_agent.utils.state import AgentState
from my_agent.utils.tools import tavily_tool, python_repl_tool

# Initialize the language model
llm = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0.1
)

    
# Define the config
class SupervisorConfig(TypedDict):
    model_name: Literal["azure_openai"]

# Define a new graph
workflow = StateGraph(AgentState, config_schema=SupervisorConfig)

# Create the agents
research_agent = create_agent(llm, [tavily_tool], "You are a web researcher.")
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

code_agent = create_agent(
    llm,
    [python_repl_tool],
    "You may generate safe Python code to analyze data and generate charts using matplotlib.",
)
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

# Create the supervisor chain
members = ["Researcher", "Coder"]
supervisor_chain = create_supervisor_chain(llm, members)

# Add nodes
workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("supervisor", supervisor_chain)

# Add edges
for member in members:
    workflow.add_edge(member, "supervisor")

# Define conditional routing from supervisor
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

# Add the entry point
workflow.add_edge(START, "supervisor")

# Compile the graph
graph = workflow.compile()
