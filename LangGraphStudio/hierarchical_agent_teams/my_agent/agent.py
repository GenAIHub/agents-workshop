# agent.py

import os
import functools
import logging
import json
import re
import operator
from typing import List, Dict, Optional, Union, Annotated
from pathlib import Path
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from typing_extensions import TypedDict
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import END, StateGraph, START
from langchain_openai import AzureChatOpenAI

# Define the state classes
class ResearchTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: List[str]
    next: str

class DocWritingState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: str
    next: str
    current_files: str

class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str

# Load environment variables from .env file
load_dotenv()

# Set up environment variables
SCRIPT_DIR = Path(__file__).parent
REPORTS_DIR = SCRIPT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Initialize tools
tavily_tool = TavilySearchResults(max_results=5)
repl = PythonREPL()

# Tools
@tool
def scrape_webpages(urls: List[str]) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    loader = WebBaseLoader(urls, verify_ssl=False)
    try:
        docs = loader.load()
        return "\n\n".join(
            [f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>' for doc in docs])
    except Exception as e:
        logging.error(f"Error scraping webpages: {str(e)}")
        return f"Error scraping webpages: {str(e)}"

@tool
def create_outline(points: List[str], file_name: str) -> str:
    """Create and save an outline with the given points to the specified file."""
    try:
        full_path = REPORTS_DIR / file_name
        with full_path.open("w") as file:
            for i, point in enumerate(points):
                file.write(f"{i + 1}. {point}\n")
        return f"Outline saved to {full_path}"
    except Exception as e:
        logging.error(f"Error creating outline: {str(e)}")
        return f"Error creating outline: {str(e)}"

@tool
def read_document(file_name: str, start: Optional[int] = None, end: Optional[int] = None) -> str:
    """Read the specified document, optionally limiting to a range of lines."""
    try:
        full_path = REPORTS_DIR / file_name
        with full_path.open("r") as file:
            lines = file.readlines()
        if start is None:
            start = 0
        return "".join(lines[start:end])
    except Exception as e:
        logging.error(f"Error reading document: {str(e)}")
        return f"Error reading document: {str(e)}"

@tool
def write_document(content: str, file_name: str) -> str:
    """Create and save a text document with the given content."""
    try:
        full_path = REPORTS_DIR / file_name
        with full_path.open("w") as file:
            file.write(content)
        return f"Document saved to {full_path}"
    except Exception as e:
        logging.error(f"Error writing document: {str(e)}")
        return f"Error writing document: {str(e)}"

@tool
def edit_document(file_name: str, inserts: Dict[Union[int, str], str]) -> str:
    """Edit a document by inserting text at specific line numbers."""
    try:
        full_path = REPORTS_DIR / file_name
        if not full_path.exists():
            return f"Error: File {file_name} does not exist."

        with full_path.open("r") as file:
            lines = file.readlines()

        # Ensure inserts is a dictionary with integer keys and string values
        try:
            inserts = {int(k): str(v) for k, v in inserts.items()}
        except ValueError:
            return "Error: Invalid inserts format. Keys must be convertible to integers and values must be strings."

        sorted_inserts = sorted(inserts.items())
        for line_number, text in sorted_inserts:
            if 1 <= line_number <= len(lines) + 1:
                lines.insert(line_number - 1, text + "\n")
            else:
                return f"Error: Line number {line_number} is out of range for {file_name}."

        with full_path.open("w") as file:
            file.writelines(lines)

        return f"Document {file_name} edited and saved successfully."
    except Exception as e:
        logging.error(f"Error editing document {file_name}: {str(e)}")
        return f"Error editing document {file_name}: {str(e)}"

@tool
def python_repl(code: str):
    """Execute Python code and return the result."""
    try:
        # Ensure the code is properly formatted as a JSON string
        code = json.loads(json.dumps(code))
        # Dynamically replace hardcoded paths in the code using regex
        code = re.sub(r"plt\.savefig\((.*?)\)", f"plt.savefig(r'{REPORTS_DIR}/\\1')", code)

        result = repl.run(code)

        # Save generated images if any
        img_path = REPORTS_DIR / "generated_image.png"
        plt.savefig(img_path)
        plt.close()

        return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}\nImage saved to {img_path}"
    except json.JSONDecodeError:
        logging.error(f"Failed to parse code as JSON: {code}")
        return f"Failed to parse code as JSON. Please ensure the code is properly formatted."
    except Exception as e:
        logging.error(f"Failed to execute Python code: {str(e)}")
        return f"Failed to execute. Error: {repr(e)}"


# LLM Configuration
def get_azure_openai(temperature=0.1, model=None):
    model = model or os.getenv("AZURE_DEPLOYMENT_NAME")
    llm = AzureChatOpenAI(
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        validate_base_url=False,
        temperature=temperature,
    )
    return llm

def get_llm():
    return get_azure_openai()

llm = get_llm()

# Prompts
def create_agent_prompt(system_message):
    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

search_agent_prompt = "You are a research assistant who can search for up-to-date info using the tavily search engine."
web_scraper_prompt = "You are a research assistant who can scrape specified urls for more detailed information using the scrape_webpages function."
doc_writer_prompt = "You are an expert writing a research document. You can create new documents using write_document, edit existing documents using edit_document, and read documents using read_document. When editing, always provide the inserts parameter as a dictionary. Below are files currently in your directory:\n{current_files}"
note_taker_prompt = "You are an expert senior researcher tasked with writing a paper outline and taking notes to craft a perfect paper.{current_files}"
chart_generator_prompt = "You are a data viz expert tasked with generating charts for a research project. When using the python_repl tool, ensure that the code is properly formatted as a string. Current files in your directory:\n{current_files}"
supervisor_prompt = "You are a supervisor tasked with managing a conversation between the following workers: {team_members}. Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status. When finished, respond with FINISH."
top_level_supervisor_prompt = "You are a supervisor tasked with managing a conversation between the following teams: {team_members}. Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status. When finished, respond with FINISH."

# Agents
def create_agent(llm: AzureChatOpenAI, tools: list, system_prompt: str) -> AgentExecutor:
    system_prompt = system_prompt + "\nWork autonomously according to your specialty, using the tools available to you. Do not ask for clarification. Your other team members (and other teams) will collaborate with you with their own specialties. You are chosen for a reason! You are one of the following team members: {team_members}."
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

def create_team_supervisor(llm: AzureChatOpenAI, system_prompt, members) -> str:
    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                },
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        SystemMessagePromptTemplate.from_template("Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}"),
    ]).partial(options=str(options), team_members=", ".join(members))
    return prompt | llm.bind_functions(functions=[function_def], function_call="route") | JsonOutputFunctionsParser()

# Define agents
search_agent = create_agent(llm, [tavily_tool], search_agent_prompt)
research_agent = create_agent(llm, [scrape_webpages], web_scraper_prompt)
doc_writer_agent = create_agent(llm, [write_document, edit_document, read_document], doc_writer_prompt)
note_taking_agent = create_agent(llm, [create_outline, read_document], note_taker_prompt)
chart_generating_agent = create_agent(llm, [read_document, python_repl], chart_generator_prompt)
supervisor_agent = create_team_supervisor(llm, supervisor_prompt, ["Search", "WebScraper"])
top_level_supervisor_agent = create_team_supervisor(llm, top_level_supervisor_prompt, ["ResearchTeam", "PaperWritingTeam"])

# Helper functions
def agent_node(state, agent, name):
    try:
        result = agent.invoke(state)
        return {"messages": [HumanMessage(content=result["output"], name=name)]}
    except Exception as e:
        logging.error(f"Error in agent_node for {name}: {str(e)}")
        return {"messages": [HumanMessage(content=f"Error occurred: {str(e)}", name=name)]}

# Graph logic
def get_last_message(state: Dict[str, list]) -> str:
    # Check if the last message is an instance of BaseMessage or a similar class
    last_message = state["messages"][-1]
    
    if isinstance(last_message, BaseMessage):
        return last_message.content
    elif isinstance(last_message, dict) and "content" in last_message:
        return last_message["content"]
    else:
        raise AttributeError("The last message does not have a 'content' attribute.")

def join_graph(response: Dict[str, list]) -> Dict[str, list]:
    return {"messages": [response["messages"][-1]]}

# Research Team
def search_node(state):
    return agent_node(state, search_agent, "Search")

def research_node(state):
    return agent_node(state, research_agent, "WebScraper")

research_graph = StateGraph(ResearchTeamState)
research_graph.add_node("Search", search_node)
research_graph.add_node("WebScraper", research_node)
research_graph.add_node("supervisor", supervisor_agent)
research_graph.add_edge("Search", "supervisor")
research_graph.add_edge("WebScraper", "supervisor")
research_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {"Search": "Search", "WebScraper": "WebScraper", "FINISH": END},
)
research_graph.add_edge(START, "supervisor")
research_chain = research_graph.compile()

def enter_research_chain(message: str):
    return {
        "messages": [HumanMessage(content=message)],
        "team_members": ["Search", "WebScraper"],
    }

research_chain = enter_research_chain | research_chain

# Document Writing Team
def prelude(state):
    written_files = []
    try:
        written_files = [f.relative_to(REPORTS_DIR) for f in REPORTS_DIR.rglob("*")]
    except Exception:
        pass
    if not written_files:
        return {**state, "current_files": "No files written."}
    return {
        **state,
        "current_files": "\nBelow are files your team has written to the directory:\n" + "\n".join(
            [f" - {f}" for f in written_files]),
    }

context_aware_doc_writer_agent = prelude | doc_writer_agent

def doc_writing_node(state):
    return agent_node(state, context_aware_doc_writer_agent, "DocWriter")

context_aware_note_taking_agent = prelude | note_taking_agent

def note_taking_node(state):
    return agent_node(state, context_aware_note_taking_agent, "NoteTaker")

context_aware_chart_generating_agent = prelude | chart_generating_agent

def chart_generating_node(state):
    return agent_node(state, context_aware_chart_generating_agent, "ChartGenerator")

doc_writing_supervisor = create_team_supervisor(
    llm,
    "You are a supervisor tasked with managing a conversation between the following workers: {team_members}. Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status. When finished, respond with FINISH.",
    ["DocWriter", "NoteTaker", "ChartGenerator"],
)

authoring_graph = StateGraph(DocWritingState)
authoring_graph.add_node("DocWriter", doc_writing_node)
authoring_graph.add_node("NoteTaker", note_taking_node)
authoring_graph.add_node("ChartGenerator", chart_generating_node)
authoring_graph.add_node("supervisor", doc_writing_supervisor)
authoring_graph.add_edge("DocWriter", "supervisor")
authoring_graph.add_edge("NoteTaker", "supervisor")
authoring_graph.add_edge("ChartGenerator", "supervisor")
authoring_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "DocWriter": "DocWriter",
        "NoteTaker": "NoteTaker",
        "ChartGenerator": "ChartGenerator",
        "FINISH": END,
    },
)
authoring_graph.add_edge(START, "supervisor")
authoring_chain = authoring_graph.compile()

def enter_authoring_chain(message: str, members: list):
    return {
        "messages": [HumanMessage(content=message)],
        "team_members": ", ".join(members),
    }

authoring_chain = functools.partial(enter_authoring_chain, members=authoring_graph.nodes) | authoring_chain

# Top-level Supervisor
supervisor_node = create_team_supervisor(
    llm,
    "You are a supervisor tasked with managing a conversation between the following teams: {team_members}. Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status. When finished, respond with FINISH.",
    ["ResearchTeam", "PaperWritingTeam"],
)

super_graph = StateGraph(State)
super_graph.add_node("ResearchTeam", get_last_message | research_chain | join_graph)
super_graph.add_node("PaperWritingTeam", get_last_message | authoring_chain | join_graph)
super_graph.add_node("supervisor", supervisor_node)
super_graph.add_edge("ResearchTeam", "supervisor")
super_graph.add_edge("PaperWritingTeam", "supervisor")
super_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "PaperWritingTeam": "PaperWritingTeam",
        "ResearchTeam": "ResearchTeam",
        "FINISH": END,
    },
)
super_graph.add_edge(START, "supervisor")
super_graph = super_graph.compile()

# Main execution
user_request="Write a brief research report on the North American sturgeon. Include a chart."

def main():
    try:
        for s in super_graph.stream(
                {
                    "messages": [
                        HumanMessage(
                            content=user_request
                        )
                    ],
                },
                {"recursion_limit": 150},
        ):
            if "__end__" not in s:
                print(s)
                print("---")

        print(f"\nReports have been saved in: {REPORTS_DIR}")
    except Exception as e:
        logging.error(f"An error occurred in the main execution: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
