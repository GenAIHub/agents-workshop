from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool

# Define tools
tavily_tool = TavilySearchResults(max_results=5)
python_repl_tool = PythonREPLTool()
