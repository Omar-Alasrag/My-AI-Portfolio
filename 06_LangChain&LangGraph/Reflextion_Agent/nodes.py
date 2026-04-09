from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState

from langchain_tavily import TavilySearch

from chains import first_chain, improve_chain
from schemas import InitialAnswer, ImprovedAnswer

load_dotenv()

search_engine = TavilySearch(max_results=3)


def web_search(queries: list[str], **kwargs):
    """Search the web for more information."""
    return search_engine.batch([{"query": q} for q in queries])


def initial_response_node(state: MessagesState):
    result = first_chain.invoke(state["messages"])
    return {"messages": [result]}


def improve_response_node(state: MessagesState):
    result = improve_chain.invoke(state["messages"])
    return {"messages": [result]}


tool_node = ToolNode(
    [
        StructuredTool.from_function(web_search, name=InitialAnswer.__name__),
        StructuredTool.from_function(web_search, name=ImprovedAnswer.__name__),
    ]
)