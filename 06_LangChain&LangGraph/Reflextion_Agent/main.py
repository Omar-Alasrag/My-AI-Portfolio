from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END, MessagesState

from nodes import initial_response_node, improve_response_node, tool_node

load_dotenv()

START = "start"
SEARCH = "search"
IMPROVE = "improve"


def stop_or_continue(state: MessagesState):
    num_tool_calls = [ele for ele in state["messages"] if isinstance(ele, ToolMessage)]

    if len(num_tool_calls) >= 2:
        return END
    return SEARCH


builder = StateGraph(MessagesState)

builder.add_node(START, initial_response_node)
builder.add_node(SEARCH, tool_node)
builder.add_node(IMPROVE, improve_response_node)

builder.set_entry_point(START)

builder.add_edge(START, SEARCH)
builder.add_edge(SEARCH, IMPROVE)

builder.add_conditional_edges(
    IMPROVE,
    stop_or_continue,
    {END: END, SEARCH: SEARCH},
)

app = builder.compile()

result = app.invoke({"messages": [HumanMessage(content="Give me good advice")]})

print(result["messages"][-1].tool_calls[0]["args"]["answer"])



 