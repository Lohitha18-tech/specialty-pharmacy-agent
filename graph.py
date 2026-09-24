import json

from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

from state import AgentState
from tools import get_case, get_case_history, search_knowledge


load_dotenv()
client = OpenAI()


# --------------------------------
# TOOL DEFINITIONS FOR LLM
# --------------------------------

tools = [
    {
        "type": "function",
        "name": "get_case",
        "description": "Get the current information and status for a specialty pharmacy case.",
        "parameters": {
            "type": "object",
            "properties": {
                "case_id": {"type": "string"}
            },
            "required": ["case_id"]
        }
    },
    {
        "type": "function",
        "name": "get_case_history",
        "description": "Get the history and past events for a specialty pharmacy case.",
        "parameters": {
            "type": "object",
            "properties": {
                "case_id": {"type": "string"}
            },
            "required": ["case_id"]
        }
    },
    {
        "type": "function",
        "name": "search_knowledge",
        "description": "Search specialty pharmacy policies and knowledge.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"}
            },
            "required": ["question"]
        }
    }
]


# --------------------------------
# LANGGRAPH AGENT NODE
# --------------------------------

def agent_node(state: AgentState):

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=f"""
You are a specialty pharmacy case intelligence agent.

Use the available tools to answer the user's question.

For the final answer, always organize the response into:

CASE SUMMARY:
Briefly summarize the current case.

OUTSTANDING ITEMS:
List anything that is still pending or required.

SUGGESTED NEXT STEPS:
List next steps supported by the retrieved case data and policy.

Do not invent information that is not supported by the tools.

User question:
{state["question"]}
""",
        tools=tools
    )

    tool_outputs = []

    for tool_call in response.output:

        if tool_call.type != "function_call":
            continue


        arguments = json.loads(tool_call.arguments)

        if tool_call.name == "get_case":
            result = get_case(arguments["case_id"])

        elif tool_call.name == "get_case_history":
            result = get_case_history(arguments["case_id"])

        elif tool_call.name == "search_knowledge":
            result = search_knowledge(arguments["question"])

        else:
            result = {"error": "Unknown tool"}

        tool_outputs.append(
            {
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": json.dumps(result)
            }
        )

    final_response = client.responses.create(
        model="gpt-5.6-luna",
        previous_response_id=response.id,
        tools=tools,
        input=tool_outputs
    )

    return {
        "answer": final_response.output_text
    }


# --------------------------------
# BUILD LANGGRAPH
# --------------------------------

builder = StateGraph(AgentState)

builder.add_node("agent", agent_node)

builder.add_edge(START, "agent")
builder.add_edge("agent", END)

graph = builder.compile()


