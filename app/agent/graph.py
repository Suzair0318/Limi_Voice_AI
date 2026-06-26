"""LangGraph orchestration for the voice AI layer.

The Realtime WebSocket owns speech-to-speech audio. This graph handles text
reasoning, tool routing, and per-device conversation memory keyed by
``thread_id`` (the device id).

    START -> agent -> (tools? -> agent) -> END
"""

from __future__ import annotations

from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.agent.tools import TOOLS
from app.config import settings

# Text model for LangGraph reasoning — Realtime API handles voice separately.
_TEXT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are Limi, a friendly, concise real-time voice assistant running on a "
    "wearable pendant. Keep replies short and natural — one or two sentences "
    "unless asked for detail. Use the available tools when they help. "
    "Never mention that you are an AI model or describe your audio format."
)


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def _build_model() -> ChatOpenAI:
    model = ChatOpenAI(
        model=_TEXT_MODEL,
        temperature=0,
        api_key=settings.openai_api_key,
    )
    return model.bind_tools(TOOLS)


_MODEL = _build_model()
_TOOL_NODE = ToolNode(TOOLS)


def _agent_node(state: AgentState) -> dict:
    messages = state["messages"]
    response = _MODEL.invoke([SystemMessage(content=SYSTEM_PROMPT), *messages])
    return {"messages": [response]}


def _should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def _build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", _agent_node)
    workflow.add_node("tools", _TOOL_NODE)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


graph_app = _build_graph()
