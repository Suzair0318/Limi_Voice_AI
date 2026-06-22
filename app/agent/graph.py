"""LangGraph definition for the multimodal voice agent.

The graph implements a minimal ReAct-style loop:

    START -> agent -> (tools? -> agent) -> END

* ``agent`` calls the multimodal audio model (``gpt-audio`` by default). The
  model is configured to emit *both* text and audio, so the final AI turn
  carries a base64 WAV payload in ``additional_kwargs["audio"]``.
* ``tools`` executes any tool calls the model requested and feeds the results
  back into the model.

Conversation memory is provided by a :class:`MemorySaver` checkpointer keyed by
``thread_id`` (we use the device's ``client_id``), so each device keeps an
independent rolling history.
"""

from __future__ import annotations

import base64
from typing import Annotated, List, Optional, Tuple, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.config import settings
from app.agent.tools import TOOLS

SYSTEM_PROMPT = (
    "You are Limi, a friendly, concise real-time voice assistant running on a "
    "wearable pendant. Keep spoken replies short and natural - one or two "
    "sentences unless asked for detail. Use the available tools when they help. "
    "Never mention that you are an AI model or describe your audio format."
)


class AgentState(TypedDict):
    """Graph state: an append-only list of conversation messages."""

    messages: Annotated[List[BaseMessage], add_messages]


def _build_model() -> ChatOpenAI:
    """Instantiate the multimodal ChatOpenAI model with audio output enabled."""
    model = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=settings.openai_api_key,
        # Ask the model for both a text transcript and synthesized audio.
        modalities=["text", "audio"],
        audio={"voice": settings.openai_voice, "format": "wav"},
    )
    return model.bind_tools(TOOLS)


# Build the model + tool node once at import time (cheap, reused per request).
_MODEL = _build_model()
_TOOL_NODE = ToolNode(TOOLS)


def _agent_node(state: AgentState) -> dict:
    """Invoke the LLM with the running conversation and a system preamble."""
    messages = state["messages"]
    # Prepend the system prompt only once (it is not persisted in state).
    response = _MODEL.invoke([SystemMessage(content=SYSTEM_PROMPT), *messages])
    return {"messages": [response]}


def _should_continue(state: AgentState) -> str:
    """Route to the tool node if the last AI message requested tool calls."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def _build_graph():
    """Compile the StateGraph into a runnable app with memory."""
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", _agent_node)
    workflow.add_node("tools", _TOOL_NODE)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# The compiled, ready-to-invoke graph (singleton).
graph_app = _build_graph()


def extract_audio_response(message: BaseMessage) -> Tuple[Optional[bytes], str]:
    """Pull the raw audio bytes and transcript out of an AI message.

    Returns a tuple of ``(wav_bytes_or_None, transcript_text)``. ``wav_bytes``
    is the decoded WAV container the model produced (already base64-decoded).
    """
    transcript = ""
    audio_bytes: Optional[bytes] = None

    audio = getattr(message, "additional_kwargs", {}).get("audio")
    if isinstance(audio, dict):
        transcript = audio.get("transcript", "") or ""
        data_b64 = audio.get("data")
        if data_b64:
            try:
                audio_bytes = base64.b64decode(data_b64)
            except (ValueError, TypeError) as exc:
                print(f"[ERROR] Failed to decode model audio payload: {exc!r}")
                audio_bytes = None

    # Fall back to plain text content if no audio transcript was present.
    if not transcript and isinstance(message.content, str):
        transcript = message.content

    return audio_bytes, transcript
