"""AI layer: LangGraph + Realtime bridge without changing the audio pipeline.

``VoiceAgentLayer`` wraps :class:`app.openai_ws.OpenAIRealtimeWS` and keeps all
PCM rates, voice, VAD, and chunking exactly as configured in ``openai_ws.py``.
It adds:

* Realtime function tools (delegating execution through LangGraph's toolset)
* Input transcription for conversation memory
* Per-device LangGraph checkpoints on completed user transcripts
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.agent.graph import graph_app
from app.agent.tools import TOOL_BY_NAME, TOOLS
from app.openai_ws import OpenAIRealtimeWS

AudioCallback = Callable[[bytes], Awaitable[None]]
EventCallback = Callable[[dict], Awaitable[None]]


def langchain_tools_to_realtime(tools: list) -> list[dict]:
    """Convert LangChain tools to Realtime ``session.tools`` entries."""
    out: list[dict] = []
    for tool in tools:
        spec = convert_to_openai_tool(tool)
        fn = spec["function"]
        out.append(
            {
                "type": "function",
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get(
                    "parameters",
                    {"type": "object", "properties": {}},
                ),
            }
        )
    return out


async def _invoke_tool(name: str, arguments: dict) -> str:
    tool = TOOL_BY_NAME.get(name)
    if tool is None:
        return f"Unknown tool: {name}"
    try:
        result = await asyncio.to_thread(tool.invoke, arguments)
    except Exception as exc:  # noqa: BLE001
        return f"Tool {name} failed: {exc!r}"
    return str(result)


class VoiceAgentLayer:
    """LangGraph-aware wrapper around the Realtime WebSocket bridge."""

    def __init__(
        self,
        device_id: str,
        on_audio: AudioCallback,
        on_event: EventCallback | None = None,
    ) -> None:
        self.device_id = device_id
        self._user_on_event = on_event
        self._handled_call_ids: set[str] = set()
        self._bridge = OpenAIRealtimeWS(
            device_id,
            on_audio=on_audio,
            on_event=self._handle_event,
        )

    async def connect(self) -> None:
        await self._bridge.connect()
        await self._configure_ai_extensions()

    async def send_audio(self, pcm_24k_mono: bytes) -> None:
        await self._bridge.send_audio(pcm_24k_mono)

    async def close(self) -> None:
        await self._bridge.close()

    async def _configure_ai_extensions(self) -> None:
        """Enable transcription + tools without touching audio format settings."""
        await self._bridge._send(
            {
                "type": "session.update",
                "session": {
                    "tools": langchain_tools_to_realtime(TOOLS),
                    "tool_choice": "auto",
                    "audio": {
                        "input": {
                            "transcription": {"model": "whisper-1"},
                        },
                    },
                },
            }
        )
        print(
            f"[AGENT] {self.device_id}: LangGraph layer ready "
            f"({len(TOOLS)} tools, transcription on)."
        )

    async def _handle_event(self, event: dict) -> None:
        if self._user_on_event is not None:
            await self._user_on_event(event)

        etype = event.get("type", "")
        if etype == "response.function_call_arguments.done":
            await self._handle_function_call(event)
        elif etype == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            if transcript:
                asyncio.create_task(self._record_user_transcript(transcript))
        elif etype == "response.done":
            await self._handle_response_done(event)

    async def _handle_function_call(self, event: dict) -> None:
        name = event.get("name", "")
        call_id = event.get("call_id", "")
        if not call_id or call_id in self._handled_call_ids:
            return
        self._handled_call_ids.add(call_id)
        raw_args = event.get("arguments", "{}")
        try:
            arguments = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            arguments = {}

        print(f"[AGENT] {self.device_id}: tool call {name}({arguments})")
        output = await _invoke_tool(name, arguments)

        await self._bridge._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                },
            }
        )
        await self._bridge._send({"type": "response.create"})

        config = {"configurable": {"thread_id": self.device_id}}
        try:
            await graph_app.ainvoke(
                {
                    "messages": [
                        HumanMessage(content=f"[voice turn requesting {name}]"),
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "id": call_id,
                                    "name": name,
                                    "args": arguments,
                                }
                            ],
                        ),
                        ToolMessage(content=output, tool_call_id=call_id),
                    ]
                },
                config=config,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[AGENT] {self.device_id}: memory update failed {exc!r}")

    async def _handle_response_done(self, event: dict) -> None:
        response = event.get("response") or {}
        for item in response.get("output") or []:
            if item.get("type") != "function_call":
                continue
            if item.get("status") not in (None, "completed"):
                continue
            await self._handle_function_call(
                {
                    "name": item.get("name", ""),
                    "call_id": item.get("call_id", ""),
                    "arguments": item.get("arguments", "{}"),
                }
            )

    async def _record_user_transcript(self, transcript: str) -> None:
        config = {"configurable": {"thread_id": self.device_id}}
        try:
            await graph_app.aupdate_state(
                config,
                {"messages": [HumanMessage(content=transcript)]},
            )
            print(f"[AGENT] {self.device_id}: transcript stored {transcript!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"[AGENT] {self.device_id}: memory update failed {exc!r}")
