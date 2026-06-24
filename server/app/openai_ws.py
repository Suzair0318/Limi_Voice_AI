from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import Awaitable, Callable

import websockets

from app.config import settings

AudioCallback = Callable[[bytes], Awaitable[None]]
EventCallback = Callable[[dict], Awaitable[None]]


class OpenAIRealtimeWS:
    """Small OpenAI Realtime WebSocket bridge.

    Device audio is already live after firmware wake. We let Realtime server VAD
    decide turn boundaries so there is no record-full-turn delay.
    """

    def __init__(
        self,
        client_id: str,
        on_audio: AudioCallback,
        on_event: EventCallback | None = None,
    ) -> None:
        self.client_id = client_id
        self.on_audio = on_audio
        self.on_event = on_event
        self.ws: websockets.ClientConnection | None = None
        self.recv_task: asyncio.Task | None = None
        self.closed = False
        self.audio_in_chunks = 0
        self.audio_in_bytes = 0
        self.audio_out_chunks = 0
        self.audio_out_bytes = 0
        self._last_audio_in_log = time.monotonic()

    async def connect(self) -> None:
        url = f"wss://api.openai.com/v1/realtime?model={settings.openai_realtime_model}"
        self.ws = await websockets.connect(
            url,
            additional_headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            max_size=None,
            ping_interval=20,
            ping_timeout=60,
        )
        await self._wait_for_session_created()
        await self._send(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": settings.openai_realtime_model,
                    "output_modalities": ["audio"],
                    "instructions": settings.realtime_instructions,
                    "audio": {
                        "input": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": settings.realtime_input_rate,
                            },
                            "turn_detection": {
                                "type": settings.realtime_vad_type,
                                "silence_duration_ms": settings.realtime_silence_ms,
                                "prefix_padding_ms": 250,
                                "create_response": True,
                            },
                        },
                        "output": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": settings.realtime_output_rate,
                            },
                            "voice": settings.openai_voice,
                        },
                    },
                },
            }
        )
        self.recv_task = asyncio.create_task(self._recv_loop())
        print(
            f"[OPENAI] {self.client_id}: connected "
            f"model={settings.openai_realtime_model} voice={settings.openai_voice}"
        )

    async def _wait_for_session_created(self) -> None:
        assert self.ws is not None
        while True:
            event = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=15))
            etype = event.get("type")
            if etype == "session.created":
                return
            if etype == "error":
                raise RuntimeError(f"Realtime error: {event.get('error')}")

    async def send_audio(self, pcm_24k_mono: bytes) -> None:
        if self.closed or not pcm_24k_mono:
            return
        await self._send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm_24k_mono).decode("ascii"),
            }
        )
        self.audio_in_chunks += 1
        self.audio_in_bytes += len(pcm_24k_mono)
        now = time.monotonic()
        if self.audio_in_chunks == 1 or now - self._last_audio_in_log >= 2.0:
            print(
                f"[OPENAI] {self.client_id}: audio -> model "
                f"chunks={self.audio_in_chunks} bytes={self.audio_in_bytes} "
                f"last={len(pcm_24k_mono)}B"
            )
            self._last_audio_in_log = now

    async def close(self) -> None:
        self.closed = True
        if self.recv_task:
            self.recv_task.cancel()
            try:
                await self.recv_task
            except asyncio.CancelledError:
                pass
        if self.ws:
            await self.ws.close()
        print(
            f"[OPENAI] {self.client_id}: closed "
            f"sent_to_model={self.audio_in_chunks} chunks/{self.audio_in_bytes}B "
            f"audio_from_model={self.audio_out_chunks} chunks/{self.audio_out_bytes}B"
        )

    async def _send(self, event: dict) -> None:
        if self.ws is None or self.closed:
            return
        await self.ws.send(json.dumps(event))

    async def _recv_loop(self) -> None:
        assert self.ws is not None
        try:
            async for raw in self.ws:
                event = json.loads(raw)
                etype = event.get("type", "")
                if etype in ("response.output_audio.delta", "response.audio.delta"):
                    delta = event.get("delta")
                    if delta:
                        audio = base64.b64decode(delta)
                        self.audio_out_chunks += 1
                        self.audio_out_bytes += len(audio)
                        await self.on_audio(audio)
                    continue
                if etype == "error":
                    print(f"[OPENAI] {self.client_id}: error {event.get('error')}")
                elif etype in (
                    "session.updated",
                    "input_audio_buffer.speech_started",
                    "input_audio_buffer.speech_stopped",
                    "response.done",
                    "response.output_audio.done",
                ):
                    print(f"[OPENAI] {self.client_id}: {etype}")
                if self.on_event:
                    await self.on_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[OPENAI] {self.client_id}: recv loop ended: {exc!r}")
