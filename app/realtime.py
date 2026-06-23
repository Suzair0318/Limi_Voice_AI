"""Bridge between a connected device and the OpenAI Realtime API.

Each :class:`RealtimeBridge` owns one outbound WebSocket to OpenAI for the
lifetime of a device session. The device's audio is forwarded into the Realtime
input buffer; the model's streamed audio (and lifecycle events) are delivered
back via async callbacks. Turn-taking is handled entirely by OpenAI's
server-side VAD, so there is no local end-of-turn detection.

Audio contract:
    * Sent to OpenAI:  16-bit mono PCM at ``settings.realtime_input_rate``.
    * Received from OpenAI: 16-bit mono PCM at ``settings.realtime_output_rate``.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Awaitable, Callable, Optional

import websockets

from app.config import settings

_REALTIME_URL = "wss://api.openai.com/v1/realtime"

# Async callback signatures.
AudioCallback = Callable[[bytes], Awaitable[None]]
EventCallback = Callable[[dict], Awaitable[None]]


class RealtimeBridge:
    """Manages a single OpenAI Realtime WebSocket session for one device."""

    def __init__(
        self,
        client_id: str,
        on_audio: AudioCallback,
        on_event: Optional[EventCallback] = None,
    ) -> None:
        self.client_id = client_id
        self._on_audio = on_audio
        self._on_event = on_event
        self._ws: Optional[websockets.ClientConnection] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._closed = False
        self._send_failed = False
        self._audio_out_chunks = 0

    @property
    def is_alive(self) -> bool:
        """True while the OpenAI socket is open and usable."""
        return not self._closed and self._ws is not None

    # --- Lifecycle ------------------------------------------------------
    async def connect(self) -> None:
        """Open the OpenAI socket, wait for session.created, configure, read."""
        url = f"{_REALTIME_URL}?model={settings.openai_realtime_model}"
        headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
        self._ws = await websockets.connect(
            url,
            additional_headers=headers,
            max_size=None,
            ping_interval=20,
            ping_timeout=120,
        )

        # GA Realtime API: must receive session.created before session.update.
        await self._wait_for_session_created()

        await self._configure_session()
        self._recv_task = asyncio.create_task(self._receive_loop())
        print(
            f"[REALTIME] '{self.client_id}' connected to OpenAI "
            f"({settings.openai_realtime_model}, voice={settings.openai_voice})."
        )

    async def reconnect(self) -> None:
        """Re-open the upstream socket after a ping timeout or disconnect."""
        print(f"[REALTIME] '{self.client_id}' reconnecting to OpenAI...")
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            self._recv_task = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass
            self._ws = None
        self._closed = False
        self._send_failed = False
        await self.connect()

    async def _wait_for_session_created(self) -> None:
        """Block until OpenAI sends ``session.created`` (or raise on error)."""
        assert self._ws is not None
        deadline = asyncio.get_event_loop().time() + 15.0
        while asyncio.get_event_loop().time() < deadline:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
            event = json.loads(raw)
            etype = event.get("type", "")
            if etype == "session.created":
                print(f"[REALTIME] '{self.client_id}' session.created OK.")
                return
            if etype == "error":
                err = event.get("error")
                print(f"[REALTIME] '{self.client_id}' API error on connect: {err}")
                raise RuntimeError(f"OpenAI Realtime error: {err}")
            print(f"[REALTIME] '{self.client_id}' (pre-session) {etype}")
        raise TimeoutError("Timed out waiting for session.created from OpenAI")

    async def _configure_session(self) -> None:
        """Send the initial ``session.update`` (voice, formats, VAD mode)."""
        input_audio: dict = {
            "format": {
                "type": "audio/pcm",
                "rate": settings.realtime_input_rate,
            },
        }
        if settings.local_input_buffering:
            # Device firmware already sends complete turns. Disable server VAD so
            # it cannot auto-commit the buffer before our manual commit.
            input_audio["turn_detection"] = None
        else:
            turn_detection: dict = {
                "type": settings.realtime_vad_type,
                "create_response": True,
            }
            if settings.realtime_vad_type == "server_vad":
                turn_detection["silence_duration_ms"] = settings.realtime_silence_ms
                turn_detection["prefix_padding_ms"] = 300
            input_audio["turn_detection"] = turn_detection

        event = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": settings.openai_realtime_model,
                "output_modalities": ["audio"],
                "instructions": settings.realtime_instructions,
                "audio": {
                    "input": input_audio,
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
        await self._send(event)

    async def close(self) -> None:
        """Tear down the read loop and the OpenAI socket. Never raises."""
        if self._closed:
            return
        self._closed = True
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001 - closing a broken socket is fine
                pass
        print(
            f"[REALTIME] '{self.client_id}' OpenAI session closed "
            f"(sent {self._audio_out_chunks} audio chunks to device)."
        )

    # --- Outbound (device -> OpenAI) ------------------------------------
    async def send_audio(self, pcm: bytes) -> None:
        """Append a chunk of mono PCM (at ``realtime_input_rate``) to the buffer."""
        if not pcm or self._closed:
            return
        audio_b64 = base64.b64encode(pcm).decode("ascii")
        await self._send({"type": "input_audio_buffer.append", "audio": audio_b64})

    async def commit_audio_and_request_response(self) -> None:
        """Finalize a locally buffered user turn and ask Realtime to answer."""
        if self._closed:
            return
        await self._send({"type": "input_audio_buffer.commit"})
        await self._send({"type": "response.create"})

    async def clear_output(self) -> None:
        """Clear any unplayed model audio (used for barge-in)."""
        await self._send({"type": "output_audio_buffer.clear"})

    async def _send(self, event: dict) -> None:
        if self._ws is None or self._closed:
            return
        try:
            await self._ws.send(json.dumps(event))
        except Exception as exc:  # noqa: BLE001 - a dead OpenAI socket shouldn't crash us
            if not self._send_failed:
                self._send_failed = True
                self._closed = True
                print(
                    f"[REALTIME] '{self.client_id}' OpenAI send failed — "
                    f"upstream dead: {exc!r}"
                )

    # --- Inbound (OpenAI -> device) -------------------------------------
    async def _receive_loop(self) -> None:
        """Read OpenAI events, surfacing audio deltas and lifecycle events."""
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    event = json.loads(raw)
                except (ValueError, TypeError):
                    continue

                etype = event.get("type", "")

                if etype in ("response.output_audio.delta", "response.audio.delta"):
                    delta = event.get("delta")
                    if delta:
                        try:
                            pcm = base64.b64decode(delta)
                            self._audio_out_chunks += 1
                            if self._audio_out_chunks == 1:
                                print(
                                    f"[REALTIME] '{self.client_id}' "
                                    f"first AI audio chunk ({len(pcm)}B)."
                                )
                            await self._on_audio(pcm)
                        except Exception as exc:  # noqa: BLE001
                            print(
                                f"[REALTIME] '{self.client_id}' audio handler "
                                f"error: {exc!r}"
                            )
                    continue

                if etype == "error":
                    print(
                        f"[REALTIME] '{self.client_id}' API error: "
                        f"{event.get('error')}"
                    )
                elif etype == "session.updated":
                    print(f"[REALTIME] '{self.client_id}' session.updated OK.")
                elif etype in (
                    "input_audio_buffer.speech_started",
                    "input_audio_buffer.speech_stopped",
                    "response.done",
                    "response.output_audio.done",
                    "response.output_audio_transcript.done",
                ):
                    pass  # handled by on_event / main.py

                if self._on_event is not None:
                    try:
                        await self._on_event(event)
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"[REALTIME] '{self.client_id}' event handler "
                            f"error: {exc!r}"
                        )
        except asyncio.CancelledError:
            raise
        except websockets.ConnectionClosed as exc:
            self._closed = True
            print(
                f"[REALTIME] '{self.client_id}' OpenAI socket closed by server "
                f"(code={exc.code}, reason={exc.reason!r})."
            )
        except Exception as exc:  # noqa: BLE001 - never let the read loop kill the app
            print(f"[REALTIME] '{self.client_id}' receive loop error: {exc!r}")
