"""Async WebSocket session — simulates the hardware pendant for the Limi backend."""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import websockets
from websockets.client import WebSocketClientProtocol

from desktop.audio import MicStreamer, SpeakerPlayer
from desktop.config import ClientConfig
from desktop.protocol import (
    build_hello,
    build_wake_detected,
    build_wake_session_end,
    event_label,
    parse_control_message,
)


class SessionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    LISTENING = "listening"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass
class SessionStats:
    mic_bytes: int = 0
    mic_frames: int = 0
    speaker_bytes: int = 0
    speaker_chunks: int = 0


@dataclass
class UiEvent:
    kind: str
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)


EventCallback = Callable[[UiEvent], None]


class VoiceSession:
    """Runs the device WebSocket protocol in a background asyncio loop."""

    def __init__(self, config: ClientConfig, on_event: EventCallback) -> None:
        self.config = config
        self._on_event = on_event
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ws: WebSocketClientProtocol | None = None
        self._mic_send_queue: asyncio.Queue[bytes | None] | None = None
        self._mic_sender_task: asyncio.Task[None] | None = None
        self._receiver_task: asyncio.Task[None] | None = None
        self._stop_event = threading.Event()
        self._mic_active = False
        self._speaker_active = False
        self.state = SessionState.DISCONNECTED
        self.stats = SessionStats()
        self._mic: MicStreamer | None = None
        self._speaker = SpeakerPlayer(config, on_level=self._on_speaker_level)

    def _emit(self, kind: str, message: str = "", **data: Any) -> None:
        self._on_event(UiEvent(kind=kind, message=message, data=data))

    def _set_state(self, state: SessionState) -> None:
        self.state = state
        self._emit("state", state=state.value)

    def _on_mic_level(self, peak: float) -> None:
        self._emit("mic_level", peak=peak)

    def _on_speaker_level(self, peak: float) -> None:
        self._emit("speaker_level", peak=peak)

    def _on_mic_pcm(self, pcm: bytes) -> None:
        if not self._loop or not self._mic_send_queue or not self._mic_active:
            return
        self._loop.call_soon_threadsafe(self._enqueue_mic, pcm)

    def _enqueue_mic(self, pcm: bytes) -> None:
        if self._mic_send_queue is None:
            return
        try:
            self._mic_send_queue.put_nowait(pcm)
        except asyncio.QueueFull:
            pass

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self.stop_mic()
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        if self._thread:
            self._thread.join(timeout=5.0)
        self._speaker.close()
        self._set_state(SessionState.DISCONNECTED)

    @property
    def mic_active(self) -> bool:
        return self._mic_active

    def start_mic(self) -> None:
        if self.state not in (SessionState.CONNECTED, SessionState.SPEAKING):
            return
        if self._mic_active:
            return
        self._mic_active = True
        if self._mic is None:
            self._mic = MicStreamer(self.config, self._on_mic_pcm, self._on_mic_level)
        self._mic.start()
        self._set_state(SessionState.LISTENING)

    def stop_mic(self) -> None:
        if not self._mic_active:
            return
        self._mic_active = False
        if self._mic is not None:
            self._mic.stop()
        if self._speaker_active:
            self._set_state(SessionState.SPEAKING)
        elif self.state != SessionState.DISCONNECTED:
            self._set_state(SessionState.CONNECTED)

    def send_wake_detected(self) -> None:
        self._run_coro(self._send_text(build_wake_detected()))

    def send_wake_session_end(self) -> None:
        self._run_coro(self._send_text(build_wake_session_end()))

    def fetch_health(self) -> dict[str, Any] | None:
        try:
            with urllib.request.urlopen(self.config.health_url, timeout=3) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            self._emit("log", message=f"Health check failed: {exc}", level="warn")
            return None

    def _run_coro(self, coro: Any) -> None:
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run())
        except Exception as exc:  # noqa: BLE001
            self._emit("log", message=f"Session crashed: {exc!r}", level="error")
            self._set_state(SessionState.ERROR)
        finally:
            self._loop.close()
            self._loop = None

    async def _run(self) -> None:
        self._set_state(SessionState.CONNECTING)
        self._emit("log", message=f"Connecting to {self.config.ws_url}", level="info")
        try:
            async with websockets.connect(self.config.ws_url, max_size=None) as ws:
                self._ws = ws
                self._mic_send_queue = asyncio.Queue(maxsize=64)
                await ws.send(build_hello(self.config))
                self._emit("log", message="Sent hello", level="info")
                self._set_state(SessionState.CONNECTED)

                self._mic_sender_task = asyncio.create_task(self._mic_sender())
                self._receiver_task = asyncio.create_task(self._receiver())

                while not self._stop_event.is_set():
                    await asyncio.sleep(0.1)
        except Exception as exc:  # noqa: BLE001
            self._emit("log", message=f"Connection error: {exc!r}", level="error")
            self._set_state(SessionState.ERROR)
        finally:
            await self._shutdown_tasks()
            self._ws = None
            if not self._stop_event.is_set():
                self._set_state(SessionState.DISCONNECTED)

    async def _shutdown(self) -> None:
        if self._mic_send_queue is not None:
            await self._mic_send_queue.put(None)
        await self._shutdown_tasks()
        if self._ws is not None:
            await self._ws.close()

    async def _shutdown_tasks(self) -> None:
        for task in (self._mic_sender_task, self._receiver_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._mic_sender_task = None
        self._receiver_task = None

    async def _send_text(self, text: str) -> None:
        if self._ws is None:
            return
        await self._ws.send(text)
        self._emit("log", message=f"Sent {text}", level="info")

    async def _mic_sender(self) -> None:
        assert self._ws is not None
        assert self._mic_send_queue is not None
        while True:
            pcm = await self._mic_send_queue.get()
            if pcm is None:
                return
            if not self._mic_active:
                continue
            await self._ws.send(pcm)
            self.stats.mic_bytes += len(pcm)
            self.stats.mic_frames += 1
            self._emit("stats", **self.stats.__dict__)

    async def _receiver(self) -> None:
        assert self._ws is not None
        while True:
            message = await self._ws.recv()
            if isinstance(message, bytes):
                self._handle_audio(message)
                continue
            self._handle_control(message)

    def _handle_audio(self, pcm: bytes) -> None:
        self.stats.speaker_bytes += len(pcm)
        self.stats.speaker_chunks += 1
        self._emit("stats", **self.stats.__dict__)
        if not self._speaker_active:
            self._speaker_active = True
            self._speaker.begin()
            self._set_state(SessionState.SPEAKING)
        self._speaker.feed(pcm)

    def _handle_control(self, text: str) -> None:
        event = parse_control_message(text)
        label = event_label(event)
        self._emit("log", message=label, level="info", event=event)

        etype = event.get("type")
        if etype == "backend_ready":
            self._emit("backend_ready", **event)
        elif etype == "speaker_begin":
            self._speaker_active = True
            self._speaker.begin()
            self._set_state(SessionState.SPEAKING)
        elif etype == "speaker_end":
            self._speaker_active = False
            self._speaker.end()
            if self._mic_active:
                self._set_state(SessionState.LISTENING)
            else:
                self._set_state(SessionState.CONNECTED)
        elif etype == "error":
            self._emit("log", message=str(event.get("detail", event)), level="error")
