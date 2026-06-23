"""FastAPI application: WebSocket entrypoint for the voice pendant.

Each device gets one WebSocket to this server and, behind it, one WebSocket to
the OpenAI Realtime API:

    device mic (16kHz PCM)  ->  resample to 24kHz  ->  OpenAI Realtime input
        firmware records a complete turn, then the backend commits it manually
    OpenAI audio (24kHz)  ->  resample to 48kHz stereo  ->  device speaker

The ESP32 POC uses half-duplex turn taking: the backend ignores device mic
audio while speaker audio is in flight, preventing speaker echo loops.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import wave
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from starlette.responses import Response

from app.audio.device_out import DeviceOutboundStreamer
from app.audio.resample import (
    MonoPcmProcessor,
    downmix_to_mono_pcm16,
    downmix_loudest_pair_to_mono_pcm16,
    resample_pcm16_mono,
)
from app.audio.vad import compute_rms
from app.config import settings
from app.connection import DeviceSession, manager
from app.realtime import RealtimeBridge


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown hooks."""
    print("[STARTUP] Voice backend starting. "
          f"Model={settings.openai_realtime_model}, voice={settings.openai_voice}, "
          f"turn_detection={'manual' if settings.local_input_buffering else settings.realtime_vad_type}, "
          f"speaker={settings.output_sample_rate}Hz "
          f"{settings.output_i2s_slot_bits}-bit slots "
          f"chunk={settings.output_chunk_bytes}B.")
    yield
    await manager.shutdown()
    print("[SHUTDOWN] Voice backend stopped.")


app = FastAPI(title="Limi Voice AI Backend", version="0.2.0", lifespan=lifespan)

# Path to the browser test console (frontend/index.html).
FRONTEND_INDEX = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


def _normalize_transcript(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _wake_phrases() -> list[str]:
    return [
        _normalize_transcript(phrase)
        for phrase in settings.wake_word_phrases.split(",")
        if _normalize_transcript(phrase)
    ]


def _contains_wake_word(transcript: str) -> bool:
    normalized = _normalize_transcript(transcript)
    if not normalized:
        return False
    return any(phrase in normalized for phrase in _wake_phrases())


class MicTurnRecorder:
    """Save selected mono mic audio for each detected user turn."""

    def __init__(self, client_id: str, sample_rate: int, label: str = "ai24k") -> None:
        self._client_id = client_id
        self._sample_rate = sample_rate
        self._label = label
        self._sample_width = settings.input_sample_width
        self._pre_roll = bytearray()
        self._frames = bytearray()
        self._active = False
        self._turn_id = 0
        self._latest_pair: int | None = None
        self._latest_rms = 0.0
        self._turn_pairs: deque[int] = deque(maxlen=80)
        self._update_limits()

    def configure(self, sample_rate: int) -> None:
        if sample_rate == self._sample_rate:
            return
        self._sample_rate = sample_rate
        self._pre_roll.clear()
        self._frames.clear()
        self._active = False
        self._update_limits()

    def _update_limits(self) -> None:
        self._pre_roll_limit = max(
            int(self._sample_rate * self._sample_width * settings.mic_record_preroll_ms / 1000),
            0,
        )
        self._min_bytes = max(
            int(self._sample_rate * self._sample_width * settings.mic_record_min_ms / 1000),
            1,
        )

    def add_frame(self, pcm: bytes, selected_pair: int | None, rms: float) -> None:
        if not settings.save_mic_turns or not pcm:
            return
        self._latest_pair = selected_pair
        self._latest_rms = rms
        if selected_pair is not None:
            self._turn_pairs.append(selected_pair)

        self._pre_roll.extend(pcm)
        if self._pre_roll_limit and len(self._pre_roll) > self._pre_roll_limit:
            del self._pre_roll[: len(self._pre_roll) - self._pre_roll_limit]

        if self._active:
            self._frames.extend(pcm)

    def mark_speech_started(self) -> None:
        if not settings.save_mic_turns:
            return
        if self._active:
            return
        self._turn_id += 1
        self._active = True
        self._frames = bytearray(self._pre_roll)
        self._turn_pairs.clear()
        if self._latest_pair is not None:
            self._turn_pairs.append(self._latest_pair)
        print(
            f"[REC] '{self._client_id}' mic turn #{self._turn_id} started "
            f"(pair={self._latest_pair}, preroll={len(self._frames)}B)."
        )

    def mark_speech_stopped(self, reason: str = "speech_stopped") -> None:
        if not settings.save_mic_turns or not self._active:
            return
        self._active = False
        self._save(reason)

    def close(self) -> None:
        if self._active:
            self._active = False
            self._save("disconnect")

    def _save(self, reason: str) -> None:
        if len(self._frames) < self._min_bytes:
            print(
                f"[REC] '{self._client_id}' mic turn #{self._turn_id} skipped "
                f"({len(self._frames)}B, reason={reason})."
            )
            self._frames.clear()
            return

        settings.mic_recordings_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        pair_label = self._dominant_pair_label()
        filename = (
            f"{self._client_id}_{timestamp}_turn{self._turn_id:04d}_"
            f"{pair_label}_{self._label}_{self._sample_rate}hz.wav"
        )
        path = settings.mic_recordings_dir / filename
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(self._sample_width)
            wav.setframerate(self._sample_rate)
            wav.writeframes(bytes(self._frames))

        duration = len(self._frames) / (self._sample_rate * self._sample_width)
        print(
            f"[REC] '{self._client_id}' saved mic turn #{self._turn_id}: "
            f"{path} ({duration:.2f}s, {pair_label}, last_rms={self._latest_rms:.0f}, "
            f"reason={reason})."
        )
        self._frames.clear()

    def _dominant_pair_label(self) -> str:
        if not self._turn_pairs:
            return "pair-auto"
        counts: dict[int, int] = {}
        for pair in self._turn_pairs:
            counts[pair] = counts.get(pair, 0) + 1
        pair = max(counts, key=counts.get)
        return f"pair{pair}"


class RawChannelDebugRecorder:
    """Save raw board mic audio for the same local turn window.

    This writes one multi-channel WAV plus one mono WAV per channel. It is meant
    for hardware/uplink diagnosis before any pair selection, cleanup, or AI.
    """

    def __init__(self, client_id: str, sample_rate: int, channels: int) -> None:
        self._client_id = client_id
        self._sample_rate = sample_rate
        self._channels = channels
        self._sample_width = settings.input_sample_width
        self._pre_roll = bytearray()
        self._frames = bytearray()
        self._active = False
        self._turn_id = 0
        self._update_limits()

    def configure(self, sample_rate: int, channels: int) -> None:
        if sample_rate == self._sample_rate and channels == self._channels:
            return
        self._sample_rate = sample_rate
        self._channels = channels
        self._pre_roll.clear()
        self._frames.clear()
        self._active = False
        self._update_limits()

    def _update_limits(self) -> None:
        bytes_per_ms = (
            self._sample_rate
            * max(self._channels, 1)
            * self._sample_width
            / 1000.0
        )
        self._pre_roll_limit = max(
            int(bytes_per_ms * settings.mic_record_preroll_ms),
            0,
        )
        self._min_bytes = max(
            int(bytes_per_ms * settings.mic_record_min_ms),
            1,
        )

    def add_frame(self, payload: bytes) -> None:
        if not settings.save_mic_debug_channels or not payload:
            return
        self._pre_roll.extend(payload)
        if self._pre_roll_limit and len(self._pre_roll) > self._pre_roll_limit:
            del self._pre_roll[: len(self._pre_roll) - self._pre_roll_limit]
        if self._active:
            self._frames.extend(payload)

    def mark_speech_started(self) -> None:
        if not settings.save_mic_debug_channels or self._active:
            return
        self._turn_id += 1
        self._active = True
        self._frames = bytearray(self._pre_roll)
        print(
            f"[REC] '{self._client_id}' raw channel debug #{self._turn_id} "
            f"started (preroll={len(self._frames)}B, channels={self._channels})."
        )

    def mark_speech_stopped(self, reason: str = "speech_stopped") -> None:
        if not settings.save_mic_debug_channels or not self._active:
            return
        self._active = False
        self._save(reason)

    def close(self) -> None:
        if self._active:
            self._active = False
            self._save("disconnect")

    def _save(self, reason: str) -> None:
        if len(self._frames) < self._min_bytes or self._channels <= 0:
            print(
                f"[REC] '{self._client_id}' raw channel debug #{self._turn_id} "
                f"skipped ({len(self._frames)}B, reason={reason})."
            )
            self._frames.clear()
            return

        frame_bytes = self._channels * self._sample_width
        usable = (len(self._frames) // frame_bytes) * frame_bytes
        if usable <= 0:
            self._frames.clear()
            return
        data = bytes(self._frames[:usable])

        settings.mic_recordings_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = (
            f"{self._client_id}_{timestamp}_turn{self._turn_id:04d}_"
            f"raw{self._channels}ch_{self._sample_rate}hz"
        )
        raw_path = settings.mic_recordings_dir / f"{stem}.wav"
        with wave.open(str(raw_path), "wb") as wav:
            wav.setnchannels(self._channels)
            wav.setsampwidth(self._sample_width)
            wav.setframerate(self._sample_rate)
            wav.writeframes(data)

        samples = memoryview(data).cast("h")
        for ch in range(self._channels):
            ch_path = settings.mic_recordings_dir / f"{stem}_ch{ch}.wav"
            out = bytearray()
            for idx in range(ch, len(samples), self._channels):
                out += int(samples[idx]).to_bytes(
                    self._sample_width,
                    byteorder="little",
                    signed=True,
                )
            with wave.open(str(ch_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(self._sample_width)
                wav.setframerate(self._sample_rate)
                wav.writeframes(bytes(out))

        duration = usable / (
            self._sample_rate * self._channels * self._sample_width
        )
        print(
            f"[REC] '{self._client_id}' saved raw channel debug #{self._turn_id}: "
            f"{raw_path} plus ch0..ch{self._channels - 1} "
            f"({duration:.2f}s, reason={reason})."
        )
        self._frames.clear()


class LocalTurnBuffer:
    """Collect one complete local speech turn before sending it upstream."""

    def __init__(self, client_id: str, sample_rate: int) -> None:
        self._client_id = client_id
        self._sample_rate = sample_rate
        self._sample_width = settings.input_sample_width
        self._bytes_per_ms = max(self._sample_rate * self._sample_width / 1000.0, 1.0)
        self._pre_roll = bytearray()
        self._frames = bytearray()
        self._active = False
        self._turn_id = 0
        self._silence_bytes = 0
        self._pre_roll_limit = int(
            self._bytes_per_ms * settings.mic_record_preroll_ms
        )
        self._silence_limit = int(
            self._bytes_per_ms * settings.local_vad_silence_ms
        )
        self._min_bytes = int(self._bytes_per_ms * settings.local_vad_min_ms)
        self._max_bytes = int(self._bytes_per_ms * settings.local_vad_max_ms)

    def add_frame(
        self,
        pcm: bytes,
        rms: float,
        selected_pair: int | None,
    ) -> tuple[bool, tuple[int, bytes, str] | None]:
        if not settings.local_input_buffering or not pcm:
            return False, None

        started = False
        if not self._active:
            self._pre_roll.extend(pcm)
            if self._pre_roll_limit and len(self._pre_roll) > self._pre_roll_limit:
                del self._pre_roll[: len(self._pre_roll) - self._pre_roll_limit]
            if rms < settings.local_vad_start_rms:
                return False, None

            self._turn_id += 1
            self._active = True
            self._frames = bytearray(self._pre_roll)
            self._silence_bytes = 0
            started = True
            print(
                f"[LOCAL_VAD] '{self._client_id}' turn #{self._turn_id} start "
                f"(rms={rms:.0f}, pair={selected_pair}, preroll={len(self._frames)}B)."
            )
            return started, None

        self._frames.extend(pcm)
        if rms < settings.local_vad_stop_rms:
            self._silence_bytes += len(pcm)
        else:
            self._silence_bytes = 0

        reason = ""
        if len(self._frames) >= self._max_bytes:
            reason = "max_turn"
        elif len(self._frames) >= self._min_bytes and self._silence_bytes >= self._silence_limit:
            reason = "silence"

        if not reason:
            return started, None

        turn_id = self._turn_id
        completed = bytes(self._frames)
        duration = len(completed) / (self._sample_rate * self._sample_width)
        print(
            f"[LOCAL_VAD] '{self._client_id}' turn #{turn_id} complete "
            f"({duration:.2f}s, {len(completed)}B, reason={reason})."
        )
        self._reset_after_turn()
        return started, (turn_id, completed, reason)

    def flush(self, reason: str) -> tuple[int, bytes, str] | None:
        if not self._active or len(self._frames) < self._min_bytes:
            self._reset_after_turn()
            return None
        turn_id = self._turn_id
        completed = bytes(self._frames)
        self._reset_after_turn()
        return turn_id, completed, reason

    def _reset_after_turn(self) -> None:
        self._active = False
        self._frames.clear()
        self._silence_bytes = 0
        self._pre_roll.clear()


@app.get("/", response_class=HTMLResponse)
async def index() -> Response:
    """Serve the browser test console (mic capture + audio playback)."""
    if FRONTEND_INDEX.is_file():
        return FileResponse(str(FRONTEND_INDEX))
    return HTMLResponse(
        "<h1>Limi Voice AI</h1><p>Frontend not found. "
        "Expected at <code>frontend/index.html</code>.</p>",
        status_code=404,
    )


@app.get("/health")
async def health() -> dict:
    """Simple liveness probe."""
    return {"status": "ok", "active_devices": manager.active_count}


@app.websocket("/ws/{client_id}")
async def voice_socket(websocket: WebSocket, client_id: str) -> None:
    """Main WebSocket endpoint: bridge one device to one OpenAI Realtime session."""
    session = await manager.connect(client_id, websocket)
    speaker = DeviceOutboundStreamer(session, client_id)

    # Tracks whether the model is currently streaming audio to the device.
    responding = {"active": False}
    ignore_mic_until = {"time": 0.0}
    wake_state = {
        "awake_until": 0.0,
        "pending": False,
        "allowed": not settings.wake_word_enabled,
        "last_transcript": "",
    }
    pending_ai_audio: list[bytes] = []
    device_turn_mode = {"enabled": False, "active": False, "turn_id": 0}
    device_turn_pcm = bytearray()
    mic_recorder: MicTurnRecorder | None = None
    raw_mic_recorder: MicTurnRecorder | None = None
    clean_mic_recorder: MicTurnRecorder | None = None
    raw_debug_recorder: RawChannelDebugRecorder | None = None
    local_turn_buffer: LocalTurnBuffer | None = None
    # Serialise OpenAI audio deltas — same ordered pacing as ws_server.py binary send.
    ai_audio_q: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def begin_speaker_response() -> None:
        if responding["active"]:
            return
        responding["active"] = True
        session.is_responding = True
        ignore_mic_until["time"] = time.monotonic() + 60.0
        await manager.send_json(session, {"type": "speaker_begin"})
        print(f"[STREAM] '{client_id}' AI audio -> device speaker.")

    async def ai_audio_worker() -> None:
        """Resample and pace model audio as 3840B binary frames (no control JSON)."""
        while True:
            pcm = await ai_audio_q.get()
            try:
                if pcm is None:
                    return
                if settings.stream_speaker_live:
                    await begin_speaker_response()
                await speaker.feed_mono_pcm(pcm, settings.realtime_output_rate)
            finally:
                ai_audio_q.task_done()

    worker_task = asyncio.create_task(ai_audio_worker())

    async def on_audio(pcm_24k_mono: bytes) -> None:
        """Queue one OpenAI delta; worker sends paced 3840B s16le frames."""
        if not pcm_24k_mono:
            return
        if settings.wake_word_enabled and not wake_state["allowed"]:
            pending_ai_audio.append(pcm_24k_mono)
            return
        await ai_audio_q.put(pcm_24k_mono)

    async def release_pending_ai_audio() -> None:
        if not pending_ai_audio:
            return
        for pcm in pending_ai_audio:
            await ai_audio_q.put(pcm)
        print(
            f"[WAKE] '{client_id}' released {len(pending_ai_audio)} buffered "
            "assistant audio chunk(s)."
        )
        pending_ai_audio.clear()

    async def drop_pending_ai_audio(reason: str) -> None:
        if pending_ai_audio:
            print(
                f"[WAKE] '{client_id}' dropped {len(pending_ai_audio)} buffered "
                f"assistant audio chunk(s): {reason}."
            )
            pending_ai_audio.clear()
        await speaker.clear_async()
        await bridge.clear_output()

    async def on_event(event: dict) -> None:
        """React to OpenAI lifecycle events (turn boundaries + barge-in)."""
        etype = event.get("type", "")
        if "input_audio_transcription" in etype and (
            etype.endswith(".completed") or etype.endswith(".done")
        ):
            transcript = event.get("transcript") or ""
            wake_state["last_transcript"] = transcript
            print(f"[REALTIME] '{client_id}' user transcript: {transcript!r}")
            if settings.wake_word_enabled and wake_state["pending"]:
                if _contains_wake_word(transcript):
                    wake_state["awake_until"] = (
                        time.monotonic() + settings.wake_session_seconds
                    )
                    wake_state["allowed"] = True
                    wake_state["pending"] = False
                    print(
                        f"[WAKE] '{client_id}' wake word accepted; "
                        f"awake for {settings.wake_session_seconds:g}s."
                    )
                    await release_pending_ai_audio()
                elif wake_state["allowed"]:
                    wake_state["pending"] = False
                    await release_pending_ai_audio()
                else:
                    wake_state["pending"] = False
                    print(
                        f"[WAKE] '{client_id}' ignored turn without wake word."
                    )
                    await drop_pending_ai_audio("missing wake word")
            return

        if etype in ("response.done", "response.output_audio.done"):
            if settings.wake_word_enabled and wake_state["pending"]:
                if wake_state["allowed"]:
                    wake_state["pending"] = False
                    await release_pending_ai_audio()
                else:
                    wake_state["pending"] = False
                    print(
                        f"[WAKE] '{client_id}' no transcript before response.done; "
                        "releasing response so speaker is not muted."
                    )
                    wake_state["allowed"] = True
                    await release_pending_ai_audio()
            await ai_audio_q.join()
            if responding["active"] or await speaker.has_pending_audio():
                await begin_speaker_response()
                await speaker.flush()
                await manager.send_json(session, {"type": "speaker_end"})
                responding["active"] = False
                session.is_responding = False
                ignore_mic_until["time"] = (
                    time.monotonic() + settings.ignore_mic_tail_ms / 1000.0
                )
                print(
                    f"[STREAM] '{client_id}' AI audio done "
                    f"({speaker.total_sent}B sent to device)."
                )
                if speaker.total_sent == 0:
                    print(
                        f"[WARN] '{client_id}' AI response with 0 speaker bytes — "
                        f"device will stay silent."
                    )
        elif etype == "input_audio_buffer.speech_started":
            print(f"[REALTIME] '{client_id}' user speech started.")
            if not settings.local_input_buffering and mic_recorder is not None:
                mic_recorder.mark_speech_started()
                if raw_mic_recorder is not None:
                    raw_mic_recorder.mark_speech_started()
                if clean_mic_recorder is not None:
                    clean_mic_recorder.mark_speech_started()
                if raw_debug_recorder is not None:
                    raw_debug_recorder.mark_speech_started()
            if responding["active"] and settings.ignore_mic_while_speaking:
                print(
                    f"[REALTIME] '{client_id}' ignoring speech_started while "
                    "speaker output is active."
                )
            elif responding["active"]:
                responding["active"] = False
                session.is_responding = False
                while not ai_audio_q.empty():
                    try:
                        ai_audio_q.get_nowait()
                        ai_audio_q.task_done()
                    except asyncio.QueueEmpty:
                        break
                await speaker.clear_async()
                await bridge.clear_output()
        elif etype == "input_audio_buffer.speech_stopped":
            print(f"[REALTIME] '{client_id}' user speech stopped (turn end).")
            if not settings.local_input_buffering and mic_recorder is not None:
                mic_recorder.mark_speech_stopped()
                if raw_mic_recorder is not None:
                    raw_mic_recorder.mark_speech_stopped()
                if clean_mic_recorder is not None:
                    clean_mic_recorder.mark_speech_stopped()
                if raw_debug_recorder is not None:
                    raw_debug_recorder.mark_speech_stopped()
        elif etype == "response.output_audio_transcript.done":
            transcript = event.get("transcript") or ""
            print(f"[REALTIME] '{client_id}' transcript: {transcript!r}")

    bridge = RealtimeBridge(client_id, on_audio=on_audio, on_event=on_event)
    try:
        await bridge.connect()
    except Exception as exc:  # noqa: BLE001 - failed upstream connect, drop the device
        print(f"[ERROR] '{client_id}' could not connect to OpenAI Realtime: {exc!r}")
        await manager.send_json(session, {"type": "error", "detail": "upstream_failed"})
        await manager.disconnect(client_id)
        return

    async def send_complete_user_turn(turn_id: int, pcm: bytes, reason: str) -> None:
        """Send one compiled user utterance to Realtime after local VAD completes."""
        if not pcm:
            return
        if settings.wake_word_enabled:
            now = time.monotonic()
            already_awake = (
                not settings.wake_require_each_turn
                and wake_state["awake_until"] > now
            )
            wake_state["pending"] = not already_awake
            wake_state["allowed"] = already_awake
            wake_state["last_transcript"] = ""
            pending_ai_audio.clear()
            if already_awake:
                remaining = max(wake_state["awake_until"] - now, 0.0)
                print(
                    f"[WAKE] '{client_id}' already awake; accepting turn #{turn_id} "
                    f"({remaining:.1f}s left)."
                )
            else:
                print(
                    f"[WAKE] '{client_id}' waiting for wake word in turn #{turn_id}."
                )
        else:
            wake_state["pending"] = False
            wake_state["allowed"] = True

        tail_bytes = int(
            settings.realtime_input_rate
            * settings.input_sample_width
            * settings.local_ai_tail_silence_ms
            / 1000
        )
        if tail_bytes > 0:
            pcm = pcm + (b"\x00" * tail_bytes)

        chunk_bytes = max(
            int(
                settings.realtime_input_rate
                * settings.input_sample_width
                * settings.local_ai_send_chunk_ms
                / 1000
            ),
            settings.input_sample_width,
        )
        duration = len(pcm) / (settings.realtime_input_rate * settings.input_sample_width)
        print(
            f"[LOCAL_VAD] '{client_id}' sending complete turn #{turn_id} to AI "
            f"({duration:.2f}s, {len(pcm)}B, chunk={chunk_bytes}B, reason={reason})."
        )

        if not bridge.is_alive:
            await bridge.reconnect()
        for offset in range(0, len(pcm), chunk_bytes):
            if not bridge.is_alive:
                await bridge.reconnect()
            await bridge.send_audio(pcm[offset : offset + chunk_bytes])
            await asyncio.sleep(0)
        await bridge.commit_audio_and_request_response()

    user_turn_q: asyncio.Queue[tuple[int, bytes, str] | None] = asyncio.Queue()

    async def user_turn_worker() -> None:
        """Forward completed local turns without blocking device packet reads."""
        while True:
            item = await user_turn_q.get()
            try:
                if item is None:
                    return
                turn_id, turn_pcm, reason = item
                await send_complete_user_turn(turn_id, turn_pcm, reason)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[LOCAL_VAD] '{client_id}' could not send completed turn: {exc!r}"
                )
            finally:
                user_turn_q.task_done()

    user_turn_task = asyncio.create_task(user_turn_worker())

    # --- Per-session mic format (may be overridden by the device "hello") ---
    in_rate = settings.input_sample_rate
    in_channels = settings.input_channels
    selected_mic_pair = settings.input_mic_pair
    raw_mic_recorder = MicTurnRecorder(client_id, in_rate, "raw16k")
    clean_mic_recorder = MicTurnRecorder(client_id, in_rate, "clean16k")
    mic_recorder = MicTurnRecorder(client_id, settings.realtime_input_rate, "ai24k")
    raw_debug_recorder = RawChannelDebugRecorder(client_id, in_rate, in_channels)
    local_turn_buffer = LocalTurnBuffer(client_id, settings.realtime_input_rate)
    mic_processor = MonoPcmProcessor(
        gain=settings.mic_preprocess_gain,
        noise_gate_percent=settings.mic_noise_gate_percent,
        dc_block=settings.mic_dc_block,
        soft_limiter=settings.mic_soft_limiter,
    )
    print(
        f"[AUDIO] '{client_id}' mic cleanup: gain={settings.mic_preprocess_gain:g}x, "
        f"gate={settings.mic_noise_gate_percent:g}%, "
        f"dc_block={settings.mic_dc_block}, limiter={settings.mic_soft_limiter}."
    )
    last_processor_source = {"id": None}

    # --- Live mic-activity meter (display only; throttled) --------------
    frame_count = 0
    total_audio_bytes = 0
    rate_frame_count = 0
    rate_audio_bytes = 0
    last_meter_log = 0.0
    last_mic_rate_log = time.monotonic()
    last_reconnect_attempt = 0.0
    meter_unit = max(settings.vad_silence_threshold / 5.0, 1.0)

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()

            # --- Binary audio frame from the device --------------------
            data = message.get("bytes")
            if data is not None:
                frame_count += 1
                total_audio_bytes += len(data)
                rate_frame_count += 1
                rate_audio_bytes += len(data)
                rate_now = time.monotonic()
                if rate_now - last_mic_rate_log >= 2.0:
                    elapsed = max(rate_now - last_mic_rate_log, 0.001)
                    expected_bps = max(
                        in_rate * in_channels * settings.input_sample_width,
                        1,
                    )
                    audio_seconds = rate_audio_bytes / expected_bps
                    print(
                        f"[AUDIO] '{client_id}' mic rx-rate: "
                        f"{rate_frame_count / elapsed:.1f} fps, "
                        f"{rate_audio_bytes / elapsed:.0f} B/s, "
                        f"audio_rt={audio_seconds / elapsed:.2f}x, "
                        f"channels={in_channels}, frame={len(data)}B."
                    )
                    rate_frame_count = 0
                    rate_audio_bytes = 0
                    last_mic_rate_log = rate_now

                if settings.ignore_mic_while_speaking:
                    now = time.monotonic()
                    if session.is_responding or now < ignore_mic_until["time"]:
                        continue
                if raw_debug_recorder is not None:
                    raw_debug_recorder.add_frame(data)

                # Pick the loudest physical PDM pair and send only that pair to Realtime.
                # The 8-mic board arrives as four interleaved stereo pairs:
                # pair0 = ch0/ch1, pair1 = ch2/ch3, pair2 = ch4/ch5, pair3 = ch6/ch7.
                if in_channels >= 2 and settings.input_mic_channel is None:
                    mono, selected_pair, selected_channel, pair_rms, pair_dc = downmix_loudest_pair_to_mono_pcm16(
                        data,
                        in_channels,
                        mic_pair=settings.input_mic_pair,
                        preferred_pair=selected_mic_pair,
                        switch_margin=settings.input_pair_switch_margin,
                        bad_dc_limit=settings.input_bad_dc_limit,
                        mono_mode=settings.input_pair_mono_mode,
                    )
                    if (
                        settings.input_mic_pair is None
                        and pair_rms
                        and max(pair_rms) >= settings.local_vad_start_rms
                    ):
                        selected_mic_pair = selected_pair
                else:
                    mono = downmix_to_mono_pcm16(
                        data, in_channels, settings.input_mic_channel
                    )
                    selected_pair = None
                    selected_channel = settings.input_mic_channel
                    pair_rms = []
                    pair_dc = []

                rms = compute_rms(mono, settings.input_sample_width)
                now = time.monotonic()
                if now - last_meter_log >= 0.25:
                    last_meter_log = now
                    bars = min(int(rms / meter_unit), 20)
                    meter = "#" * bars + "-" * (20 - bars)
                    state = "SPEAKING" if rms >= settings.vad_silence_threshold else "silence "
                    pair_detail = ""
                    if selected_pair is not None:
                        pair_detail = (
                            f" pair={selected_pair} "
                            f"ch={selected_channel} "
                            f"ac=[{', '.join(f'{value:.0f}' for value in pair_rms)}] "
                            f"dc=[{', '.join(f'{value:.0f}' for value in pair_dc)}]"
                        )
                    print(
                        f"[AUDIO] '{client_id}' {state} "
                        f"rms={rms:6.0f} [{meter}] "
                        f"frame={len(data)}B total={total_audio_bytes // 1024}KB "
                        f"#{frame_count}{pair_detail}"
                    )

                # Resample mono device audio to the Realtime input rate, forward.
                processor_source = selected_channel if selected_channel is not None else selected_pair
                if last_processor_source["id"] != processor_source:
                    mic_processor.reset()
                    last_processor_source["id"] = processor_source
                cleaned_mono = mic_processor.process(mono)
                if raw_mic_recorder is not None:
                    raw_mic_recorder.add_frame(mono, selected_pair, rms)
                if clean_mic_recorder is not None:
                    clean_mic_recorder.add_frame(cleaned_mono, selected_pair, rms)
                pcm = resample_pcm16_mono(
                    cleaned_mono,
                    in_rate,
                    settings.realtime_input_rate,
                )
                if mic_recorder is not None:
                    mic_recorder.add_frame(pcm, selected_pair, rms)
                if device_turn_mode["enabled"]:
                    if not device_turn_mode["active"]:
                        device_turn_mode["turn_id"] += 1
                        device_turn_mode["active"] = True
                        device_turn_pcm.clear()
                        print(
                            f"[DEVICE_VAD] '{client_id}' recorded turn "
                            f"#{device_turn_mode['turn_id']} receive start."
                        )
                        if mic_recorder is not None:
                            mic_recorder.mark_speech_started()
                            if raw_mic_recorder is not None:
                                raw_mic_recorder.mark_speech_started()
                            if clean_mic_recorder is not None:
                                clean_mic_recorder.mark_speech_started()
                            if raw_debug_recorder is not None:
                                raw_debug_recorder.mark_speech_started()
                    device_turn_pcm.extend(pcm)
                    continue
                if settings.local_input_buffering and local_turn_buffer is not None:
                    started, completed = local_turn_buffer.add_frame(pcm, rms, selected_pair)
                    if started and mic_recorder is not None:
                        mic_recorder.mark_speech_started()
                        if raw_mic_recorder is not None:
                            raw_mic_recorder.mark_speech_started()
                        if clean_mic_recorder is not None:
                            clean_mic_recorder.mark_speech_started()
                        if raw_debug_recorder is not None:
                            raw_debug_recorder.mark_speech_started()
                    if completed is not None:
                        turn_id, turn_pcm, reason = completed
                        if mic_recorder is not None:
                            mic_recorder.mark_speech_stopped(f"local_{reason}")
                            if raw_mic_recorder is not None:
                                raw_mic_recorder.mark_speech_stopped(f"local_{reason}")
                            if clean_mic_recorder is not None:
                                clean_mic_recorder.mark_speech_stopped(f"local_{reason}")
                            if raw_debug_recorder is not None:
                                raw_debug_recorder.mark_speech_stopped(f"local_{reason}")
                        await user_turn_q.put((turn_id, turn_pcm, reason))
                    continue

                if not bridge.is_alive:
                    now = time.monotonic()
                    if now - last_reconnect_attempt >= 5.0:
                        last_reconnect_attempt = now
                        try:
                            await bridge.reconnect()
                        except Exception as exc:  # noqa: BLE001
                            print(
                                f"[REALTIME] '{client_id}' reconnect failed: {exc!r}"
                            )
                    if not bridge.is_alive:
                        continue
                await bridge.send_audio(pcm)
                continue

            # --- Text control message ----------------------------------
            text = message.get("text")
            if text is not None:
                print(f"[CONTROL] '{client_id}' sent text: {text!r}")
                # The device announces its mic format via a "hello" message;
                # honor it so we downmix/resample correctly.
                try:
                    payload = json.loads(text)
                except (ValueError, TypeError):
                    payload = None
                if isinstance(payload, dict) and payload.get("type") == "hello":
                    in_rate = int(payload.get("mic_rate", in_rate) or in_rate)
                    in_channels = int(payload.get("mic_channels", in_channels) or in_channels)
                    device_turn_mode["enabled"] = (
                        str(payload.get("mic_mode", "")).lower() == "mono_turn_buffer"
                    )
                    device_turn_mode["active"] = False
                    device_turn_pcm.clear()
                    rate_frame_count = 0
                    rate_audio_bytes = 0
                    last_mic_rate_log = time.monotonic()
                    if raw_mic_recorder is not None:
                        raw_mic_recorder.configure(in_rate)
                    if clean_mic_recorder is not None:
                        clean_mic_recorder.configure(in_rate)
                    if raw_debug_recorder is not None:
                        raw_debug_recorder.configure(in_rate, in_channels)
                    slot_bits = payload.get("speaker_slot_bits")
                    fmt = str(payload.get("format", "")).lower()
                    if slot_bits in (16, 32):
                        speaker.configure_output(int(slot_bits))
                    elif fmt == "s16le":
                        speaker.configure_output(16)
                    speaker_channels = "mono" if settings.output_channels == 1 else "stereo"
                    print(
                        f"[CONTROL] '{client_id}' hello: mic {in_rate}Hz x{in_channels}ch, "
                        f"speaker {settings.output_sample_rate}Hz s16le {speaker_channels}, "
                        f"chunk={speaker.chunk_bytes}B, "
                        f"device_turn_mode={device_turn_mode['enabled']}."
                    )
                elif isinstance(payload, dict) and payload.get("type") == "mic_turn_end":
                    if device_turn_mode["enabled"] and device_turn_mode["active"] and device_turn_pcm:
                        turn_id = int(payload.get("turn_id", device_turn_mode["turn_id"]) or device_turn_mode["turn_id"])
                        reason = str(payload.get("reason", "device_end") or "device_end")
                        board_bytes = int(payload.get("bytes", 0) or 0)
                        board_ms = int(payload.get("duration_ms", 0) or 0)
                        peak_rms = int(payload.get("peak_rms", 0) or 0)
                        pcm_bytes = bytes(device_turn_pcm)
                        duration = len(pcm_bytes) / (
                            settings.realtime_input_rate * settings.input_sample_width
                        )
                        print(
                            f"[DEVICE_VAD] '{client_id}' recorded turn #{turn_id} end "
                            f"({duration:.2f}s ai_pcm={len(pcm_bytes)}B, "
                            f"board={board_bytes}B/{board_ms}ms, peak_rms={peak_rms}, "
                            f"reason={reason})."
                        )
                        if mic_recorder is not None:
                            mic_recorder.mark_speech_stopped(f"device_{reason}")
                            if raw_mic_recorder is not None:
                                raw_mic_recorder.mark_speech_stopped(f"device_{reason}")
                            if clean_mic_recorder is not None:
                                clean_mic_recorder.mark_speech_stopped(f"device_{reason}")
                            if raw_debug_recorder is not None:
                                raw_debug_recorder.mark_speech_stopped(f"device_{reason}")
                        await user_turn_q.put((turn_id, pcm_bytes, f"device_{reason}"))
                        device_turn_pcm.clear()
                        device_turn_mode["active"] = False
                    else:
                        print(
                            f"[DEVICE_VAD] '{client_id}' ignored mic_turn_end "
                            f"(enabled={device_turn_mode['enabled']}, "
                            f"active={device_turn_mode['active']}, "
                            f"bytes={len(device_turn_pcm)})."
                        )
                continue

    except WebSocketDisconnect:
        print(f"[DISCONNECT] '{client_id}' closed the connection.")
    except Exception as exc:  # noqa: BLE001 - never let one client crash the server
        print(f"[ERROR] Unexpected error for '{client_id}': {exc!r}")
    finally:
        session.is_responding = False
        if device_turn_mode["active"] and device_turn_pcm:
            turn_id = int(device_turn_mode["turn_id"])
            pcm_bytes = bytes(device_turn_pcm)
            print(
                f"[DEVICE_VAD] '{client_id}' flushing recorded turn #{turn_id} "
                f"on disconnect ({len(pcm_bytes)}B)."
            )
            if mic_recorder is not None:
                mic_recorder.mark_speech_stopped("device_disconnect")
                if raw_mic_recorder is not None:
                    raw_mic_recorder.mark_speech_stopped("device_disconnect")
                if clean_mic_recorder is not None:
                    clean_mic_recorder.mark_speech_stopped("device_disconnect")
                if raw_debug_recorder is not None:
                    raw_debug_recorder.mark_speech_stopped("device_disconnect")
            await user_turn_q.put((turn_id, pcm_bytes, "device_disconnect"))
            device_turn_pcm.clear()
            device_turn_mode["active"] = False
        if local_turn_buffer is not None:
            completed = local_turn_buffer.flush("disconnect")
            if completed is not None:
                turn_id, turn_pcm, reason = completed
                if mic_recorder is not None:
                    mic_recorder.mark_speech_stopped(f"local_{reason}")
                    if raw_mic_recorder is not None:
                        raw_mic_recorder.mark_speech_stopped(f"local_{reason}")
                    if clean_mic_recorder is not None:
                        clean_mic_recorder.mark_speech_stopped(f"local_{reason}")
                    if raw_debug_recorder is not None:
                        raw_debug_recorder.mark_speech_stopped(f"local_{reason}")
                try:
                    await user_turn_q.put((turn_id, turn_pcm, reason))
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[LOCAL_VAD] '{client_id}' could not flush turn on close: {exc!r}"
                    )
        if mic_recorder is not None:
            mic_recorder.close()
        if raw_mic_recorder is not None:
            raw_mic_recorder.close()
        if clean_mic_recorder is not None:
            clean_mic_recorder.close()
        if raw_debug_recorder is not None:
            raw_debug_recorder.close()
        await user_turn_q.put(None)
        await user_turn_task
        await ai_audio_q.put(None)
        await worker_task
        await speaker.shutdown()
        await bridge.close()
        await manager.disconnect(client_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
