from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.audio import chunk_bytes, ensure_mono_pcm16, resample_pcm16_mono
from app.config import settings
from app.openai_ws import OpenAIRealtimeWS
from app.recording import MicSessionRecorder


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(
        "[STARTUP] Limi backend: "
        f"device_in={settings.device_input_rate}Hz, "
        f"device_out={settings.device_output_rate}Hz, "
        f"openai={settings.openai_realtime_model}/{settings.openai_voice}"
    )
    yield
    print("[SHUTDOWN] Limi backend stopped")


app = FastAPI(title="Limi Voice Backend", version="0.1.0", lifespan=lifespan)

active_devices: dict[str, WebSocket] = {}
active_devices_lock = asyncio.Lock()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.websocket("/ws/{device_id}")
async def device_ws(websocket: WebSocket, device_id: str) -> None:
    await websocket.accept()
    print(f"[DEVICE] {device_id}: connected")

    async with active_devices_lock:
        old_socket = active_devices.get(device_id)
        active_devices[device_id] = websocket
    if old_socket is not None and old_socket is not websocket:
        try:
            await old_socket.close(code=1012, reason="new connection for same device")
            print(f"[DEVICE] {device_id}: closed older duplicate connection")
        except Exception:  # noqa: BLE001
            pass

    send_q: asyncio.Queue[bytes | str | None] = asyncio.Queue(maxsize=settings.device_send_queue_chunks)
    send_lock = asyncio.Lock()
    audio_burst_remaining = settings.device_output_initial_burst_chunks
    speaker_response_active = False
    speaker_response_flushed = False
    pending_speaker_audio = bytearray()
    recorder = MicSessionRecorder(
        device_id,
        settings.backend_mic_recordings_dir,
        settings.device_input_rate,
        settings.realtime_input_rate,
    )
    stats = {
        "mic_frames": 0,
        "mic_bytes": 0,
        "speaker_frames": 0,
        "speaker_bytes": 0,
        "last_log": time.monotonic(),
    }

    async def send_device_text(text: str) -> None:
        await send_q.put(text)

    async def send_device_binary(payload: bytes) -> None:
        await send_q.put(payload)

    async def device_sender() -> None:
        nonlocal audio_burst_remaining
        chunk_delay = settings.device_output_chunk_ms / 1000.0
        while True:
            item = await send_q.get()
            try:
                if item is None:
                    return
                async with send_lock:
                    if isinstance(item, str):
                        await websocket.send_text(item)
                    else:
                        await websocket.send_bytes(item)
                if isinstance(item, bytes):
                    if audio_burst_remaining > 0:
                        audio_burst_remaining -= 1
                    else:
                        await asyncio.sleep(chunk_delay)
            finally:
                send_q.task_done()

    sender_task = asyncio.create_task(device_sender())

    async def ensure_speaker_started() -> None:
        nonlocal audio_burst_remaining, speaker_response_active
        if speaker_response_active:
            return
        speaker_response_active = True
        audio_burst_remaining = settings.device_output_initial_burst_chunks
        await send_device_text('{"type":"speaker_begin"}')

    async def queue_speaker_pcm(pcm_device: bytes) -> None:
        await ensure_speaker_started()
        if settings.device_output_channels != 1:
            raise RuntimeError("Only mono device output is supported in this baseline")
        for chunk in chunk_bytes(pcm_device, settings.device_output_chunk_bytes):
            stats["speaker_frames"] += 1
            stats["speaker_bytes"] += len(chunk)
            await send_device_binary(chunk)

    async def flush_speaker_response(reason: str) -> None:
        nonlocal speaker_response_active, speaker_response_flushed
        if speaker_response_flushed:
            return
        speaker_response_flushed = True
        if pending_speaker_audio:
            print(
                f"[STREAM] {device_id}: sending buffered AI response "
                f"{len(pending_speaker_audio)}B to speaker ({reason})"
            )
            await queue_speaker_pcm(bytes(pending_speaker_audio))
            pending_speaker_audio.clear()
            speaker_response_active = False
            await send_device_text('{"type":"speaker_end"}')

    async def on_ai_audio(pcm_24k_mono: bytes) -> None:
        pcm_device = resample_pcm16_mono(
            pcm_24k_mono,
            settings.realtime_output_rate,
            settings.device_output_rate,
        )
        if settings.device_buffer_response_audio:
            pending_speaker_audio.extend(pcm_device)
        else:
            await queue_speaker_pcm(pcm_device)

    async def on_ai_event(event: dict) -> None:
        nonlocal audio_burst_remaining, speaker_response_active, speaker_response_flushed
        etype = event.get("type", "")
        if etype in ("response.output_audio.delta", "response.audio.delta"):
            return
        if etype == "response.created":
            if settings.device_buffer_response_audio:
                pending_speaker_audio.clear()
                speaker_response_active = True
                speaker_response_flushed = False
            else:
                await ensure_speaker_started()
        elif etype == "response.output_item.added":
            if settings.device_buffer_response_audio:
                speaker_response_active = True
            else:
                await ensure_speaker_started()
        elif etype in ("response.done", "response.output_audio.done"):
            if settings.device_buffer_response_audio:
                await flush_speaker_response(etype)
            elif speaker_response_active:
                speaker_response_active = False
                await send_device_text('{"type":"speaker_end"}')

    bridge = OpenAIRealtimeWS(device_id, on_audio=on_ai_audio, on_event=on_ai_event)

    try:
        await bridge.connect()
        await send_device_text(
            json.dumps(
                {
                    "type": "backend_ready",
                    "device_input_rate": settings.device_input_rate,
                    "device_output_rate": settings.device_output_rate,
                    "device_output_chunk_ms": settings.device_output_chunk_ms,
                    "openai_transport": "websocket",
                    "firmware_wake_required": True,
                }
            )
        )

        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect
            if "bytes" in message and message["bytes"] is not None:
                payload = message["bytes"]
                stats["mic_frames"] += 1
                stats["mic_bytes"] += len(payload)
                mono = ensure_mono_pcm16(payload, settings.device_input_channels)
                ai_pcm = resample_pcm16_mono(
                    mono,
                    settings.device_input_rate,
                    settings.realtime_input_rate,
                )
                if settings.save_backend_mic:
                    recorder.write(mono, ai_pcm)
                await bridge.send_audio(ai_pcm)
            elif "text" in message and message["text"] is not None:
                text = message["text"]
                try:
                    event = json.loads(text)
                except json.JSONDecodeError:
                    event = {"type": "text", "text": text}
                etype = event.get("type")
                if etype == "hello":
                    print(f"[DEVICE] {device_id}: hello {event}")
                elif etype == "wake_detected":
                    print(f"[WAKE] {device_id}: firmware wake detected")
                    if settings.save_backend_mic:
                        recorder.start("wake_detected")
                elif etype == "wake_session_end":
                    print(f"[WAKE] {device_id}: firmware wake session ended")
                    if settings.save_backend_mic:
                        recorder.stop("wake_session_end")
                else:
                    print(f"[DEVICE] {device_id}: text {event}")

            now = time.monotonic()
            if now - stats["last_log"] >= 2.0:
                dt = now - stats["last_log"]
                print(
                    f"[AUDIO] {device_id}: mic={stats['mic_frames']/dt:.1f}fps "
                    f"{stats['mic_bytes']/dt:.0f}B/s, "
                    f"speaker={stats['speaker_frames']/dt:.1f}fps "
                    f"{stats['speaker_bytes']/dt:.0f}B/s q={send_q.qsize()}"
                )
                stats["mic_frames"] = 0
                stats["mic_bytes"] = 0
                stats["speaker_frames"] = 0
                stats["speaker_bytes"] = 0
                stats["last_log"] = now

    except WebSocketDisconnect:
        print(f"[DEVICE] {device_id}: disconnected")
    except Exception as exc:  # noqa: BLE001
        print(f"[DEVICE] {device_id}: session error {exc!r}")
    finally:
        async with active_devices_lock:
            if active_devices.get(device_id) is websocket:
                active_devices.pop(device_id, None)
        if settings.save_backend_mic:
            recorder.stop("disconnect")
        await bridge.close()
        await send_q.put(None)
        try:
            await sender_task
        except Exception:  # noqa: BLE001
            pass
        print(f"[DEVICE] {device_id}: closed")
