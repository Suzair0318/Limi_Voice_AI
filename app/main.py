"""FastAPI application: WebSocket entrypoint for the voice pendant.

Pipeline per device:

    raw PCM chunks  ->  VAD (silence detection)  ->  end-of-turn
        ->  wrap utterance as WAV  ->  base64  ->  LangGraph agent
        ->  model audio (WAV)  ->  resample to 48kHz stereo
        ->  stream back to device in 20ms chunks
"""
# .\.venv\Scripts\Activate.ps1
from __future__ import annotations

import asyncio
import base64
import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from starlette.responses import Response
from langchain_core.messages import HumanMessage
from pydub import AudioSegment

from app.agent.graph import extract_audio_response, graph_app
from app.audio.vad import VoiceActivityDetector
from app.config import settings
from app.connection import DeviceSession, manager


# --------------------------------------------------------------------------- #
# Audio conversion helpers (pydub is blocking -> always run via asyncio.to_thread)
# --------------------------------------------------------------------------- #
def _pcm_to_wav_b64(pcm: bytes) -> str:
    """Wrap raw input PCM into a WAV container and base64-encode it."""
    segment = AudioSegment(
        data=pcm,
        sample_width=settings.input_sample_width,
        frame_rate=settings.input_sample_rate,
        channels=settings.input_channels,
    )
    buf = io.BytesIO()
    segment.export(buf, format="wav")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _wav_to_output_pcm(wav_bytes: bytes) -> bytes:
    """Decode the model's WAV output and resample to 48kHz stereo 16-bit PCM."""
    segment = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
    segment = (
        segment.set_frame_rate(settings.output_sample_rate)
        .set_channels(settings.output_channels)
        .set_sample_width(2)
    )
    return segment.raw_data


# --------------------------------------------------------------------------- #
# Agent invocation
# --------------------------------------------------------------------------- #
async def _run_agent(client_id: str, audio_b64: str) -> Optional[bytes]:
    """Send the user's audio turn to the graph and return AI audio (WAV bytes)."""
    human = HumanMessage(
        content=[
            {
                "type": "input_audio",
                "input_audio": {"data": audio_b64, "format": "wav"},
            }
        ]
    )
    config = {"configurable": {"thread_id": client_id}}

    print(f"[AGENT] Invoking LangGraph for '{client_id}'...")
    result = await graph_app.ainvoke({"messages": [human]}, config=config)

    final_message = result["messages"][-1]
    audio_bytes, transcript = extract_audio_response(final_message)
    print(f"[AGENT] '{client_id}' response transcript: {transcript!r}")
    return audio_bytes


async def _stream_audio_back(session: DeviceSession, wav_bytes: bytes) -> None:
    """Resample the AI audio and stream it to the device in paced 20ms chunks."""
    client_id = session.client_id
    try:
        pcm = await asyncio.to_thread(_wav_to_output_pcm, wav_bytes)
    except Exception as exc:  # noqa: BLE001 - ffmpeg/pydub failure shouldn't kill the loop
        print(f"[ERROR] Resampling failed for '{client_id}': {exc!r}")
        return

    chunk_size = settings.output_chunk_bytes
    chunk_delay = settings.output_chunk_ms / 1000.0
    total_chunks = (len(pcm) + chunk_size - 1) // max(chunk_size, 1)

    print(f"[STREAM] Streaming {len(pcm)} bytes (~{total_chunks} chunks) "
          f"to '{client_id}'...")

    await manager.send_json(session, {"type": "response_start"})

    for offset in range(0, len(pcm), chunk_size):
        chunk = pcm[offset:offset + chunk_size]
        ok = await manager.send_bytes(session, chunk)
        if not ok:
            print(f"[STREAM] Aborting stream to '{client_id}' (socket closed).")
            return
        # Pace the stream so the device receives audio roughly in real time.
        await asyncio.sleep(chunk_delay)

    await manager.send_json(session, {"type": "response_end"})
    print(f"[STREAM] Finished streaming to '{client_id}'.")


async def _handle_utterance(session: DeviceSession, utterance: bytes) -> None:
    """Full end-of-turn handling: encode -> agent -> stream response."""
    client_id = session.client_id
    session.is_responding = True
    try:
        audio_b64 = await asyncio.to_thread(_pcm_to_wav_b64, utterance)
        wav_bytes = await _run_agent(client_id, audio_b64)
        if wav_bytes:
            await _stream_audio_back(session, wav_bytes)
        else:
            print(f"[AGENT] No audio returned for '{client_id}'.")
            await manager.send_json(
                session, {"type": "error", "detail": "no_audio_response"}
            )
    except Exception as exc:  # noqa: BLE001 - protect the receive loop
        print(f"[ERROR] Utterance handling failed for '{client_id}': {exc!r}")
        await manager.send_json(session, {"type": "error", "detail": "agent_failure"})
    finally:
        session.is_responding = False


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown hooks."""
    print("[STARTUP] Voice backend starting. "
          f"Model={settings.openai_model}, voice={settings.openai_voice}.")
    yield
    await manager.shutdown()
    print("[SHUTDOWN] Voice backend stopped.")


app = FastAPI(title="Limi Voice AI Backend", version="0.1.0", lifespan=lifespan)

# Path to the browser test console (voice_backend/frontend/index.html).
FRONTEND_INDEX = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


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
    """Main WebSocket endpoint: one connection per hardware device."""
    session = await manager.connect(client_id, websocket)
    vad = VoiceActivityDetector(settings)

    try:
        while True:
            message = await websocket.receive()

            msg_type = message.get("type")
            if msg_type == "websocket.disconnect":
                raise WebSocketDisconnect()

            # --- Binary audio frame ------------------------------------
            data = message.get("bytes")
            if data is not None:
                # Half-duplex: ignore inbound audio while we are talking back.
                if session.is_responding:
                    continue

                result = vad.process_chunk(data)
                if result.speech_started:
                    print(f"[VAD] Speech started for '{client_id}'.")
                if result.utterance is not None:
                    print(f"[VAD] Silence detected -> end of turn for "
                          f"'{client_id}' ({len(result.utterance)} bytes).")
                    # Handle the turn concurrently so we keep draining the socket.
                    asyncio.create_task(_handle_utterance(session, result.utterance))
                continue

            # --- Text control message ----------------------------------
            text = message.get("text")
            if text is not None:
                print(f"[CONTROL] '{client_id}' sent text: {text!r}")
                if text.strip().lower() == "end_turn":
                    forced = vad.force_flush()
                    if forced and not session.is_responding:
                        asyncio.create_task(_handle_utterance(session, forced))
                continue

    except WebSocketDisconnect:
        print(f"[DISCONNECT] '{client_id}' closed the connection.")
    except Exception as exc:  # noqa: BLE001 - never let one client crash the server
        print(f"[ERROR] Unexpected error for '{client_id}': {exc!r}")
    finally:
        await manager.disconnect(client_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
