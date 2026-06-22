"""Standalone WebSocket test client for the Limi voice backend.

It simulates the hardware pendant end-to-end:

1. Captures a user "turn" of audio - either recorded live from your microphone
   (``--mic``) or loaded from a WAV/MP3 file (``--file path``).
2. Resamples it to the server's expected input format (16kHz mono 16-bit PCM by
   default) and streams it over the WebSocket in paced 20ms chunks, exactly like
   the real board would.
3. Sends an ``end_turn`` control message so the agent processes the turn
   immediately (no need to wait on server-side silence detection).
4. Receives the model's streamed audio response (48kHz stereo PCM), saves it to
   ``response.wav`` and plays it back through your speakers.

Usage examples
--------------
    # Record 5 seconds from your mic, then hear the reply:
    python test_client.py --mic --seconds 5

    # Send an existing recording:
    python test_client.py --file my_question.wav

    # Point at a remote/other host:
    python test_client.py --mic --url ws://localhost:8000/ws/test-device

Requires: websockets, pydub, numpy, sounddevice (sounddevice only for mic
capture / playback; file->file still works without it).
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
import wave
from typing import List, Optional

import websockets
from pydub import AudioSegment

try:  # sounddevice is only needed for live mic capture / speaker playback
    import numpy as np
    import sounddevice as sd

    _HAS_AUDIO_IO = True
except Exception:  # noqa: BLE001
    _HAS_AUDIO_IO = False


# --- Must match the server's app/config.py defaults ----------------------
INPUT_SAMPLE_RATE = 16000
INPUT_CHANNELS = 1
OUTPUT_SAMPLE_RATE = 48000
OUTPUT_CHANNELS = 2
CHUNK_MS = 20
SAMPLE_WIDTH = 2  # 16-bit PCM


def _chunk_bytes(sample_rate: int, channels: int, chunk_ms: int) -> int:
    """Number of PCM bytes in one ``chunk_ms`` frame."""
    frames = int(sample_rate * chunk_ms / 1000)
    return frames * channels * SAMPLE_WIDTH


def record_from_mic(seconds: float) -> bytes:
    """Record ``seconds`` of mono 16kHz 16-bit PCM from the default microphone."""
    if not _HAS_AUDIO_IO:
        raise RuntimeError(
            "sounddevice/numpy not installed - cannot use --mic. "
            "Install with: pip install sounddevice numpy"
        )
    print(f"[CLIENT] Recording {seconds:.1f}s from microphone... speak now!")
    frames = int(seconds * INPUT_SAMPLE_RATE)
    recording = sd.rec(
        frames, samplerate=INPUT_SAMPLE_RATE, channels=INPUT_CHANNELS, dtype="int16"
    )
    sd.wait()
    print("[CLIENT] Recording finished.")
    return recording.tobytes()


def load_from_file(path: str) -> bytes:
    """Load any audio file and convert it to the server's input PCM format."""
    print(f"[CLIENT] Loading audio from '{path}'...")
    segment = AudioSegment.from_file(path)
    segment = (
        segment.set_frame_rate(INPUT_SAMPLE_RATE)
        .set_channels(INPUT_CHANNELS)
        .set_sample_width(SAMPLE_WIDTH)
    )
    print(f"[CLIENT] Loaded {segment.duration_seconds:.2f}s of audio.")
    return segment.raw_data


def save_wav(path: str, pcm: bytes, sample_rate: int, channels: int) -> None:
    """Write raw PCM out to a playable WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    print(f"[CLIENT] Saved response to '{path}'.")


def play_pcm(pcm: bytes, sample_rate: int, channels: int) -> None:
    """Play raw PCM through the default speakers (if sounddevice is available)."""
    if not _HAS_AUDIO_IO:
        print("[CLIENT] sounddevice not installed - skipping playback "
              "(response was still saved to disk).")
        return
    if not pcm:
        print("[CLIENT] No audio to play.")
        return
    print("[CLIENT] Playing response...")
    array = np.frombuffer(pcm, dtype=np.int16)
    if channels > 1:
        array = array.reshape(-1, channels)
    sd.play(array, samplerate=sample_rate)
    sd.wait()
    print("[CLIENT] Playback finished.")


async def run_session(url: str, pcm: bytes) -> None:
    """Connect, stream the captured turn, and collect the audio reply."""
    chunk_size = _chunk_bytes(INPUT_SAMPLE_RATE, INPUT_CHANNELS, CHUNK_MS)
    chunk_delay = CHUNK_MS / 1000.0

    print(f"[CLIENT] Connecting to {url} ...")
    async with websockets.connect(url, max_size=None) as ws:
        print("[CLIENT] Connected. Streaming audio turn...")

        # 1) Stream the user's audio in real-time-paced chunks.
        for offset in range(0, len(pcm), chunk_size):
            await ws.send(pcm[offset:offset + chunk_size])
            await asyncio.sleep(chunk_delay)

        # 2) Explicitly signal end-of-turn so the agent runs right away.
        print("[CLIENT] Sent full turn. Signalling 'end_turn'...")
        await ws.send("end_turn")

        # 3) Collect the streamed response until we see 'response_end'.
        audio_chunks: List[bytes] = []
        streaming = False
        try:
            while True:
                message = await asyncio.wait_for(ws.recv(), timeout=60.0)
                if isinstance(message, bytes):
                    if streaming:
                        audio_chunks.append(message)
                    continue
                # Text -> JSON control message.
                print(f"[CLIENT] Control message: {message}")
                if '"response_start"' in message:
                    streaming = True
                elif '"response_end"' in message:
                    break
                elif '"error"' in message:
                    print("[CLIENT] Server reported an error; aborting.")
                    break
        except asyncio.TimeoutError:
            print("[CLIENT] Timed out waiting for a response.")

    response_pcm = b"".join(audio_chunks)
    print(f"[CLIENT] Received {len(response_pcm)} bytes of audio response.")
    if response_pcm:
        save_wav("response.wav", response_pcm, OUTPUT_SAMPLE_RATE, OUTPUT_CHANNELS)
        play_pcm(response_pcm, OUTPUT_SAMPLE_RATE, OUTPUT_CHANNELS)
    else:
        print("[CLIENT] No audio received from the server.")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Limi voice backend test client.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--mic", action="store_true", help="Record from microphone.")
    source.add_argument("--file", type=str, help="Path to an audio file to send.")
    parser.add_argument(
        "--seconds", type=float, default=5.0,
        help="Seconds to record when using --mic (default: 5).",
    )
    parser.add_argument(
        "--url", type=str, default="ws://localhost:8000/ws/test-device",
        help="WebSocket URL of the backend.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.file:
        pcm = load_from_file(args.file)
    elif args.mic:
        pcm = record_from_mic(args.seconds)
    else:
        # Default to mic if available, otherwise instruct the user.
        if _HAS_AUDIO_IO:
            pcm = record_from_mic(args.seconds)
        else:
            print("No input source given. Use --file <path> (or install "
                  "sounddevice for --mic).")
            return 1

    if not pcm:
        print("[CLIENT] No audio captured; nothing to send.")
        return 1

    try:
        asyncio.run(run_session(args.url, pcm))
    except (ConnectionRefusedError, OSError) as exc:
        print(f"[CLIENT] Could not connect to the server: {exc!r}")
        print("[CLIENT] Is the backend running? "
              "Start it with: uvicorn app.main:app --reload")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
