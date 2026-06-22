# Limi Voice AI Backend

Real-time voice backend for the **Limi AI Voice Pendant** — a wearable hardware
device that streams microphone audio to the server and plays back a spoken AI
response. The server runs a multimodal LangGraph agent on top of OpenAI's
`gpt-audio` model (audio in → audio out) over a single WebSocket per device.

## Architecture

```
Device mic ──(PCM 16-bit chunks)──▶ WebSocket /ws/{client_id}
                                          │
                                          ▼
                              VAD (silence detection)
                                          │ end-of-turn utterance
                                          ▼
                          wrap as WAV → base64 → LangGraph agent
                                          │ (gpt-audio, tools)
                                          ▼
                      model audio (WAV) → resample to 48kHz stereo
                                          │ 20ms paced chunks
                                          ▼
Device speaker ◀──(PCM frames)── WebSocket
```

### Pipeline summary
1. The hardware board connects to `/ws/{client_id}` and streams raw 16-bit PCM audio frames.
2. A lightweight RMS-based **Voice Activity Detector** buffers the utterance and detects end-of-turn from trailing silence.
3. The utterance is wrapped in a WAV container, base64-encoded, and sent to a **LangGraph** ReAct agent.
4. The agent invokes OpenAI's multimodal `gpt-audio` model (with tool support) and returns synthesized audio.
5. The response is resampled to 48kHz stereo and streamed back to the device in paced 20ms chunks.

## Project structure

```
voice_backend/
├── app/
│   ├── main.py            # FastAPI app + WebSocket entrypoint & audio pipeline
│   ├── config.py          # Pydantic settings loaded from .env
│   ├── connection.py      # WebSocket session manager (per-device state)
│   ├── agent/
│   │   ├── graph.py        # LangGraph agent (gpt-audio multimodal model)
│   │   └── tools.py        # Sample agent tools (weather, time, reminders)
│   └── audio/
│       └── vad.py          # RMS-based voice activity detection
├── frontend/
│   └── index.html         # Browser test console (mic capture + playback)
├── test_client.py         # Local mic/speaker WebSocket test client
├── requirements.txt
├── .env.example           # Copy to .env and add your OpenAI key
└── README.md
```

## Prerequisites

- Python 3.11+ (the VAD avoids the removed `audioop` module so 3.13 works too)
- **ffmpeg** installed on the host (required by `pydub` for audio conversion)
  - Windows: `winget install Gyan.FFmpeg` (or download from ffmpeg.org)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt-get install ffmpeg`
- An OpenAI API key with access to the `gpt-audio` model

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
copy .env.example .env      # Windows
# cp .env.example .env      # macOS / Linux
# then edit .env and set OPENAI_API_KEY
```

## Running

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# or simply:
python -m app.main
```

- Browser test console: <http://localhost:8000/>
- Health check: <http://localhost:8000/health>
- WebSocket endpoint: `ws://localhost:8000/ws/{client_id}`

### Local test client

`test_client.py` captures audio from your microphone and plays back the AI
response through your speakers using `sounddevice`:

```bash
python test_client.py
```

## Configuration

All settings live in `.env` (see `.env.example`) and are validated by
`app/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | _(required)_ | OpenAI API key |
| `OPENAI_MODEL` | `gpt-audio` | Multimodal audio model (`gpt-audio-mini` for lower cost) |
| `OPENAI_VOICE` | `alloy` | TTS voice |
| `INPUT_SAMPLE_RATE` | `16000` | Inbound audio sample rate (Hz) |
| `INPUT_CHANNELS` | `1` | Inbound channel count |
| `INPUT_SAMPLE_WIDTH` | `2` | Inbound bytes per sample (16-bit PCM) |
| `OUTPUT_SAMPLE_RATE` | `48000` | Outbound sample rate (Hz) |
| `OUTPUT_CHANNELS` | `2` | Outbound channel count |
| `VAD_SILENCE_THRESHOLD` | `500.0` | RMS level below which a chunk is silence |
| `VAD_SILENCE_DURATION` | `1.5` | Seconds of silence that ends a turn |
| `VAD_MIN_SPEECH_DURATION` | `0.3` | Minimum speech before invoking the model |
| `OUTPUT_CHUNK_MS` | `20` | Outbound frame size (ms) |

## WebSocket protocol

- **Client → Server:** binary frames of raw 16-bit PCM audio. Optional text
  control message `end_turn` forces an immediate end-of-turn flush.
- **Server → Client:**
  - `{"type": "response_start"}` JSON before audio playback
  - binary PCM frames (48kHz stereo 16-bit)
  - `{"type": "response_end"}` JSON when playback completes
  - `{"type": "error", "detail": "..."}` on failure

The connection is half-duplex: inbound audio is ignored while the server is
streaming a response.

## Notes

- Conversation memory is per-device, keyed by `client_id` via a LangGraph
  `MemorySaver` checkpointer (in-memory; resets on restart).
- The tools in `app/agent/tools.py` are deterministic stand-ins — replace their
  bodies with real integrations as needed.
