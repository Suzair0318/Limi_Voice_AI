# Limi Voice AI Backend

Real-time voice backend for the **Limi AI Voice Pendant** — a wearable hardware
device that streams microphone audio to the server and plays back a spoken AI
response. The server bridges each device to OpenAI's **Realtime API**
(`gpt-realtime`) for low-latency, native speech-to-speech, over a single
WebSocket per device.

## Architecture

```
Device mic ──(16kHz PCM)──▶ WebSocket /ws/{client_id}
                                   │  resample 16k → 24k
                                   ▼
                       OpenAI Realtime API (gpt-realtime)
                       server-side VAD detects the turn,
                       streams speech-to-speech audio back
                                   │  resample 24k → 48k stereo
                                   ▼
Device speaker ◀──(48kHz stereo PCM)── WebSocket
```

### Pipeline summary
1. The board connects to `/ws/{client_id}` and streams raw 16-bit PCM audio frames.
2. The server opens one OpenAI Realtime WebSocket per device and forwards the
   audio (resampled to 24kHz) into the input buffer.
3. **OpenAI's server-side VAD** owns turn-taking — it detects when the user
   starts/stops speaking and when to respond. There is no local VAD tuning.
4. The model streams audio back; the server resamples it to 48kHz stereo and
   relays it to the device, framed with `response_start` / `response_end`.
5. **Barge-in** is supported: if the user speaks while the model is talking, the
   in-progress reply is cancelled and a `response_interrupted` is sent.

## Project structure

```
Limi_Voice_AI/
├── app/
│   ├── main.py            # FastAPI app + WebSocket bridge (device <-> OpenAI)
│   ├── config.py          # Pydantic settings loaded from .env
│   ├── connection.py      # WebSocket session manager (per-device state)
│   ├── realtime.py        # OpenAI Realtime API WebSocket bridge
│   └── audio/
│       ├── resample.py     # NumPy PCM resampling (16k/24k/48k, mono<->stereo)
│       └── vad.py          # RMS helper for the terminal mic-activity meter
├── frontend/
│   └── index.html         # Browser test console (mic capture + playback)
├── test_client.py         # Local mic/speaker WebSocket test client
├── requirements.txt
├── .env.example           # Copy to .env and add your OpenAI key
└── README.md
```

## Prerequisites

- Python 3.11+ (3.13 works too — no `audioop` dependency)
- An OpenAI API key with access to the Realtime API (`gpt-realtime`)
- ffmpeg is **only** needed for the optional `test_client.py --file` decoding,
  not for the server itself.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # then edit .env and set OPENAI_API_KEY
```

## Running

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# or:
python -m app.main
```

- Browser test console: <http://localhost:8000/>
- Health check: <http://localhost:8000/health>
- WebSocket endpoint: `ws://localhost:8000/ws/{client_id}`

## Configuration

All settings live in `.env` (see `.env.example`) and are validated by
`app/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | _(required)_ | OpenAI API key |
| `OPENAI_REALTIME_MODEL` | `gpt-realtime` | Realtime speech-to-speech model |
| `OPENAI_VOICE` | `alloy` | Output voice |
| `REALTIME_INSTRUCTIONS` | _(Limi prompt)_ | System prompt / personality |
| `REALTIME_VAD_TYPE` | `server_vad` | `server_vad` or `semantic_vad` |
| `REALTIME_SILENCE_MS` | `500` | (server_vad) trailing silence to end a turn |
| `REALTIME_INPUT_RATE` | `24000` | Sample rate sent to the Realtime API |
| `INPUT_SAMPLE_RATE` | `16000` | Inbound audio sample rate (Hz) |
| `INPUT_CHANNELS` | `1` | Inbound channel count |
| `INPUT_SAMPLE_WIDTH` | `2` | Inbound bytes per sample (16-bit PCM) |
| `OUTPUT_SAMPLE_RATE` | `48000` | Outbound sample rate (Hz) |
| `OUTPUT_CHANNELS` | `2` | Outbound channel count |
| `VAD_SILENCE_THRESHOLD` | `500.0` | Mic-meter display threshold only (not turn-taking) |

## WebSocket protocol

- **Client → Server:** binary frames of raw 16-bit PCM audio, streamed
  continuously. (Legacy text control messages like `end_turn` are ignored —
  turn-taking is automatic via the server-side VAD.)
- **Server → Client:**
  - `{"type": "response_start"}` before audio playback
  - binary PCM frames (48kHz stereo 16-bit)
  - `{"type": "response_end"}` when the reply completes
  - `{"type": "response_interrupted"}` if the user barges in
  - `{"type": "error", "detail": "..."}` on failure

## Notes

- Conversation state is held inside the OpenAI Realtime session for the life of
  the WebSocket connection. A Realtime session lasts up to ~60 minutes.
- This server keeps per-device state in memory in a single process — run **one**
  instance (no multi-worker/cluster mode), since each device's socket and its
  OpenAI session must stay on the same process.
