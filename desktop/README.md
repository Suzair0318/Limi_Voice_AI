# Limi Voice Desktop Client

Python desktop app that simulates the **hardware pendant** against the `app/` backend.
Audio formats and WebSocket control messages match `app/config.py` defaults.

## Protocol (matches backend)

| Direction | Format |
|-----------|--------|
| Mic → server | 16 kHz mono, 16-bit PCM (`s16le`) |
| Server → speaker | 24 kHz mono, 16-bit PCM |
| Control | `hello`, `backend_ready`, `speaker_begin`, `speaker_end`, `wake_detected`, `wake_session_end` |

Turn-taking uses **OpenAI server-side VAD** on the backend — stream mic audio continuously while talking, then pause; the AI replies automatically.

## Prerequisites

- Python 3.11+
- Microphone and speakers (or headset)
- Backend running: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- Linux: `sudo apt install libportaudio2` (for sounddevice)

## Install

```bash
cd /var/www/Limi_Voice_AI
python -m venv .venv
source .venv/bin/activate
pip install -r desktop/requirements.txt
```

## Run

```bash
python -m desktop
```

Or:

```bash
python -m desktop.gui
```

## Usage

1. Enter server **Host**, **Port** (default `8000`), and **Device ID**.
2. Click **Connect** — sends `hello` and waits for `backend_ready`.
3. Click **Start Talking** — streams mic PCM to the backend.
4. Speak, then pause — the AI response plays through your speakers at **24 kHz mono**.
5. Optional: **Wake Detected** / **Wake Session End** simulate firmware wake events.

## Project layout

```
desktop/
├── config.py         # Audio defaults (16k in / 24k mono out)
├── protocol.py       # hello / wake JSON helpers
├── audio.py          # Mic capture + speaker playback (sounddevice)
├── voice_session.py  # Async WebSocket session
├── gui.py            # Tkinter UI
└── requirements.txt
```

## Remote VPS

Set Host to your VPS IP (e.g. `203.0.113.10`) and ensure port `8000` is open in the firewall.
