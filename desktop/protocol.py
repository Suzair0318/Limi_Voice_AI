"""WebSocket control messages for the Limi device protocol."""

from __future__ import annotations

import json
from typing import Any

from desktop.config import ClientConfig


def build_hello(config: ClientConfig) -> str:
    """Handshake sent after connect (same fields the firmware / browser use)."""
    return json.dumps(
        {
            "type": "hello",
            "client": "desktop",
            "mic_rate": config.device_input_rate,
            "mic_channels": config.device_input_channels,
            "speaker_rate": config.device_output_rate,
            "speaker_channels": config.device_output_channels,
            "format": "s16le",
        }
    )


def build_wake_detected() -> str:
    return json.dumps({"type": "wake_detected"})


def build_wake_session_end() -> str:
    return json.dumps({"type": "wake_session_end"})


def parse_control_message(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"type": "text", "text": text}
    if not isinstance(payload, dict):
        return {"type": "unknown", "raw": payload}
    return payload


def event_label(event: dict[str, Any]) -> str:
    etype = event.get("type", "unknown")
    if etype == "backend_ready":
        return (
            f"backend_ready in={event.get('device_input_rate')} "
            f"out={event.get('device_output_rate')} "
            f"chunk_ms={event.get('device_output_chunk_ms')}"
        )
    if etype == "speaker_begin":
        return "speaker_begin"
    if etype == "speaker_end":
        return "speaker_end"
    if etype == "error":
        return f"error: {event.get('detail', event)}"
    return etype
