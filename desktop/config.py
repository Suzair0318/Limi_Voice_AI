"""Desktop client settings — defaults match ``app/config.py`` device audio pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClientConfig:
    """Voice packet layout aligned with the Limi backend WebSocket device protocol."""

    server_host: str = "127.0.0.1"
    server_port: int = 8000
    device_id: str = "desktop-client"

    # Device → server (mic)
    device_input_rate: int = 16000
    device_input_channels: int = 1

    # Server → device (speaker)
    device_output_rate: int = 24000
    device_output_channels: int = 1
    device_output_chunk_ms: int = 80

    # How often the desktop streams mic PCM (hardware-like pacing).
    mic_chunk_ms: int = 20
    sample_width: int = 2  # 16-bit PCM

    # UI log ring buffer size.
    max_log_lines: int = 500

    @property
    def ws_url(self) -> str:
        return f"ws://{self.server_host}:{self.server_port}/ws/{self.device_id}"

    @property
    def health_url(self) -> str:
        return f"http://{self.server_host}:{self.server_port}/health"

    @property
    def mic_chunk_bytes(self) -> int:
        frames = int(self.device_input_rate * self.mic_chunk_ms / 1000)
        return frames * self.device_input_channels * self.sample_width

    @property
    def speaker_chunk_bytes(self) -> int:
        frames = int(self.device_output_rate * self.device_output_chunk_ms / 1000)
        return frames * self.device_output_channels * self.sample_width

    def summary(self) -> str:
        return (
            f"mic {self.device_input_rate} Hz mono s16le · "
            f"speaker {self.device_output_rate} Hz mono s16le · "
            f"chunk {self.device_output_chunk_ms} ms"
        )


# Module-level default instance (GUI may copy and mutate fields).
default_config = ClientConfig()
