from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    openai_realtime_model: str = Field("gpt-realtime", alias="OPENAI_REALTIME_MODEL")
    openai_voice: str = Field("alloy", alias="OPENAI_VOICE")
    realtime_input_rate: int = Field(24000, alias="REALTIME_INPUT_RATE")
    realtime_output_rate: int = Field(24000, alias="REALTIME_OUTPUT_RATE")
    realtime_vad_type: str = Field("server_vad", alias="REALTIME_VAD_TYPE")
    realtime_silence_ms: int = Field(650, alias="REALTIME_SILENCE_MS")
    realtime_instructions: str = Field(
        "You are Limi, a concise friendly voice assistant.",
        alias="REALTIME_INSTRUCTIONS",
    )

    device_input_rate: int = Field(16000, alias="DEVICE_INPUT_RATE")
    device_input_channels: int = Field(1, alias="DEVICE_INPUT_CHANNELS")
    device_output_rate: int = Field(24000, alias="DEVICE_OUTPUT_RATE")
    device_output_channels: int = Field(1, alias="DEVICE_OUTPUT_CHANNELS")
    device_output_chunk_ms: int = Field(80, alias="DEVICE_OUTPUT_CHUNK_MS")
    device_output_initial_burst_chunks: int = Field(16, alias="DEVICE_OUTPUT_INITIAL_BURST_CHUNKS")
    device_send_queue_chunks: int = Field(2048, alias="DEVICE_SEND_QUEUE_CHUNKS")
    device_buffer_response_audio: bool = Field(True, alias="DEVICE_BUFFER_RESPONSE_AUDIO")
    save_backend_mic: bool = Field(True, alias="SAVE_BACKEND_MIC")
    backend_mic_recordings_dir: Path = Field(
        Path("backend_mic_recordings"),
        alias="BACKEND_MIC_RECORDINGS_DIR",
    )

    @property
    def device_output_chunk_bytes(self) -> int:
        return (
            self.device_output_rate
            * self.device_output_channels
            * 2
            * self.device_output_chunk_ms
            // 1000
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


settings = get_settings()
