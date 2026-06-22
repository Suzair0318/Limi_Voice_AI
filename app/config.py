"""Application configuration.

All settings are loaded from environment variables (or a local ``.env`` file)
via :class:`pydantic_settings.BaseSettings`. Import the singleton ``settings``
object anywhere in the app instead of reading ``os.environ`` directly.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings.

    Values are read from the process environment first and fall back to the
    ``.env`` file located next to the project root. Unknown environment
    variables are ignored so the same environment can host multiple services.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- OpenAI / model configuration -----------------------------------
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    # GA audio model (replaces the retired "gpt-4o-audio-preview").
    # Alternatives: "gpt-audio-mini" (cheaper), or a pinned date e.g. "gpt-audio-2025-08-28".
    openai_model: str = Field("gpt-audio", alias="OPENAI_MODEL")
    openai_voice: str = Field("alloy", alias="OPENAI_VOICE")

    # --- Inbound audio (from the hardware board) ------------------------
    input_sample_rate: int = Field(16000, alias="INPUT_SAMPLE_RATE")
    input_channels: int = Field(1, alias="INPUT_CHANNELS")
    input_sample_width: int = Field(2, alias="INPUT_SAMPLE_WIDTH")  # bytes => 16-bit PCM

    # --- Outbound audio (back to the device) ----------------------------
    output_sample_rate: int = Field(48000, alias="OUTPUT_SAMPLE_RATE")
    output_channels: int = Field(2, alias="OUTPUT_CHANNELS")

    # --- Voice Activity Detection ---------------------------------------
    # RMS amplitude below which a chunk is considered "silence".
    vad_silence_threshold: float = Field(500.0, alias="VAD_SILENCE_THRESHOLD")
    # How long (seconds) of continuous silence ends a user turn.
    vad_silence_duration: float = Field(1.5, alias="VAD_SILENCE_DURATION")
    # Minimum amount of speech (seconds) required before we bother invoking the LLM.
    vad_min_speech_duration: float = Field(0.3, alias="VAD_MIN_SPEECH_DURATION")

    # --- Outbound streaming pacing --------------------------------------
    # Size of each outbound audio frame in milliseconds.
    output_chunk_ms: int = Field(20, alias="OUTPUT_CHUNK_MS")

    @property
    def output_chunk_bytes(self) -> int:
        """Number of bytes in a single outbound chunk of ``output_chunk_ms``."""
        bytes_per_sample = 2  # 16-bit PCM
        frames_per_chunk = int(self.output_sample_rate * self.output_chunk_ms / 1000)
        return frames_per_chunk * bytes_per_sample * self.output_channels


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    Cached so the ``.env`` file is parsed exactly once per process.
    """
    return Settings()  # type: ignore[call-arg]


# Module-level singleton for convenient importing: ``from app.config import settings``
settings: Settings = get_settings()
