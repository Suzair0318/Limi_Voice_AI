"""Application configuration.

All settings are loaded from environment variables (or a local ``.env`` file)
via :class:`pydantic_settings.BaseSettings`. Import the singleton ``settings``
object anywhere in the app instead of reading ``os.environ`` directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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

    # --- OpenAI / Realtime model configuration --------------------------
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    # GA speech-to-speech model. Alternatives: a pinned date, or a preview model.
    openai_realtime_model: str = Field("gpt-realtime", alias="OPENAI_REALTIME_MODEL")
    openai_voice: str = Field("alloy", alias="OPENAI_VOICE")
    # System prompt / behaviour for the assistant.
    realtime_instructions: str = Field(
        "You are Limi, a friendly, concise real-time voice assistant running on "
        "a wearable pendant. Always reply in English. Keep every spoken reply to "
        "one short sentence, ideally under 10 words, unless the user explicitly "
        "asks for detail. If the transcript is unclear or appears to be another "
        "language, briefly ask them to repeat in English. If a greeting includes "
        "a request, answer the request instead of only greeting back. If the user "
        "asks for weather and no location is known, ask which city. Never mention "
        "that you are an AI model or describe your audio format. Treat 'hey Limi' "
        "as the wake phrase; if it is followed by a request, answer the request "
        "directly.",
        alias="REALTIME_INSTRUCTIONS",
    )
    # Sample rate (Hz) of the PCM we hand to the Realtime API. 24000 is the
    # documented standard for "audio/pcm"; we resample the device audio to it.
    realtime_input_rate: int = Field(24000, alias="REALTIME_INPUT_RATE")
    # Sample rate (Hz) the model generates output audio at. The GA API requires
    # this to be set explicitly; we resample it to the device's output rate.
    realtime_output_rate: int = Field(24000, alias="REALTIME_OUTPUT_RATE")
    # Server-side turn detection: "server_vad" (silence-based) or "semantic_vad"
    # (the model decides when you've finished a thought).
    realtime_vad_type: str = Field("server_vad", alias="REALTIME_VAD_TYPE")
    # For server_vad: ms of trailing silence before the turn is considered over.
    realtime_silence_ms: int = Field(900, alias="REALTIME_SILENCE_MS")

    # --- Inbound audio (from the hardware board) ------------------------
    input_sample_rate: int = Field(16000, alias="INPUT_SAMPLE_RATE")
    # The pendant sends 8-channel interleaved PCM; default matches the device
    # so the first frames are correct even before the "hello" message arrives.
    input_channels: int = Field(8, alias="INPUT_CHANNELS")
    input_sample_width: int = Field(2, alias="INPUT_SAMPLE_WIDTH")  # bytes => 16-bit PCM
    # When set (0-7), use only this mic channel instead of averaging all channels.
    # Useful if the board puts voice on one channel and silence/noise on the rest.
    input_mic_channel: int | None = Field(None, alias="INPUT_MIC_CHANNEL")
    # When set (0-3), use one physical stereo mic pair. Leave unset for automatic
    # loudest-pair selection with stickiness.
    input_mic_pair: int | None = Field(None, alias="INPUT_MIC_PAIR")
    # How to make one mono stream from the chosen stereo PDM pair. The local Mac
    # monitor sounded clean because it listened to the pair directly; for AI,
    # using the better single channel is usually cleaner than averaging both.
    input_pair_mono_mode: str = Field("average", alias="INPUT_PAIR_MONO_MODE")
    # Automatic pair selection only switches when another valid pair is this much
    # louder than the currently selected pair, avoiding frame-to-frame jumping.
    input_pair_switch_margin: float = Field(1.6, alias="INPUT_PAIR_SWITCH_MARGIN")
    # A pair with this much DC offset is treated as stuck/broken for auto-select.
    input_bad_dc_limit: float = Field(10000.0, alias="INPUT_BAD_DC_LIMIT")
    # Save every Realtime-detected user turn as a WAV file so we can hear exactly
    # what the backend forwarded after mic-pair selection.
    save_mic_turns: bool = Field(True, alias="SAVE_MIC_TURNS")
    mic_recordings_dir: Path = Field(Path("mic_recordings"), alias="MIC_RECORDINGS_DIR")
    mic_record_preroll_ms: int = Field(700, alias="MIC_RECORD_PREROLL_MS")
    mic_record_min_ms: int = Field(150, alias="MIC_RECORD_MIN_MS")
    # Debug capture for solving mic quality: save the exact raw multi-channel
    # board audio plus one WAV per physical channel for every local turn.
    save_mic_debug_channels: bool = Field(True, alias="SAVE_MIC_DEBUG_CHANNELS")
    # POC reliability mode: collect a complete local speech turn first, then
    # send that compiled turn to Realtime in one burst. This avoids fragmented
    # input when device/network packets arrive unevenly.
    local_input_buffering: bool = Field(True, alias="LOCAL_INPUT_BUFFERING")
    local_vad_start_rms: float = Field(180.0, alias="LOCAL_VAD_START_RMS")
    local_vad_stop_rms: float = Field(120.0, alias="LOCAL_VAD_STOP_RMS")
    local_vad_silence_ms: int = Field(900, alias="LOCAL_VAD_SILENCE_MS")
    local_vad_min_ms: int = Field(300, alias="LOCAL_VAD_MIN_MS")
    local_vad_max_ms: int = Field(8000, alias="LOCAL_VAD_MAX_MS")
    local_ai_send_chunk_ms: int = Field(100, alias="LOCAL_AI_SEND_CHUNK_MS")
    local_ai_tail_silence_ms: int = Field(250, alias="LOCAL_AI_TAIL_SILENCE_MS")
    # Same style of cleanup used by ws_audio_server.py's clean monitor path:
    # DC blocker -> noise gate -> gain -> soft limiter.
    mic_preprocess_gain: float = Field(8.0, alias="MIC_PREPROCESS_GAIN")
    mic_noise_gate_percent: float = Field(0.35, alias="MIC_NOISE_GATE_PERCENT")
    mic_dc_block: bool = Field(True, alias="MIC_DC_BLOCK")
    mic_soft_limiter: bool = Field(True, alias="MIC_SOFT_LIMITER")
    # Wake-word gate. The backend still uses Realtime for transcription, but it
    # buffers assistant audio and only releases it after the user transcript
    # contains one of these phrases.
    wake_word_enabled: bool = Field(True, alias="WAKE_WORD_ENABLED")
    wake_word_phrases: str = Field(
        "hey limi,hi limi,hey leemi,hi leemi,hey le mee,hi le mee,hey lemi,hi lemi",
        alias="WAKE_WORD_PHRASES",
    )
    wake_session_seconds: float = Field(20.0, alias="WAKE_SESSION_SECONDS")
    wake_require_each_turn: bool = Field(False, alias="WAKE_REQUIRE_EACH_TURN")

    # --- Outbound audio (back to the device) ----------------------------
    output_sample_rate: int = Field(8000, alias="OUTPUT_SAMPLE_RATE")
    output_channels: int = Field(1, alias="OUTPUT_CHANNELS")
    # ESP32 pendant WebSocket speaker path expects plain 16-bit stereo PCM
    # (same as ws_server.py / firmware hello "format":"s16le"). Set to 32 only if
    # firmware explicitly requests speaker_slot_bits:32.
    output_i2s_slot_bits: int = Field(16, alias="OUTPUT_I2S_SLOT_BITS")
    # Outbound WebSocket frame size. Must stay below the firmware limit (8192 B).
    # 2560 = 80 ms @ 16 kHz mono s16le. Lower downlink bandwidth matters more
    # than wideband output on the ESP32 cloud path, where network stalls cause
    # audible underruns.
    output_chunk_ms: int = Field(160, alias="OUTPUT_CHUNK_MS")
    output_max_ws_bytes: int = Field(5120, alias="OUTPUT_MAX_WS_BYTES")
    # Send the first few speaker chunks immediately so the ESP32 jitter buffer
    # can start playback from prebuffer instead of timing out one chunk at a time.
    output_initial_burst_chunks: int = Field(8, alias="OUTPUT_INITIAL_BURST_CHUNKS")

    @property
    def output_chunk_bytes(self) -> int:
        """Bytes in one paced outbound frame (matches firmware SPEAKER_FRAME_MS)."""
        bytes_per_sample = self.output_i2s_slot_bits // 8
        frames = int(self.output_sample_rate * self.output_chunk_ms / 1000)
        size = frames * bytes_per_sample * self.output_channels
        return min(size, self.output_max_ws_bytes)

    # --- Mic activity meter (display only) ------------------------------
    # RMS amplitude above which an inbound frame is logged as "SPEAKING". This
    # is purely for the terminal VU meter; actual turn-taking is handled by the
    # OpenAI server-side VAD.
    vad_silence_threshold: float = Field(500.0, alias="VAD_SILENCE_THRESHOLD")

    # --- Echo/barge-in handling ----------------------------------------
    # POC default: half-duplex. While we are sending model audio to the pendant,
    # ignore mic frames so the speaker output is not fed back into Realtime VAD.
    # Later, true barge-in should use AEC or a stricter local VAD.
    ignore_mic_while_speaking: bool = Field(True, alias="IGNORE_MIC_WHILE_SPEAKING")
    ignore_mic_tail_ms: int = Field(1200, alias="IGNORE_MIC_TAIL_MS")
    # Hybrid playback: start streaming model audio to the pendant immediately,
    # while firmware waits for its own prebuffer before it starts I2S playback.
    stream_speaker_live: bool = Field(True, alias="STREAM_SPEAKER_LIVE")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    Cached so the ``.env`` file is parsed exactly once per process.
    """
    return Settings()  # type: ignore[call-arg]


# Module-level singleton for convenient importing: ``from app.config import settings``
settings: Settings = get_settings()
