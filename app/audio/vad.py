"""Volume-based Voice Activity Detection (VAD).

This is a deliberately lightweight, dependency-free silence detector. It does
*not* try to be a neural VAD - instead it measures the RMS amplitude of each
incoming PCM chunk and uses a hangover timer: once the user has been speaking
and then stays below the silence threshold for ``vad_silence_duration``
seconds, the current turn is considered finished and the buffered utterance is
returned for processing.

The detector is fully synchronous/stateful so it can be driven from an async
WebSocket receive loop without any extra threads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from app.config import Settings


def compute_rms(pcm: bytes, sample_width: int) -> float:
    """Compute the RMS amplitude of a little-endian signed PCM buffer.

    Implemented without ``audioop`` (removed in Python 3.13). Only 16-bit
    audio is supported, which matches the hardware contract (PCM 16-bit).
    """
    if not pcm:
        return 0.0
    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM is supported, got sample_width={sample_width}")

    # Trim a dangling odd byte so we always read whole 16-bit samples.
    usable = len(pcm) - (len(pcm) % 2)
    if usable <= 0:
        return 0.0

    total_squares = 0
    count = 0
    for i in range(0, usable, 2):
        # Little-endian signed 16-bit sample.
        sample = int.from_bytes(pcm[i:i + 2], byteorder="little", signed=True)
        total_squares += sample * sample
        count += 1

    if count == 0:
        return 0.0
    return math.sqrt(total_squares / count)


@dataclass
class VadResult:
    """Outcome of feeding a single chunk to the detector."""

    # Full buffered utterance (raw PCM bytes) when a turn just ended, else None.
    utterance: Optional[bytes] = None
    # True on the chunk where speech first started (useful for logging/UX).
    speech_started: bool = False


class VoiceActivityDetector:
    """Per-connection stateful silence detector.

    Usage::

        vad = VoiceActivityDetector(settings)
        result = vad.process_chunk(chunk)
        if result.utterance is not None:
            await run_agent(result.utterance)
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._sample_rate = settings.input_sample_rate
        self._channels = settings.input_channels
        self._sample_width = settings.input_sample_width

        self._threshold = settings.vad_silence_threshold
        self._silence_duration = settings.vad_silence_duration
        self._min_speech_duration = settings.vad_min_speech_duration

        self._reset()

    def _reset(self) -> None:
        """Clear all buffers/timers back to the idle state."""
        self._buffer = bytearray()
        self._is_speaking = False
        self._trailing_silence_s = 0.0
        self._speech_duration_s = 0.0

    def _chunk_duration_s(self, chunk: bytes) -> float:
        """Duration in seconds represented by ``chunk`` of raw PCM."""
        bytes_per_frame = self._sample_width * self._channels
        if bytes_per_frame <= 0:
            return 0.0
        frames = len(chunk) / bytes_per_frame
        return frames / float(self._sample_rate)

    def process_chunk(self, chunk: bytes) -> VadResult:
        """Feed one raw PCM chunk; return a :class:`VadResult`.

        The chunk is appended to the running utterance buffer. Volume is
        measured to decide whether the user is still talking. When a speaking
        user goes quiet for longer than the configured hangover window, the
        buffered utterance is returned and internal state resets.
        """
        if not chunk:
            return VadResult()

        duration = self._chunk_duration_s(chunk)
        rms = compute_rms(chunk, self._sample_width)
        is_loud = rms >= self._threshold

        speech_started = False

        if is_loud:
            if not self._is_speaking:
                self._is_speaking = True
                speech_started = True
            self._buffer.extend(chunk)
            self._speech_duration_s += duration
            self._trailing_silence_s = 0.0
            return VadResult(speech_started=speech_started)

        # --- Chunk is silence ------------------------------------------
        if not self._is_speaking:
            # Idle silence before any speech: ignore it, don't grow the buffer.
            return VadResult()

        # We were speaking and just hit silence: keep buffering (the gap is part
        # of natural speech) but advance the hangover timer.
        self._buffer.extend(chunk)
        self._trailing_silence_s += duration

        if self._trailing_silence_s >= self._silence_duration:
            utterance = bytes(self._buffer)
            had_enough_speech = self._speech_duration_s >= self._min_speech_duration
            self._reset()
            if had_enough_speech:
                return VadResult(utterance=utterance)
            # Too little actual speech (e.g. a cough); discard and stay idle.
            return VadResult()

        return VadResult()

    def force_flush(self) -> Optional[bytes]:
        """Return whatever is buffered (if it qualifies as speech) and reset.

        Useful on disconnect or explicit end-of-turn signals from the client.
        """
        if self._is_speaking and self._speech_duration_s >= self._min_speech_duration:
            utterance = bytes(self._buffer)
            self._reset()
            return utterance
        self._reset()
        return None
