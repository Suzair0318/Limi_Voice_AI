"""Microphone capture and speaker playback for the desktop client."""

from __future__ import annotations

import queue
import threading
from typing import Callable

import numpy as np
import sounddevice as sd

from desktop.config import ClientConfig


def resample_pcm16_mono(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear resample mono int16 PCM (same algorithm as ``app.audio.stream``)."""
    if not pcm or src_rate == dst_rate:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16)
    if samples.size <= 1:
        return pcm
    dst_count = max(1, round(samples.size * dst_rate / src_rate))
    src_positions = np.arange(dst_count, dtype=np.float64) * (src_rate / dst_rate)
    left = np.floor(src_positions).astype(np.int64)
    left = np.clip(left, 0, samples.size - 2)
    frac = src_positions - left
    a = samples[left].astype(np.float64)
    b = samples[left + 1].astype(np.float64)
    out = np.round(a + (b - a) * frac).astype(np.int16)
    return out.tobytes()


class MicStreamer:
    """Capture PCM from the default microphone at the backend input rate."""

    def __init__(
        self,
        config: ClientConfig,
        on_pcm: Callable[[bytes], None],
        on_level: Callable[[float], None] | None = None,
    ) -> None:
        self._config = config
        self._on_pcm = on_pcm
        self._on_level = on_level
        self._stream: sd.InputStream | None = None
        self._capture_rate = config.device_input_rate

    @property
    def active(self) -> bool:
        return self._stream is not None and self._stream.active

    def start(self) -> None:
        if self.active:
            return
        blocksize = max(1, int(self._capture_rate * self._config.mic_chunk_ms / 1000))

        def callback(indata, _frames, _time, status) -> None:  # noqa: ANN001
            if status:
                return
            mono = indata[:, 0] if indata.shape[1] > 1 else indata.reshape(-1)
            pcm = mono.astype(np.int16, copy=False).tobytes()
            if self._on_level is not None:
                peak = float(np.max(np.abs(mono))) / 32768.0
                self._on_level(peak)
            self._on_pcm(pcm)

        self._stream = sd.InputStream(
            samplerate=self._capture_rate,
            channels=1,
            dtype="int16",
            blocksize=blocksize,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None


class SpeakerPlayer:
    """Play 24 kHz mono PCM chunks from the backend speaker stream."""

    def __init__(
        self,
        config: ClientConfig,
        on_level: Callable[[float], None] | None = None,
    ) -> None:
        self._config = config
        self._on_level = on_level
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=256)
        self._stream: sd.OutputStream | None = None
        self._pending = bytearray()
        self._lock = threading.Lock()
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def begin(self) -> None:
        if self._active:
            return
        self._active = True
        with self._lock:
            self._pending.clear()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        blocksize = max(1, int(self._config.device_output_rate * 20 / 1000))

        def callback(outdata, frames, _time, status) -> None:  # noqa: ANN001
            if status:
                outdata.fill(0)
                return
            need = frames * self._config.device_output_channels
            with self._lock:
                while len(self._pending) < need * 2:
                    try:
                        chunk = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if chunk is None:
                        break
                    self._pending.extend(chunk)
                if len(self._pending) >= need * 2:
                    raw = bytes(self._pending[: need * 2])
                    del self._pending[: need * 2]
                else:
                    raw = bytes(self._pending)
                    self._pending.clear()
                    raw += b"\x00" * (need * 2 - len(raw))
            samples = np.frombuffer(raw, dtype=np.int16, count=need)
            outdata[:, 0] = samples.astype(np.float32) / 32768.0
            if self._on_level is not None and samples.size:
                peak = float(np.max(np.abs(samples))) / 32768.0
                self._on_level(peak)

        self._stream = sd.OutputStream(
            samplerate=self._config.device_output_rate,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            callback=callback,
        )
        self._stream.start()

    def feed(self, pcm: bytes) -> None:
        if not self._active or not pcm:
            return
        try:
            self._queue.put_nowait(pcm)
        except queue.Full:
            pass

    def end(self) -> None:
        self._active = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        with self._lock:
            self._pending.clear()

    def close(self) -> None:
        self.end()
