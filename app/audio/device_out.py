"""Paced, chunked outbound audio for the ESP32 voice pendant.

The firmware WebSocket handler drops any binary frame larger than 8192 bytes
(``MAX_WS_AUDIO_BYTES``) and disconnects. OpenAI Realtime deltas are often
much larger after resampling to 48 kHz stereo, so we buffer and send fixed
20 ms frames with real-time pacing — matching ``SPEAKER_FRAME_MS`` in firmware.

Pacing runs in a **background task** so it never blocks the OpenAI receive
loop (blocking there causes ``keepalive ping timeout`` on the upstream socket).
"""

from __future__ import annotations

import asyncio

from app.audio.resample import to_device_pcm
from app.config import settings
from app.connection import DeviceSession, manager


class DeviceOutboundStreamer:
    """Buffer model audio, convert to device format, send paced WS binary frames."""

    def __init__(
        self,
        session: DeviceSession,
        client_id: str,
        i2s_slot_bits: int | None = None,
    ) -> None:
        self._session = session
        self._client_id = client_id
        self._buffer = bytearray()
        self._i2s_slot_bits = i2s_slot_bits or settings.output_i2s_slot_bits
        self._total_sent = 0
        self._chunks_sent = 0
        self._deltas_seen = 0
        self._rate_chunks = 0
        self._rate_bytes = 0
        self._rate_send_ms = 0.0
        self._underruns = 0
        self._playing = False
        self._finishing = False
        self._running = True
        self._pacer_task: asyncio.Task | None = None
        self._buffer_lock = asyncio.Lock()
        self._auto_start = settings.stream_speaker_live
        self._refresh_chunk_size()
        self._prebuffer_chunks = max(int(settings.output_initial_burst_chunks), 1)
        self._rate_log_at = asyncio.get_running_loop().time() + 1.0
        print(
            f"[STREAM] '{self._client_id}' pacer: "
            f"chunk={self._chunk_bytes}B, delay={self._chunk_delay * 1000:.1f}ms, "
            f"prebuffer={self._prebuffer_chunks} chunks, "
            f"initial_burst={settings.output_initial_burst_chunks} chunks, "
            f"mode={'live-fast' if self._auto_start else 'buffered'}."
        )

    def configure_output(self, i2s_slot_bits: int) -> None:
        """Switch slot width (16 for browser, 32 for ESP32 pendant)."""
        if i2s_slot_bits in (16, 32) and i2s_slot_bits != self._i2s_slot_bits:
            self._i2s_slot_bits = i2s_slot_bits
            self._refresh_chunk_size()
            print(
                f"[STREAM] '{self._client_id}' output format: "
                f"{settings.output_sample_rate}Hz, {self._i2s_slot_bits}-bit slots, "
                f"chunk={self._chunk_bytes}B, delay={self._chunk_delay * 1000:.1f}ms."
            )

    def _refresh_chunk_size(self) -> None:
        bytes_per_sample = self._i2s_slot_bits // 8
        bytes_per_frame = bytes_per_sample * settings.output_channels
        requested_frames = max(
            int(settings.output_sample_rate * settings.output_chunk_ms / 1000),
            1,
        )
        max_ws_frames = max(settings.output_max_ws_bytes // bytes_per_frame, 1)

        self._chunk_frames = min(requested_frames, max_ws_frames)
        self._chunk_bytes = self._chunk_frames * bytes_per_frame
        self._chunk_delay = self._chunk_frames / settings.output_sample_rate

    @property
    def total_sent(self) -> int:
        return self._total_sent

    @property
    def chunk_bytes(self) -> int:
        return self._chunk_bytes

    async def has_pending_audio(self) -> bool:
        """Return whether a response has audio buffered or currently draining."""
        async with self._buffer_lock:
            return bool(self._buffer or self._playing)

    async def feed_mono_pcm(self, pcm_mono: bytes, src_rate: int) -> None:
        """Buffer one OpenAI delta; the background pacer sends 20 ms frames."""
        if not pcm_mono or not self._running:
            return
        device_pcm = to_device_pcm(
            pcm_mono,
            src_rate,
            settings.output_sample_rate,
            settings.output_channels,
            i2s_slot_bits=self._i2s_slot_bits,
        )
        async with self._buffer_lock:
            self._buffer.extend(device_pcm)
            buffered = len(self._buffer)
        self._deltas_seen += 1
        if self._deltas_seen <= 3:
            print(
                f"[STREAM] '{self._client_id}' delta {self._deltas_seen}: "
                f"in={len(pcm_mono)}B @{src_rate}Hz -> out={len(device_pcm)}B, "
                f"buffer={buffered}B."
            )
        if self._auto_start:
            await self._drain_buffer_fast(include_partial=False)

    async def _drain_buffer_fast(self, *, include_partial: bool, label: str | None = None) -> None:
        """Send queued chunks as fast as the device WebSocket accepts them.

        Firmware owns the actual I2S pacing and prebuffer. In live mode the
        backend should fill the firmware queue quickly, not throttle itself to
        the audio clock and leave no room for network jitter.
        """
        if label:
            async with self._buffer_lock:
                buffered = len(self._buffer)
            if buffered:
                print(f"[STREAM] '{self._client_id}' {label}: {buffered}B queued.")

        flush_start = asyncio.get_running_loop().time()
        flushed_chunks = 0
        flushed_bytes = 0
        while True:
            chunk: bytes | None = None
            async with self._buffer_lock:
                if len(self._buffer) >= self._chunk_bytes:
                    chunk = bytes(self._buffer[: self._chunk_bytes])
                    del self._buffer[: self._chunk_bytes]
                elif include_partial and self._buffer:
                    chunk = bytes(self._buffer)
                    self._buffer.clear()
            if chunk is None:
                break
            await self._send_chunk(chunk)
            flushed_chunks += 1
            flushed_bytes += len(chunk)
            if len(chunk) < self._chunk_bytes:
                break

        if label and flushed_chunks:
            elapsed = max(asyncio.get_running_loop().time() - flush_start, 0.001)
            rate = flushed_bytes / elapsed
            print(
                f"[STREAM] '{self._client_id}' {label} complete: "
                f"{flushed_chunks} chunks, {flushed_bytes}B sent in "
                f"{elapsed:.2f}s ({rate:.0f} B/s)."
            )

    def _ensure_pacer(self) -> None:
        if self._pacer_task is None or self._pacer_task.done():
            self._pacer_task = asyncio.create_task(self._pacer_loop())

    async def _pacer_loop(self) -> None:
        """Send full chunks at real-time pace without blocking the caller."""
        next_send_at = asyncio.get_running_loop().time()
        burst_remaining = 0
        try:
            while self._running:
                chunk: bytes | None = None
                async with self._buffer_lock:
                    buffered = len(self._buffer)
                    if not self._playing:
                        ready_to_start = self._auto_start or self._finishing
                        if ready_to_start and buffered >= self._chunk_bytes * self._prebuffer_chunks:
                            self._playing = True
                            print(
                                f"[STREAM] '{self._client_id}' playback start: "
                                f"buffer={buffered}B."
                            )
                            burst_remaining = max(
                                int(settings.output_initial_burst_chunks) - 1,
                                0,
                            )
                    if self._playing and len(self._buffer) >= self._chunk_bytes:
                        chunk = bytes(self._buffer[: self._chunk_bytes])
                        del self._buffer[: self._chunk_bytes]
                    elif self._playing and self._finishing and self._buffer:
                        chunk = bytes(self._buffer)
                        self._buffer.clear()
                    elif self._playing and not self._buffer:
                        self._playing = False
                        if self._finishing:
                            self._finishing = False
                            print(
                                f"[STREAM] '{self._client_id}' paced playback drained."
                            )
                        else:
                            self._underruns += 1
                            print(
                                f"[STREAM] '{self._client_id}' playback underrun "
                                f"#{self._underruns}; waiting for prebuffer."
                            )
                if chunk is not None:
                    await self._send_chunk(chunk)
                    if burst_remaining > 0:
                        burst_remaining -= 1
                        next_send_at = asyncio.get_running_loop().time()
                    else:
                        next_send_at += self._chunk_delay
                        now = asyncio.get_running_loop().time()
                        if next_send_at > now:
                            await asyncio.sleep(next_send_at - now)
                        else:
                            # If the socket send or event loop fell behind, do not add
                            # extra sleep. Catch up until we are back on the audio clock.
                            next_send_at = now
                else:
                    next_send_at = asyncio.get_running_loop().time()
                    await asyncio.sleep(0.005)
        except asyncio.CancelledError:
            raise

    async def flush(self) -> None:
        """Drain any remaining queued audio to the device WebSocket."""
        if self._pacer_task is not None and not self._pacer_task.done():
            self._pacer_task.cancel()
            try:
                await self._pacer_task
            except asyncio.CancelledError:
                pass
            self._pacer_task = None

        if not self._auto_start:
            await self._drain_buffer_fast(
                include_partial=True,
                label="complete-response buffered send",
            )
            return

        await self._drain_buffer_fast(
            include_partial=True,
            label="live-response tail send",
        )

    def clear(self) -> None:
        """Drop buffered audio (barge-in / interrupt)."""
        self._buffer.clear()
        self._playing = False
        self._finishing = False

    async def clear_async(self) -> None:
        """Drop buffered audio under lock (safe during concurrent feed)."""
        async with self._buffer_lock:
            self._buffer.clear()
            self._playing = False
            self._finishing = False

    async def shutdown(self) -> None:
        """Stop the background pacer."""
        self._running = False
        if self._pacer_task is not None:
            self._pacer_task.cancel()
            try:
                await self._pacer_task
            except asyncio.CancelledError:
                pass
            self._pacer_task = None

    async def _send_chunk(self, chunk: bytes) -> None:
        if len(chunk) > settings.output_max_ws_bytes:
            for offset in range(0, len(chunk), settings.output_max_ws_bytes):
                part = chunk[offset : offset + settings.output_max_ws_bytes]
                ok = await manager.send_bytes(self._session, part)
                if not ok:
                    return
                self._total_sent += len(part)
                self._chunks_sent += 1
            return
        send_start = asyncio.get_running_loop().time()
        ok = await manager.send_bytes(self._session, chunk)
        send_ms = (asyncio.get_running_loop().time() - send_start) * 1000.0
        if ok:
            self._total_sent += len(chunk)
            self._chunks_sent += 1
            self._rate_chunks += 1
            self._rate_bytes += len(chunk)
            self._rate_send_ms += send_ms
            if self._chunks_sent == 1:
                print(
                    f"[STREAM] '{self._client_id}' first speaker frame "
                    f"({len(chunk)}B, slot={self._i2s_slot_bits}-bit)."
                )
            now = asyncio.get_running_loop().time()
            if now >= self._rate_log_at:
                async with self._buffer_lock:
                    buffered = len(self._buffer)
                avg_send_ms = self._rate_send_ms / max(self._rate_chunks, 1)
                print(
                    f"[STREAM] '{self._client_id}' send-rate: "
                    f"{self._rate_chunks} chunks/s, {self._rate_bytes} B/s, "
                    f"avg_send={avg_send_ms:.1f}ms, "
                    f"buffer={buffered}B, total={self._total_sent}B."
                )
                self._rate_chunks = 0
                self._rate_bytes = 0
                self._rate_send_ms = 0.0
                self._rate_log_at = now + 1.0
