"""Lightweight, in-process PCM resampling for the real-time audio bridge.

The Realtime API speaks 16-bit mono PCM at 24kHz, while the hardware board
captures 16kHz mono and plays back 48kHz stereo. We cannot shell out to ffmpeg
per audio chunk (far too slow for a streaming loop), so these helpers do fast
NumPy linear-interpolation resampling entirely in memory.

All buffers are little-endian signed 16-bit PCM (``<i2``).
"""

from __future__ import annotations

import numpy as np

_DTYPE = "<i2"  # little-endian signed 16-bit
_INT16_MIN = -32768
_INT16_MAX = 32767


class MonoPcmProcessor:
    """Stateful cleanup for selected mono mic PCM.

    This mirrors the useful parts of ``ws_audio_server.py``'s monitor path:
    one-pole DC blocking, simple noise gate, software gain, and soft limiting.
    """

    def __init__(
        self,
        gain: float = 1.0,
        noise_gate_percent: float = 0.0,
        dc_block: bool = True,
        soft_limiter: bool = True,
    ) -> None:
        self.gain = gain
        self.noise_gate_percent = noise_gate_percent
        self.dc_block = dc_block
        self.soft_limiter = soft_limiter
        self._dc_prev_in = 0.0
        self._dc_prev_out = 0.0

    def process(self, pcm: bytes) -> bytes:
        if not pcm:
            return pcm

        samples = np.frombuffer(pcm, dtype=_DTYPE).astype(np.float32)
        if samples.size == 0:
            return b""

        if self.dc_block:
            samples = self._dc_block(samples)

        gate_gain = 1.0
        if self.noise_gate_percent > 0.0:
            avg_abs = float(np.mean(np.abs(samples)))
            threshold = 32768.0 * (self.noise_gate_percent / 100.0)
            if avg_abs < threshold:
                gate_gain = 0.0

        samples = samples * float(self.gain) * gate_gain

        if self.soft_limiter:
            samples = np.tanh(samples / 32768.0) * 32767.0

        return np.clip(samples, _INT16_MIN, _INT16_MAX).astype(_DTYPE).tobytes()

    def reset(self) -> None:
        self._dc_prev_in = 0.0
        self._dc_prev_out = 0.0

    def _dc_block(self, samples: np.ndarray) -> np.ndarray:
        # One-pole DC blocker: y[n] = x[n] - x[n-1] + R*y[n-1].
        r = 0.995
        out = np.empty_like(samples)
        prev_in = self._dc_prev_in
        prev_out = self._dc_prev_out
        for idx, sample in enumerate(samples):
            y = float(sample) - prev_in + r * prev_out
            out[idx] = y
            prev_in = float(sample)
            prev_out = y
        self._dc_prev_in = prev_in
        self._dc_prev_out = prev_out
        return out


def resample_pcm16_mono(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample mono 16-bit PCM from ``src_rate`` to ``dst_rate``.

    Uses linear interpolation. Per-chunk boundary artifacts are negligible for
    speech at the chunk sizes we stream.
    """
    if not pcm or src_rate == dst_rate:
        return pcm

    samples = np.frombuffer(pcm, dtype=_DTYPE).astype(np.float32)
    if samples.size == 0:
        return b""

    dst_n = max(int(round(samples.size * dst_rate / src_rate)), 1)
    src_idx = np.arange(samples.size, dtype=np.float64)
    dst_idx = np.linspace(0.0, samples.size - 1, num=dst_n)
    resampled = np.interp(dst_idx, src_idx, samples)
    return np.clip(resampled, _INT16_MIN, _INT16_MAX).astype(_DTYPE).tobytes()


def downmix_to_mono_pcm16(pcm: bytes, channels: int, mic_channel: int | None = None) -> bytes:
    """Convert multi-channel interleaved 16-bit PCM to mono.

    If ``mic_channel`` is set, take only that channel (0-based). Otherwise
    average all channels (best for a real mic array).
    """
    if channels <= 1 or not pcm:
        return pcm
    samples = np.frombuffer(pcm, dtype=_DTYPE)
    usable = (samples.size // channels) * channels
    if usable == 0:
        return b""
    frames = samples[:usable].reshape(-1, channels)
    if mic_channel is not None:
        ch = int(mic_channel)
        if ch < 0 or ch >= channels:
            raise ValueError(f"mic_channel={ch} out of range for {channels} channels")
        mono = frames[:, ch].astype(np.float32)
    else:
        mono = frames.mean(axis=1)
    return np.clip(mono, _INT16_MIN, _INT16_MAX).astype(_DTYPE).tobytes()


def downmix_loudest_pair_to_mono_pcm16(
    pcm: bytes,
    channels: int,
    mic_pair: int | None = None,
    preferred_pair: int | None = None,
    switch_margin: float = 1.6,
    bad_dc_limit: float = 10000.0,
    mono_mode: str = "best-channel",
) -> tuple[bytes, int, int, list[float], list[float]]:
    """Pick the loudest useful stereo PDM pair and return it as mono PCM.

    Channels are expected as interleaved pairs:
    pair 0 = channels 0/1, pair 1 = channels 2/3, etc.

    Selection uses AC RMS after subtracting each pair's DC offset. This avoids
    selecting a broken/noisy mic pair that sits at a huge constant level. If a
    preferred pair is provided, selection sticks to it unless another valid pair
    is clearly louder.

    Returns ``(mono_pcm, selected_pair_index, selected_channel,
    pair_ac_rms_values, pair_dc_values)``.
    """
    if channels < 2 or not pcm:
        return pcm, 0, 0, [], []

    samples = np.frombuffer(pcm, dtype=_DTYPE)
    usable = (samples.size // channels) * channels
    if usable == 0:
        return b"", 0, 0, [], []

    frames = samples[:usable].reshape(-1, channels).astype(np.float32)
    pair_count = channels // 2
    pair_monos: list[np.ndarray] = []
    pair_best_channels: list[np.ndarray] = []
    pair_best_channel_indexes: list[int] = []
    pair_ac_rms: list[float] = []
    pair_dc: list[float] = []
    pair_scores: list[float] = []

    for pair in range(pair_count):
        pair_samples = frames[:, pair * 2 : pair * 2 + 2]
        left = pair_samples[:, 0]
        right = pair_samples[:, 1]
        left_dc = float(left.mean())
        right_dc = float(right.mean())
        left_ac = left - left_dc
        right_ac = right - right_dc
        left_rms = float(np.sqrt(np.mean(left_ac * left_ac)))
        right_rms = float(np.sqrt(np.mean(right_ac * right_ac)))

        raw_mono = pair_samples.mean(axis=1)
        mono_dc = float(raw_mono.mean())
        ac_mono = raw_mono - mono_dc
        mono_rms = float(np.sqrt(np.mean(ac_mono * ac_mono)))

        if left_rms >= right_rms:
            best_channel = left_ac
            best_channel_index = pair * 2
            best_dc = left_dc
            best_rms = left_rms
        else:
            best_channel = right_ac
            best_channel_index = pair * 2 + 1
            best_dc = right_dc
            best_rms = right_rms

        pair_monos.append(ac_mono)
        pair_best_channels.append(best_channel)
        pair_best_channel_indexes.append(best_channel_index)
        pair_ac_rms.append(best_rms if mono_mode == "best-channel" else mono_rms)
        pair_dc.append(best_dc if mono_mode == "best-channel" else mono_dc)

        # A pair with huge DC offset is usually a bad/stuck mic line. Do not
        # let it win just because its absolute level is large.
        pair_bad = (
            abs(left_dc) > bad_dc_limit
            and abs(right_dc) > bad_dc_limit
        )
        pair_scores.append(0.0 if pair_bad else pair_ac_rms[-1])

    if mic_pair is not None:
        selected_pair = max(0, min(int(mic_pair), pair_count - 1))
    else:
        selected_pair = int(np.argmax(pair_scores)) if pair_scores else 0
        if preferred_pair is not None and 0 <= int(preferred_pair) < pair_count:
            preferred = int(preferred_pair)
            preferred_score = pair_scores[preferred]
            selected_score = pair_scores[selected_pair]
            if preferred_score > 0.0 and selected_score < preferred_score * switch_margin:
                selected_pair = preferred

    if pair_scores and pair_scores[selected_pair] <= 0.0:
        selected_pair = int(np.argmax(pair_ac_rms))

    if mono_mode == "average":
        mono = pair_monos[selected_pair] if pair_monos else frames[:, 0]
        selected_channel = selected_pair * 2
    else:
        mono = pair_best_channels[selected_pair] if pair_best_channels else frames[:, 0]
        selected_channel = (
            pair_best_channel_indexes[selected_pair]
            if pair_best_channel_indexes
            else selected_pair * 2
        )
    return (
        np.clip(mono, _INT16_MIN, _INT16_MAX).astype(_DTYPE).tobytes(),
        selected_pair,
        selected_channel,
        pair_ac_rms,
        pair_dc,
    )


def mono_to_stereo_pcm16(pcm: bytes) -> bytes:
    """Duplicate a mono 16-bit PCM buffer into interleaved stereo (L=R)."""
    if not pcm:
        return pcm
    mono = np.frombuffer(pcm, dtype=_DTYPE)
    stereo = np.repeat(mono, 2)  # [a, a, b, b, ...] => interleaved L/R
    return stereo.astype(_DTYPE).tobytes()


def pcm16_to_i2s32_slots(pcm: bytes) -> bytes:
    """Expand interleaved 16-bit PCM into 32-bit I2S slots (sample << 16).

    The ESP32 pendant configures I2S TX with 32-bit slot width and 16-bit
    data. The TAS5805M expects each sample in the upper 16 bits of a 32-bit
    word (standard Philips I2S left-justified in 32-bit slot).
    """
    if not pcm:
        return pcm
    samples = np.frombuffer(pcm, dtype=_DTYPE)
    slots = samples.astype(np.int32) << 16
    return slots.astype("<i4").tobytes()


def to_device_pcm(
    pcm: bytes,
    src_rate: int,
    dst_rate: int,
    dst_channels: int,
    i2s_slot_bits: int = 16,
) -> bytes:
    """Convert mono ``src_rate`` PCM into the device's speaker layout."""
    out = resample_pcm16_mono(pcm, src_rate, dst_rate)
    if dst_channels == 2:
        out = mono_to_stereo_pcm16(out)
    if i2s_slot_bits == 32:
        out = pcm16_to_i2s32_slots(out)
    return out
