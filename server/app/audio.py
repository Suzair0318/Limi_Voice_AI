from __future__ import annotations

from collections.abc import Iterable


def _read_i16(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 2], "little", signed=True)


def _write_i16(out: bytearray, value: int) -> None:
    if value > 32767:
        value = 32767
    elif value < -32768:
        value = -32768
    out.extend(int(value).to_bytes(2, "little", signed=True))


def ensure_mono_pcm16(pcm: bytes, channels: int) -> bytes:
    if channels <= 1:
        return pcm
    frame_bytes = channels * 2
    frames = len(pcm) // frame_bytes
    out = bytearray(frames * 2)
    out.clear()
    for frame in range(frames):
        base = frame * frame_bytes
        total = 0
        for channel in range(channels):
            total += _read_i16(pcm, base + channel * 2)
        _write_i16(out, int(total / channels))
    return bytes(out)


def resample_pcm16_mono(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    if not pcm or src_rate == dst_rate:
        return pcm
    sample_count = len(pcm) // 2
    if sample_count <= 1:
        return pcm[: sample_count * 2]

    dst_count = max(1, round(sample_count * dst_rate / src_rate))
    out = bytearray(dst_count * 2)
    out.clear()

    # Stateless linear interpolation is enough for 20 ms voice frames and avoids
    # relying on stdlib audioop, which is gone from newer Python builds.
    step = src_rate / dst_rate
    for out_index in range(dst_count):
        src_pos = out_index * step
        left = int(src_pos)
        if left >= sample_count - 1:
            sample = _read_i16(pcm, (sample_count - 1) * 2)
        else:
            frac = src_pos - left
            a = _read_i16(pcm, left * 2)
            b = _read_i16(pcm, (left + 1) * 2)
            sample = round(a + (b - a) * frac)
        _write_i16(out, sample)
    return bytes(out)


def chunk_bytes(data: bytes, size: int) -> Iterable[bytes]:
    if size <= 0:
        yield data
        return
    for offset in range(0, len(data), size):
        chunk = data[offset : offset + size]
        if chunk:
            yield chunk
