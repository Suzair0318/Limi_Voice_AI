"""RMS amplitude helper for the terminal mic-activity meter.

Turn-taking is now handled by the OpenAI Realtime server-side VAD, so this
module no longer implements end-of-turn detection. The only thing left is a
small, dependency-free RMS calculation used purely to render the "[AUDIO] ...
SPEAKING" VU meter in the server log.

Implemented without ``audioop`` (removed in Python 3.13). Only 16-bit audio is
supported, which matches the hardware contract (PCM 16-bit).
"""

from __future__ import annotations

import math


def compute_rms(pcm: bytes, sample_width: int) -> float:
    """Compute the RMS amplitude of a little-endian signed PCM buffer."""
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
        sample = int.from_bytes(pcm[i:i + 2], byteorder="little", signed=True)
        total_squares += sample * sample
        count += 1

    if count == 0:
        return 0.0
    return math.sqrt(total_squares / count)
