#!/usr/bin/env python3
"""Tiny WebSocket audio server for the Voice Pendant POC.

Dependencies:
    python3 -m pip install websockets sounddevice

Examples:
    python3 server/ws_audio_server.py --play-mic --monitor-gain 8 --test-tone
    python3 server/ws_audio_server.py --play-mic --monitor-gain 8 --noise-gate 1.0 --dc-block
    python3 server/ws_audio_server.py --play-mic --monitor-mode loudest-pair --monitor-gain 8
    python3 server/ws_audio_server.py --play-mic --wav speaker_test_48k_stereo.wav
    python3 server/ws_audio_server.py --save-mic mic_capture.wav --test-tone
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import signal
import struct
import sys
import time
import wave
from pathlib import Path

try:
    import websockets
except ImportError as exc:  # pragma: no cover - user environment check
    raise SystemExit("Missing dependency: python3 -m pip install websockets") from exc

try:
    import sounddevice as sd
except ImportError:
    sd = None


SPEAKER_RATE = 48_000
SPEAKER_CHANNELS = 2
SAMPLE_WIDTH = 2


def generate_tone(duration_s: float = 2.0, freq_hz: float = 440.0, amplitude: float = 0.20) -> bytes:
    frames = bytearray()
    total = int(SPEAKER_RATE * duration_s)
    for n in range(total):
        # Fade in/out avoids hard clicks during first board tests.
        fade = min(n / 1200, (total - n) / 1200, 1.0)
        sample = int(32767 * amplitude * fade * math.sin(2.0 * math.pi * freq_hz * n / SPEAKER_RATE))
        frames += struct.pack("<hh", sample, sample)
    return bytes(frames)


def load_wav_as_s16le_stereo(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width != SAMPLE_WIDTH:
        raise ValueError(f"{path} must be 16-bit PCM WAV, got sample width {sample_width}")
    if rate != SPEAKER_RATE:
        raise ValueError(f"{path} must be {SPEAKER_RATE} Hz for this POC, got {rate} Hz")
    if channels == SPEAKER_CHANNELS:
        return frames
    if channels == 1:
        mono = memoryview(frames).cast("h")
        out = bytearray()
        for sample in mono:
            out += struct.pack("<hh", sample, sample)
        return bytes(out)
    raise ValueError(f"{path} must be mono or stereo, got {channels} channels")


async def send_speaker_audio(websocket, audio: bytes, loop: bool) -> None:
    chunk = int(SPEAKER_RATE * 0.020) * SPEAKER_CHANNELS * SAMPLE_WIDTH
    await asyncio.sleep(1.0)
    while True:
        print(f"Sending {len(audio) / (SPEAKER_RATE * SPEAKER_CHANNELS * SAMPLE_WIDTH):.2f}s speaker audio")
        for offset in range(0, len(audio), chunk):
            await websocket.send(audio[offset : offset + chunk])
            await asyncio.sleep(0.020)
        if not loop:
            return
        await asyncio.sleep(1.0)


class MicSink:
    def __init__(
        self,
        play: bool,
        save_path: Path | None,
        monitor_mode: str,
        monitor_gain: float,
        monitor_pair: int,
        print_levels: bool,
        noise_gate: float,
        dc_block: bool,
        soft_limiter: bool,
    ):
        self.play = play
        self.save_path = save_path
        self.monitor_mode = monitor_mode
        self.monitor_gain = monitor_gain
        self.monitor_pair = monitor_pair
        self.print_levels = print_levels
        self.noise_gate = noise_gate
        self.dc_block = dc_block
        self.soft_limiter = soft_limiter
        self.rate = 16_000
        self.channels = 2
        self.play_channels = 2
        self.stream = None
        self.wav = None
        self.bytes_received = 0
        self.started = time.monotonic()
        self.last_pair_logged = None
        self.level_scores = []
        self.level_frames = 0
        self.level_started = time.monotonic()
        self.dc_prev_in = [0.0, 0.0]
        self.dc_prev_out = [0.0, 0.0]

    def configure(self, rate: int, channels: int) -> None:
        self.rate = rate
        self.channels = channels

        if self.play:
            if sd is None:
                print("sounddevice is not installed; mic playback disabled")
            else:
                self.stream = sd.RawOutputStream(
                    samplerate=self.rate,
                    channels=self.play_channels,
                    dtype="int16",
                    blocksize=0,
                )
                self.stream.start()
                if self.channels > self.play_channels:
                    print(
                        f"Playing live mic monitor: {self.monitor_mode}, "
                        f"{self.channels} input channel(s), gain {self.monitor_gain:g}x, "
                        f"{self.rate} Hz"
                    )
                else:
                    print(
                        f"Playing live mic audio: {self.rate} Hz, {self.channels} channel(s), "
                        f"gain {self.monitor_gain:g}x"
                    )

        if self.save_path:
            self.wav = wave.open(str(self.save_path), "wb")
            self.wav.setnchannels(self.channels)
            self.wav.setsampwidth(SAMPLE_WIDTH)
            self.wav.setframerate(self.rate)
            print(f"Saving mic audio to {self.save_path}")

    def write(self, payload: bytes) -> None:
        self.bytes_received += len(payload)
        self._accumulate_levels(payload)
        if self.stream:
            self.stream.write(self._monitor_payload(payload))
        if self.wav:
            self.wav.writeframes(payload)

        now = time.monotonic()
        if self.print_levels and now - self.level_started >= 1.0:
            self._print_levels()
            self.level_started = now

        if now - self.started >= 2.0:
            kbps = (self.bytes_received * 8.0 / 1000.0) / (now - self.started)
            print(f"Mic RX: {kbps:.1f} kbps")
            self.bytes_received = 0
            self.started = now

    def close(self) -> None:
        if self.stream:
            self.stream.stop()
            self.stream.close()
        if self.wav:
            self.wav.close()

    def _monitor_payload(self, payload: bytes) -> bytes:
        samples = memoryview(payload).cast("h")
        stereo_samples: list[tuple[int, int]] = []

        if self.channels == 1:
            for sample in samples:
                stereo_samples.append((sample, sample))
            return self._process_monitor_stereo(stereo_samples)

        if self.channels == self.play_channels:
            for idx in range(0, len(samples), self.channels):
                stereo_samples.append((samples[idx], samples[idx + 1]))
            return self._process_monitor_stereo(stereo_samples)

        if self.monitor_mode in ("first-pair", "pair"):
            pair = 0 if self.monitor_mode == "first-pair" else self.monitor_pair
            pair = max(0, min(pair, (self.channels // 2) - 1))
            left_ch = pair * 2
            right_ch = left_ch + 1
            for idx in range(0, len(samples), self.channels):
                stereo_samples.append((samples[idx + left_ch], samples[idx + right_ch]))
            return self._process_monitor_stereo(stereo_samples)

        if self.monitor_mode == "loudest-pair":
            pair = self._loudest_pair(samples)
            if pair != self.last_pair_logged:
                print(f"Monitoring mic pair {pair}")
                self.last_pair_logged = pair
            left_ch = pair * 2
            right_ch = left_ch + 1
            for idx in range(0, len(samples), self.channels):
                stereo_samples.append((samples[idx + left_ch], samples[idx + right_ch]))
            return self._process_monitor_stereo(stereo_samples)

        left_count = (self.channels + 1) // 2
        right_count = self.channels // 2
        for idx in range(0, len(samples), self.channels):
            left_sum = 0
            right_sum = 0
            for ch in range(self.channels):
                if ch & 1:
                    right_sum += samples[idx + ch]
                else:
                    left_sum += samples[idx + ch]
            stereo_samples.append((int(left_sum / left_count), int(right_sum / right_count)))
        return self._process_monitor_stereo(stereo_samples)

    def _process_monitor_stereo(self, stereo_samples: list[tuple[int, int]]) -> bytes:
        if not stereo_samples:
            return b""

        avg_abs = sum(abs(left) + abs(right) for left, right in stereo_samples) / (len(stereo_samples) * 2)
        gate_threshold = 32768.0 * (self.noise_gate / 100.0)
        gate_gain = 0.0 if self.noise_gate > 0 and avg_abs < gate_threshold else 1.0

        out = bytearray()
        for left, right in stereo_samples:
            left_f = float(left)
            right_f = float(right)

            if self.dc_block:
                left_f = self._dc_block_sample(left_f, 0)
                right_f = self._dc_block_sample(right_f, 1)

            left_f *= self.monitor_gain * gate_gain
            right_f *= self.monitor_gain * gate_gain

            if self.soft_limiter:
                left_f = self._soft_limit(left_f)
                right_f = self._soft_limit(right_f)

            out += struct.pack("<hh", self._clip16(left_f), self._clip16(right_f))
        return bytes(out)

    def _dc_block_sample(self, sample: float, channel: int) -> float:
        # One-pole DC blocker: y[n] = x[n] - x[n-1] + R*y[n-1].
        r = 0.995
        y = sample - self.dc_prev_in[channel] + r * self.dc_prev_out[channel]
        self.dc_prev_in[channel] = sample
        self.dc_prev_out[channel] = y
        return y

    @staticmethod
    def _soft_limit(value: float) -> float:
        # Smoothly compress near full-scale instead of hard clipping.
        normalized = value / 32768.0
        limited = math.tanh(normalized) * 32767.0
        return limited

    def _loudest_pair(self, samples) -> int:
        scores, frame_count = self._pair_scores(samples)
        if frame_count == 0 or not scores:
            return 0
        return max(range(len(scores)), key=lambda pair: scores[pair])

    def _pair_scores(self, samples) -> tuple[list[int], int]:
        pair_count = max(1, self.channels // 2)
        scores = [0] * pair_count
        frame_count = 0

        for idx in range(0, len(samples), self.channels):
            frame_count += 1
            for pair in range(pair_count):
                left = samples[idx + pair * 2]
                right = samples[idx + pair * 2 + 1]
                scores[pair] += abs(left) + abs(right)

        return scores, frame_count

    def _accumulate_levels(self, payload: bytes) -> None:
        if self.channels < 2:
            return

        samples = memoryview(payload).cast("h")
        scores, frame_count = self._pair_scores(samples)
        if not scores or frame_count == 0:
            return

        if len(self.level_scores) != len(scores):
            self.level_scores = [0] * len(scores)

        for idx, score in enumerate(scores):
            self.level_scores[idx] += score
        self.level_frames += frame_count

    def _print_levels(self) -> None:
        if not self.level_scores or self.level_frames == 0:
            return

        levels = []
        for idx, score in enumerate(self.level_scores):
            # Average absolute sample value per channel, normalized to 16-bit full scale.
            avg_abs = score / (self.level_frames * 2)
            percent = 100.0 * avg_abs / 32768.0
            levels.append(f"pair{idx}={percent:5.2f}%")

        loudest = max(range(len(self.level_scores)), key=lambda idx: self.level_scores[idx])
        print(f"Pair levels: {' '.join(levels)} loudest=pair{loudest}")
        self.level_scores = [0] * len(self.level_scores)
        self.level_frames = 0

    @staticmethod
    def _clip16(value: float) -> int:
        return max(-32768, min(32767, int(value)))


async def handle_client(websocket, args) -> None:
    print(f"Board connected from {websocket.remote_address}")
    sink = MicSink(
        play=args.play_mic,
        save_path=args.save_mic,
        monitor_mode=args.monitor_mode,
        monitor_gain=args.monitor_gain,
        monitor_pair=args.monitor_pair,
        print_levels=args.print_levels,
        noise_gate=args.noise_gate,
        dc_block=args.dc_block,
        soft_limiter=not args.no_soft_limiter,
    )
    audio = load_wav_as_s16le_stereo(args.wav) if args.wav else generate_tone()
    sender_task = asyncio.create_task(send_speaker_audio(websocket, audio, args.loop_audio))

    try:
        async for message in websocket:
            if isinstance(message, str):
                print(f"Text from board: {message}")
                try:
                    hello = json.loads(message)
                except json.JSONDecodeError:
                    continue
                if hello.get("type") == "hello":
                    sink.configure(int(hello["mic_rate"]), int(hello["mic_channels"]))
                continue

            if sink.stream is None and sink.wav is None and (args.play_mic or args.save_mic):
                sink.configure(16_000, 2)
            sink.write(message)
    finally:
        sender_task.cancel()
        sink.close()
        print("Board disconnected")


async def main_async(args) -> None:
    stop = asyncio.Event()

    def request_stop(*_):
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    async def client_handler(websocket, path=None):
        del path
        await handle_client(websocket, args)

    async with websockets.serve(client_handler, args.host, args.port, max_size=None):
        print(f"Listening on ws://{args.host}:{args.port}/audio")
        print("Set firmware WebSocket URI to ws://<this-mac-ip>:8765/audio")
        await stop.wait()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--play-mic", action="store_true", help="play live mic audio from the board")
    parser.add_argument("--save-mic", type=Path, help="save mic uplink to a WAV file")
    parser.add_argument(
        "--monitor-mode",
        choices=("first-pair", "pair", "loudest-pair", "mix-all"),
        default="first-pair",
        help="how to turn multi-channel mic input into stereo monitor audio",
    )
    parser.add_argument(
        "--monitor-pair",
        type=int,
        default=0,
        help="mic pair index for --monitor-mode pair; 0=DATA0, 1=DATA1, 2=DATA2, 3=DATA3",
    )
    parser.add_argument(
        "--monitor-gain",
        type=float,
        default=8.0,
        help="software gain applied only to live mic monitoring",
    )
    parser.add_argument(
        "--print-levels",
        action="store_true",
        help="print per-pair mic levels once per second",
    )
    parser.add_argument(
        "--noise-gate",
        type=float,
        default=0.0,
        help="mute monitor audio when average level is below this percent of full scale; try 0.5 to 2.0",
    )
    parser.add_argument(
        "--dc-block",
        action="store_true",
        help="remove DC/low-frequency rumble from monitor audio",
    )
    parser.add_argument(
        "--no-soft-limiter",
        action="store_true",
        help="disable monitor soft limiter",
    )
    parser.add_argument("--wav", type=Path, help="16-bit PCM WAV to stream to the board speaker")
    parser.add_argument("--test-tone", action="store_true", help="send generated tone if --wav is omitted")
    parser.add_argument("--loop-audio", action="store_true", help="repeat the speaker test audio")
    args = parser.parse_args(argv)
    if args.wav and not args.wav.exists():
        parser.error(f"--wav file not found: {args.wav}")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    asyncio.run(main_async(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
