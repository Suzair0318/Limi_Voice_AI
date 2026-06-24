from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import wave


class MicSessionRecorder:
    def __init__(
        self,
        device_id: str,
        out_dir: Path,
        device_rate: int,
        ai_rate: int,
    ) -> None:
        self.device_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in device_id)
        self.out_dir = out_dir
        self.device_rate = device_rate
        self.ai_rate = ai_rate
        self.turn = 0
        self.raw_wav: wave.Wave_write | None = None
        self.ai_wav: wave.Wave_write | None = None
        self.raw_path: Path | None = None
        self.ai_path: Path | None = None
        self.raw_bytes = 0
        self.ai_bytes = 0

    def active(self) -> bool:
        return self.raw_wav is not None or self.ai_wav is not None

    def start(self, reason: str = "audio") -> None:
        if self.active():
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.turn += 1
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        base = f"{self.device_id}_{stamp}_turn{self.turn:04d}"
        self.raw_path = self.out_dir / f"{base}_backend_heard_{self.device_rate}hz.wav"
        self.ai_path = self.out_dir / f"{base}_sent_to_ai_{self.ai_rate}hz.wav"
        self.raw_wav = self._open_wav(self.raw_path, self.device_rate)
        self.ai_wav = self._open_wav(self.ai_path, self.ai_rate)
        self.raw_bytes = 0
        self.ai_bytes = 0
        print(f"[REC] {self.device_id}: started backend recording ({reason})")

    def write(self, backend_heard_pcm: bytes, ai_pcm: bytes) -> None:
        if not self.active():
            self.start()
        if self.raw_wav and backend_heard_pcm:
            self.raw_wav.writeframes(backend_heard_pcm)
            self.raw_bytes += len(backend_heard_pcm)
        if self.ai_wav and ai_pcm:
            self.ai_wav.writeframes(ai_pcm)
            self.ai_bytes += len(ai_pcm)

    def stop(self, reason: str = "end") -> None:
        if not self.active():
            return
        raw_seconds = self.raw_bytes / max(1, self.device_rate * 2)
        ai_seconds = self.ai_bytes / max(1, self.ai_rate * 2)
        if self.raw_wav:
            self.raw_wav.close()
        if self.ai_wav:
            self.ai_wav.close()
        self.raw_wav = None
        self.ai_wav = None
        print(
            f"[REC] {self.device_id}: saved backend-heard={self.raw_path} "
            f"({raw_seconds:.2f}s), sent-to-ai={self.ai_path} ({ai_seconds:.2f}s), "
            f"reason={reason}"
        )

    @staticmethod
    def _open_wav(path: Path, rate: int) -> wave.Wave_write:
        wav = wave.open(str(path), "wb")
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        return wav
