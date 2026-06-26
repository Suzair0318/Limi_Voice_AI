"""Tkinter desktop UI for the Limi voice backend test client."""

from __future__ import annotations

import queue
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Any

from desktop.config import ClientConfig, default_config
from desktop.voice_session import SessionState, UiEvent, VoiceSession


class LimiDesktopApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Limi Voice — Desktop Client")
        self.root.minsize(720, 560)
        self.root.geometry("860x640")

        self.config = ClientConfig(
            server_host=default_config.server_host,
            server_port=default_config.server_port,
            device_id=default_config.device_id,
            device_input_rate=default_config.device_input_rate,
            device_input_channels=default_config.device_input_channels,
            device_output_rate=default_config.device_output_rate,
            device_output_channels=default_config.device_output_channels,
            device_output_chunk_ms=default_config.device_output_chunk_ms,
            mic_chunk_ms=default_config.mic_chunk_ms,
        )
        self.session: VoiceSession | None = None
        self._ui_queue: queue.Queue[UiEvent] = queue.Queue()

        self._build_ui()
        self._poll_ui_queue()

    def _build_ui(self) -> None:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")

        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(
            outer,
            text="Limi Voice Desktop Client",
            font=("Segoe UI", 16, "bold"),
        )
        header.pack(anchor=tk.W)
        sub = ttk.Label(
            outer,
            text=self.config.summary(),
            foreground="#555",
        )
        sub.pack(anchor=tk.W, pady=(2, 12))
        self.protocol_label = sub

        conn = ttk.LabelFrame(outer, text="Connection", padding=10)
        conn.pack(fill=tk.X, pady=(0, 10))

        row1 = ttk.Frame(conn)
        row1.pack(fill=tk.X)
        ttk.Label(row1, text="Host").grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        self.host_var = tk.StringVar(value=self.config.server_host)
        ttk.Entry(row1, textvariable=self.host_var, width=18).grid(row=0, column=1, padx=(0, 12))
        ttk.Label(row1, text="Port").grid(row=0, column=2, sticky=tk.W, padx=(0, 6))
        self.port_var = tk.StringVar(value=str(self.config.server_port))
        ttk.Entry(row1, textvariable=self.port_var, width=8).grid(row=0, column=3, padx=(0, 12))
        ttk.Label(row1, text="Device ID").grid(row=0, column=4, sticky=tk.W, padx=(0, 6))
        self.device_var = tk.StringVar(value=self.config.device_id)
        ttk.Entry(row1, textvariable=self.device_var, width=20).grid(row=0, column=5)

        row2 = ttk.Frame(conn)
        row2.pack(fill=tk.X, pady=(10, 0))
        self.connect_btn = ttk.Button(row2, text="Connect", command=self._on_connect)
        self.connect_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.disconnect_btn = ttk.Button(row2, text="Disconnect", command=self._on_disconnect, state=tk.DISABLED)
        self.disconnect_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.health_btn = ttk.Button(row2, text="Check Health", command=self._on_health)
        self.health_btn.pack(side=tk.LEFT)

        self.ws_label = ttk.Label(row2, text="WebSocket: disconnected")
        self.ws_label.pack(side=tk.RIGHT)

        talk = ttk.LabelFrame(outer, text="Voice", padding=10)
        talk.pack(fill=tk.X, pady=(0, 10))

        talk_row = ttk.Frame(talk)
        talk_row.pack(fill=tk.X)
        self.talk_btn = ttk.Button(
            talk_row,
            text="Start Talking",
            command=self._on_talk_toggle,
            state=tk.DISABLED,
        )
        self.talk_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.wake_btn = ttk.Button(
            talk_row,
            text="Wake Detected",
            command=self._on_wake,
            state=tk.DISABLED,
        )
        self.wake_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.wake_end_btn = ttk.Button(
            talk_row,
            text="Wake Session End",
            command=self._on_wake_end,
            state=tk.DISABLED,
        )
        self.wake_end_btn.pack(side=tk.LEFT)

        self.state_label = ttk.Label(talk_row, text="State: disconnected")
        self.state_label.pack(side=tk.RIGHT)

        meters = ttk.Frame(talk)
        meters.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(meters, text="Mic").grid(row=0, column=0, sticky=tk.W)
        self.mic_meter = ttk.Progressbar(meters, length=320, mode="determinate", maximum=100)
        self.mic_meter.grid(row=0, column=1, sticky=tk.EW, padx=8)
        ttk.Label(meters, text="Speaker").grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        self.speaker_meter = ttk.Progressbar(meters, length=320, mode="determinate", maximum=100)
        self.speaker_meter.grid(row=1, column=1, sticky=tk.EW, padx=8, pady=(8, 0))
        meters.columnconfigure(1, weight=1)

        stats = ttk.LabelFrame(outer, text="Session Stats", padding=10)
        stats.pack(fill=tk.X, pady=(0, 10))
        self.stats_label = ttk.Label(
            stats,
            text="Mic sent: 0 B · Speaker rx: 0 B · Mic frames: 0 · Speaker chunks: 0",
        )
        self.stats_label.pack(anchor=tk.W)
        self.backend_label = ttk.Label(stats, text="Backend: —", foreground="#555")
        self.backend_label.pack(anchor=tk.W, pady=(6, 0))

        log_frame = ttk.LabelFrame(outer, text="Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_box = scrolledtext.ScrolledText(log_frame, height=14, state=tk.DISABLED, wrap=tk.WORD)
        self.log_box.pack(fill=tk.BOTH, expand=True)
        ttk.Button(log_frame, text="Clear Log", command=self._clear_log).pack(anchor=tk.E, pady=(8, 0))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _apply_config_from_form(self) -> bool:
        try:
            port = int(self.port_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid port", "Port must be a number.")
            return False
        host = self.host_var.get().strip()
        device_id = self.device_var.get().strip()
        if not host or not device_id:
            messagebox.showerror("Missing fields", "Host and Device ID are required.")
            return False
        self.config.server_host = host
        self.config.server_port = port
        self.config.device_id = device_id
        self.protocol_label.config(text=self.config.summary())
        return True

    def _on_connect(self) -> None:
        if not self._apply_config_from_form():
            return
        if self.session is not None:
            return
        self.session = VoiceSession(self.config, self._enqueue_event)
        self.session.start()
        self.connect_btn.config(state=tk.DISABLED)
        self.disconnect_btn.config(state=tk.NORMAL)
        self._log(f"Connecting to {self.config.ws_url}")

    def _on_disconnect(self) -> None:
        if self.session is None:
            return
        self.session.stop()
        self.session = None
        self._set_connected_ui(False)
        self._set_state_label(SessionState.DISCONNECTED)
        self._log("Disconnected")

    def _on_talk_toggle(self) -> None:
        if self.session is None:
            return
        if self.session.state == SessionState.LISTENING:
            self.session.stop_mic()
            self.talk_btn.config(text="Start Talking")
            self._log("Mic stopped")
        else:
            self.session.start_mic()
            self.talk_btn.config(text="Stop Talking")
            self._log("Mic streaming — speak naturally, pause when done")

    def _on_wake(self) -> None:
        if self.session:
            self.session.send_wake_detected()

    def _on_wake_end(self) -> None:
        if self.session:
            self.session.send_wake_session_end()

    def _on_health(self) -> None:
        if not self._apply_config_from_form():
            return
        temp = VoiceSession(self.config, lambda _e: None)
        health = temp.fetch_health()
        if health is None:
            messagebox.showwarning("Health", "Could not reach the backend.")
            return
        msg = (
            f"status={health.get('status')}\n"
            f"active_devices={health.get('active_devices')}\n"
            f"mongo={health.get('mongo')}\n"
            f"chunk_bytes={health.get('device_output_chunk_bytes')}"
        )
        messagebox.showinfo("Health", msg)
        self._log(f"Health: {health}")

    def _on_close(self) -> None:
        if self.session is not None:
            self.session.stop()
        self.root.destroy()

    def _enqueue_event(self, event: UiEvent) -> None:
        self._ui_queue.put(event)

    def _poll_ui_queue(self) -> None:
        while True:
            try:
                event = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_event(event)
        self.root.after(80, self._poll_ui_queue)

    def _handle_event(self, event: UiEvent) -> None:
        if event.kind == "log":
            level = event.data.get("level", "info")
            self._log(event.message, level)
        elif event.kind == "state":
            state = SessionState(event.data.get("state", SessionState.DISCONNECTED.value))
            self._set_state_label(state)
            if state == SessionState.CONNECTING:
                self.ws_label.config(text="WebSocket: connecting…")
            elif state == SessionState.DISCONNECTED:
                self._set_connected_ui(False)
            elif state in (
                SessionState.CONNECTED,
                SessionState.LISTENING,
                SessionState.SPEAKING,
            ):
                self._set_connected_ui(True)
            if state == SessionState.LISTENING:
                self.talk_btn.config(text="Stop Talking")
            elif state in (SessionState.CONNECTED, SessionState.SPEAKING):
                if self.session and not self.session.mic_active:
                    self.talk_btn.config(text="Start Talking")
        elif event.kind == "stats":
            self._update_stats(event.data)
        elif event.kind == "mic_level":
            self.mic_meter["value"] = min(100.0, event.data.get("peak", 0) * 100)
        elif event.kind == "speaker_level":
            self.speaker_meter["value"] = min(100.0, event.data.get("peak", 0) * 100)
        elif event.kind == "backend_ready":
            self.backend_label.config(
                text=(
                    f"Backend ready — in={event.data.get('device_input_rate')} Hz "
                    f"out={event.data.get('device_output_rate')} Hz "
                    f"chunk={event.data.get('device_output_chunk_ms')} ms"
                )
            )

    def _set_connected_ui(self, connected: bool) -> None:
        state = tk.NORMAL if connected else tk.DISABLED
        self.connect_btn.config(state=tk.DISABLED if connected else tk.NORMAL)
        self.disconnect_btn.config(state=state)
        self.talk_btn.config(state=state)
        self.wake_btn.config(state=state)
        self.wake_end_btn.config(state=state)
        self.ws_label.config(text=f"WebSocket: {'connected' if connected else 'disconnected'}")

    def _set_state_label(self, state: SessionState) -> None:
        self.state_label.config(text=f"State: {state.value}")

    def _update_stats(self, data: dict[str, Any]) -> None:
        self.stats_label.config(
            text=(
                f"Mic sent: {data.get('mic_bytes', 0)} B · "
                f"Speaker rx: {data.get('speaker_bytes', 0)} B · "
                f"Mic frames: {data.get('mic_frames', 0)} · "
                f"Speaker chunks: {data.get('speaker_chunks', 0)}"
            )
        )

    def _log(self, message: str, level: str = "info") -> None:
        prefix = {"info": "", "warn": "[WARN] ", "error": "[ERR] "}.get(level, "")
        self.log_box.config(state=tk.NORMAL)
        self.log_box.insert(tk.END, f"{prefix}{message}\n")
        self.log_box.see(tk.END)
        self.log_box.config(state=tk.DISABLED)

    def _clear_log(self) -> None:
        self.log_box.config(state=tk.NORMAL)
        self.log_box.delete("1.0", tk.END)
        self.log_box.config(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    LimiDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
