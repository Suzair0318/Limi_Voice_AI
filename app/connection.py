"""WebSocket connection management.

The :class:`ConnectionManager` keeps track of every connected hardware device,
provides safe broadcast/send helpers, and guarantees that a misbehaving or
abruptly-disconnected client can never take down the event loop or other
clients. All send operations are wrapped so that a dead socket is silently
pruned instead of raising into business logic.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional

from fastapi import WebSocket
from starlette.websockets import WebSocketState


@dataclass
class DeviceSession:
    """Per-device runtime state held by the :class:`ConnectionManager`."""

    client_id: str
    websocket: WebSocket
    # Serialises outbound writes so concurrent tasks (e.g. streaming audio back
    # while a keep-alive fires) never interleave frames on the same socket.
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Set while we are streaming a response so inbound audio can be ignored
    # (simple half-duplex turn-taking / barge-in guard).
    is_responding: bool = False


class ConnectionManager:
    """Tracks active WebSocket sessions in a thread-safe (async) manner."""

    def __init__(self) -> None:
        self._sessions: Dict[str, DeviceSession] = {}
        self._lock = asyncio.Lock()

    async def connect(self, client_id: str, websocket: WebSocket) -> DeviceSession:
        """Accept a new WebSocket and register its session.

        If a session with the same ``client_id`` already exists (e.g. a board
        reconnecting after a network blip) the stale one is dropped first.
        """
        await websocket.accept()
        async with self._lock:
            existing = self._sessions.get(client_id)
            if existing is not None:
                await self._safe_close(existing.websocket)
            session = DeviceSession(client_id=client_id, websocket=websocket)
            self._sessions[client_id] = session
        print(f"[CONNECT] Device '{client_id}' connected. "
              f"Active devices: {self.active_count}")
        return session

    async def disconnect(self, client_id: str) -> None:
        """Remove a session and close its socket if still open."""
        async with self._lock:
            session = self._sessions.pop(client_id, None)
        if session is not None:
            await self._safe_close(session.websocket)
            print(f"[DISCONNECT] Device '{client_id}' disconnected. "
                  f"Active devices: {self.active_count}")

    def get(self, client_id: str) -> Optional[DeviceSession]:
        """Return the session for ``client_id`` if present."""
        return self._sessions.get(client_id)

    @property
    def active_count(self) -> int:
        """Number of currently registered sessions."""
        return len(self._sessions)

    async def send_bytes(self, session: DeviceSession, data: bytes) -> bool:
        """Send a binary frame to a single device.

        Returns ``True`` on success and ``False`` if the socket was dead (in
        which case the session is pruned). Never raises.
        """
        if session.websocket.application_state != WebSocketState.CONNECTED:
            await self.disconnect(session.client_id)
            return False
        try:
            async with session.send_lock:
                await session.websocket.send_bytes(data)
            return True
        except Exception as exc:  # noqa: BLE001 - we never want a bad socket to bubble up
            print(f"[ERROR] Failed to send bytes to '{session.client_id}': {exc!r}")
            await self.disconnect(session.client_id)
            return False

    async def send_json(self, session: DeviceSession, payload: dict) -> bool:
        """Send a JSON control message to a single device. Never raises."""
        if session.websocket.application_state != WebSocketState.CONNECTED:
            await self.disconnect(session.client_id)
            return False
        try:
            async with session.send_lock:
                await session.websocket.send_json(payload)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Failed to send json to '{session.client_id}': {exc!r}")
            await self.disconnect(session.client_id)
            return False

    async def broadcast_json(self, payload: dict) -> None:
        """Send a JSON message to every connected device, pruning dead ones."""
        for session in list(self._sessions.values()):
            await self.send_json(session, payload)

    async def shutdown(self) -> None:
        """Close every active socket. Used during application shutdown."""
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            await self._safe_close(session.websocket)
        print("[SHUTDOWN] All WebSocket sessions closed.")

    @staticmethod
    async def _safe_close(websocket: WebSocket) -> None:
        """Best-effort socket close that swallows all errors."""
        try:
            if websocket.application_state == WebSocketState.CONNECTED:
                await websocket.close()
        except Exception:  # noqa: BLE001 - closing a broken socket is fine to ignore
            pass


# Module-level singleton shared across the app.
manager = ConnectionManager()
