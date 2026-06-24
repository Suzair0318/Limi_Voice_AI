"""MongoDB connection management via Motor (async driver)."""

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import settings

_client: AsyncIOMotorClient | None = None


async def connect_db() -> None:
    """Open the MongoDB client and verify connectivity with a ping."""
    global _client
    _client = AsyncIOMotorClient(settings.mongo_url)
    await _client.admin.command("ping")
    db = _client.get_default_database()
    await db["devices"].create_index("device_id", unique=True)
    print(f"[MONGO] Connected to {settings.mongo_url} (database={db.name}).")


async def close_db() -> None:
    """Close the MongoDB client."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
        print("[MONGO] Connection closed.")


def get_client() -> AsyncIOMotorClient:
    """Return the shared Motor client (raises if not connected)."""
    if _client is None:
        raise RuntimeError(
            "MongoDB is not connected. The app lifespan must call connect_db() first."
        )
    return _client


def get_db() -> AsyncIOMotorDatabase:
    """Return the default database from ``MONGO_URL`` (e.g. ``limi_project_dev``)."""
    return get_client().get_default_database()
