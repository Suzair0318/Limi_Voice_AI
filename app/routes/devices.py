"""REST API for voice pendant device registration."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status
from pymongo.errors import DuplicateKeyError

from app.db import get_db
from app.schemas.device import DeviceListResponse, DeviceRegisterRequest, DeviceResponse

router = APIRouter(prefix="/api/devices", tags=["devices"])

DEVICES_COLLECTION = "devices"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_response(doc: dict) -> DeviceResponse:
    return DeviceResponse(
        device_id=doc["device_id"],
        name=doc["name"],
        device_type=doc["device_type"],
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
    )


async def _insert_device(payload: DeviceRegisterRequest) -> DeviceResponse:
    now = _utc_now()
    doc = {
        "device_id": payload.device_id,
        "name": payload.name,
        "device_type": payload.device_type,
        "created_at": now,
        "updated_at": now,
    }

    collection = get_db()[DEVICES_COLLECTION]
    try:
        await collection.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Device '{payload.device_id}' is already registered.",
        ) from None

    print(f"[DEVICE] Registered voice pendant '{payload.device_id}' ({payload.name}).")
    return _to_response(doc)


@router.get("/register")
async def register_device_get_help() -> None:
    """Registration requires POST, not GET."""
    raise HTTPException(
        status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
        detail=(
            "Use POST /api/devices/register with JSON body "
            '{"device_id": "pendant-001", "name": "My Pendant"}. '
            "To list devices use GET /api/devices. "
            "To fetch one device use GET /api/devices/{device_id}."
        ),
    )


@router.post(
    "/register",
    response_model=DeviceResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_device(payload: DeviceRegisterRequest) -> DeviceResponse:
    """Register a voice pendant by ``device_id`` and ``name``."""
    return await _insert_device(payload)


@router.post(
    "",
    response_model=DeviceResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_device_root(payload: DeviceRegisterRequest) -> DeviceResponse:
    """Alias for POST /api/devices/register."""
    return await _insert_device(payload)


@router.get("", response_model=DeviceListResponse)
async def list_devices() -> DeviceListResponse:
    """List all registered voice pendants."""
    collection = get_db()[DEVICES_COLLECTION]
    cursor = collection.find().sort("created_at", -1)
    devices = [_to_response(doc) async for doc in cursor]
    return DeviceListResponse(total=len(devices), devices=devices)


@router.get("/{device_id}", response_model=DeviceResponse)
async def get_device(device_id: str) -> DeviceResponse:
    """Return a registered device by ``device_id``."""
    doc = await get_db()[DEVICES_COLLECTION].find_one({"device_id": device_id})
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Device '{device_id}' is not registered.",
        )
    return _to_response(doc)
