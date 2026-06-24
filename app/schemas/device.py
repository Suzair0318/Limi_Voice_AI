"""Pydantic models for voice pendant device registration."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


DeviceType = Literal["voice_pendant"]


class DeviceRegisterRequest(BaseModel):
    """Payload for registering a voice pendant."""

    device_id: str = Field(..., min_length=1, max_length=128, examples=["pendant-001"])
    name: str = Field(..., min_length=1, max_length=256, examples=["Living Room Pendant"])
    device_type: DeviceType = Field("voice_pendant", examples=["voice_pendant"])

    @field_validator("device_id", "name")
    @classmethod
    def strip_and_validate(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned

    @field_validator("device_id")
    @classmethod
    def validate_device_id(cls, value: str) -> str:
        if any(ch.isspace() for ch in value):
            raise ValueError("device_id must not contain spaces")
        return value


class DeviceResponse(BaseModel):
    """Stored device record returned by the API."""

    device_id: str
    name: str
    device_type: DeviceType
    created_at: datetime
    updated_at: datetime


class DeviceListResponse(BaseModel):
    """Paginated-style list wrapper for registered devices."""

    total: int
    devices: list[DeviceResponse]
