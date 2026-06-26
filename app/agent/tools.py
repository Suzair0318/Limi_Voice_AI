"""Tools exposed to the LangGraph agent and Realtime function calling."""

from __future__ import annotations

import datetime as _dt
from typing import List

from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city.

    Args:
        city: Name of the city to look up, e.g. "London" or "Tokyo".

    Returns:
        A short human-readable weather description.
    """
    sample = {
        "london": "12C, light rain and overcast.",
        "tokyo": "21C, clear skies.",
        "new york": "17C, partly cloudy with a light breeze.",
    }
    key = city.strip().lower()
    report = sample.get(key, "22C and sunny.")
    return f"The weather in {city.title()} is {report}"


@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date and time.

    Args:
        timezone: IANA timezone name. Only "UTC" is implemented in this dummy
            tool; any other value still returns UTC for simplicity.

    Returns:
        An ISO-8601 formatted timestamp string.
    """
    now = _dt.datetime.now(_dt.timezone.utc)
    return f"The current time ({timezone}) is {now.isoformat(timespec='seconds')}."


@tool
def add_reminder(text: str) -> str:
    """Save a short reminder for the user.

    Args:
        text: The reminder content, e.g. "call mom at 6pm".

    Returns:
        A confirmation string.
    """
    return f"Reminder saved: '{text}'."


TOOLS: List = [get_weather, get_current_time, add_reminder]

TOOL_BY_NAME = {t.name: t for t in TOOLS}
