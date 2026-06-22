"""Dummy tools exposed to the LangGraph agent.

These are intentionally simple, deterministic stand-ins so the agent has
something to call. Replace the bodies with real integrations (weather APIs,
home-automation, databases, etc.) as needed. Each tool is strictly typed and
decorated with :func:`langchain_core.tools.tool` so LangChain can build the
JSON schema automatically from the signature + docstring.
"""

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
    # Dummy deterministic response. Swap for a real weather API call.
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
    # Dummy: in production persist to a DB / task queue.
    return f"Reminder saved: '{text}'."


# Single source of truth for the agent's toolset.
TOOLS: List = [get_weather, get_current_time, add_reminder]
