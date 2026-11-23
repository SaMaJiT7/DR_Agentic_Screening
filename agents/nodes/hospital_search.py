"""Hospital search node: OpenStreetMap Nominatim lookup for eye hospitals.

This is the tool half of the care-navigator role. It only *finds* places; it
never invents names, phone numbers or ratings, and it carries no prices.

Network failures are non-fatal: the node records an error, returns an empty
list, and the report still renders (with a "search locally" note).
"""

from __future__ import annotations

from typing import Callable

from agents.config import settings
from agents.state import DR_State, Hospital


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_results(payload) -> list[Hospital]:
    """Convert a Nominatim JSON payload into ``Hospital`` records."""

    if not isinstance(payload, list):
        return []

    hospitals: list[Hospital] = []
    seen: set[str] = set()
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        name = (entry.get("name") or "").strip()
        display_name = (entry.get("display_name") or "").strip()
        if not name and display_name:
            name = display_name.split(",")[0].strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        hospitals.append(
            Hospital(
                name=name,
                address=display_name or None,
                lat=_as_float(entry.get("lat")),
                lon=_as_float(entry.get("lon")),
            )
        )
    return hospitals


def search_hospitals(
    city: str,
    *,
    limit: int | None = None,
    http_get: Callable | None = None,
) -> list[Hospital]:
    """Return eye hospitals near ``city`` using Nominatim.

    Args:
        city: free-text city name supplied by the patient.
        limit: maximum number of results (defaults to ``HOSPITAL_LIMIT``).
        http_get: injected ``requests.get``-style callable, for tests.
    """

    if not city or not city.strip():
        raise ValueError("hospital search requires a city")

    if http_get is None:
        import requests

        http_get = requests.get

    params = {
        "q": f"eye hospital, {city.strip()}",
        "format": "jsonv2",
        "limit": str(limit if limit is not None else settings.hospital_limit),
        "addressdetails": "1",
    }
    response = http_get(
        settings.nominatim_url,
        params=params,
        headers={"User-Agent": settings.nominatim_user_agent},
        timeout=settings.http_timeout,
    )
    response.raise_for_status()
    return parse_results(response.json())


def hospital_search_node(state: DR_State) -> dict:
    """LangGraph node: fill ``hospitals``, keeping the run alive on failure."""

    try:
        hospitals = search_hospitals(state.city)
    except Exception as exc:  # network, JSON or timeout problems
        return {
            "hospitals": [],
            "errors": state.errors + [f"hospital search failed: {exc}"],
        }

    update: dict = {"hospitals": hospitals}
    if not hospitals:
        update["errors"] = state.errors + [
            f"hospital search returned no results for {state.city!r}"
        ]
    return update
