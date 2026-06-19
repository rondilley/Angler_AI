"""Open-Meteo daily forecast ingestion. Used to extend the NWS 7-day
horizon out to 16 days with a real second forecast source instead of
persistence-projecting.

Open-Meteo is free, no API key required, attribution requested. The
endpoint returns daily max/min temperature, precipitation, and
precipitation probability for up to 16 days.

Source tag on output: 'OpenMeteo_daily' for the full window. The forecast
scoring layer treats NWS as authoritative for days 0-7 and Open-Meteo for
days 8-16 (the two sources are independently produced; NWS is more
locally calibrated for the US within its horizon).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime

import httpx

log = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


@dataclass(frozen=True, slots=True)
class OpenMeteoDaily:
    """One Open-Meteo daily forecast row."""

    forecast_date: date
    high_c: float | None
    low_c: float | None
    precip_probability_pct: int | None
    precip_mm: float | None
    source: str = "OpenMeteo_daily"


def fetch_daily_forecast(
    lat: float, lon: float, *, days: int = 16, client: httpx.Client | None = None,
) -> list[OpenMeteoDaily]:
    """Return up to `days` daily forecast entries for the lat/lon.

    Args:
        lat, lon: forecast point (decimal degrees, WGS84).
        days: total days to return (max 16).

    Returns:
        List of OpenMeteoDaily in chronological order. Empty list on failure.
    """
    own_client = client is None
    cl = client or httpx.Client(timeout=30, headers={
        "User-Agent": "angler-ai/0.1 (ron.dilley@gmail.com)",
    })
    try:
        params = {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "daily": "temperature_2m_max,temperature_2m_min,"
                     "precipitation_probability_mean,precipitation_sum",
            "timezone": "UTC",
            "forecast_days": str(min(days, 16)),
        }
        r = cl.get(OPEN_METEO_URL, params=params)
        r.raise_for_status()
        d = r.json().get("daily") or {}
    except httpx.HTTPError as exc:
        log.warning("Open-Meteo fetch failed for %.4f,%.4f: %s", lat, lon, exc)
        return []
    finally:
        if own_client:
            cl.close()

    dates = d.get("time") or []
    highs = d.get("temperature_2m_max") or []
    lows = d.get("temperature_2m_min") or []
    probs = d.get("precipitation_probability_mean") or []
    sums = d.get("precipitation_sum") or []
    out: list[OpenMeteoDaily] = []
    n = min(len(dates), days)
    for i in range(n):
        try:
            d_iso = date.fromisoformat(dates[i])
        except (ValueError, TypeError):
            continue
        out.append(OpenMeteoDaily(
            forecast_date=d_iso,
            high_c=float(highs[i]) if i < len(highs) and highs[i] is not None else None,
            low_c=float(lows[i]) if i < len(lows) and lows[i] is not None else None,
            precip_probability_pct=int(probs[i]) if i < len(probs) and probs[i] is not None else None,
            precip_mm=float(sums[i]) if i < len(sums) and sums[i] is not None else None,
        ))
    return out
