"""NOAA NWS forecast ingestion. Pulls 7-day daily forecast at a lat/lon
from the public api.weather.gov service. No auth, no rate limit beyond the
NWS politeness requirement (User-Agent must identify the caller).

The forecast is a *watershed-level* signal: we sample one point per HUC8
(the centroid of the loaded reach extent) and apply the same air-temp +
precipitation forecast to every reach in that HUC8 for the scoring layer.
This is a simplification - real reaches in a HUC8 see different microclimates
- but it is HONEST: we attach the same NOAA forecast row to every reach in
the watershed rather than fabricating per-reach forecasts.

NWS daily forecast covers ~7 days. We extend to 14 by stitching with a
persistence projection of the last day's values (clearly tagged
'persistence_projection' so the scoring layer can attenuate confidence
beyond day 7).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import httpx

log = logging.getLogger(__name__)

NWS_ROOT = "https://api.weather.gov"
USER_AGENT = "angler-ai/0.1 (ron.dilley@gmail.com)"


@dataclass(frozen=True, slots=True)
class DailyForecast:
    """One forecast day. All fields can be None if NWS did not return them."""

    forecast_date: date
    high_c: float | None
    low_c: float | None
    precip_probability_pct: int | None
    short_forecast: str | None
    source: str
    """'NWS_NDFD_daily' for the first ~7 days, 'persistence_projection'
    when projected beyond NWS's horizon."""


def fetch_daily_forecast(
    lat: float, lon: float, *, days: int = 14, client: httpx.Client | None = None,
) -> list[DailyForecast]:
    """Return up to `days` daily forecast entries for the lat/lon.

    NWS provides ~7 days of daily forecast. Days 8..N are filled by
    persistence-projecting the last NWS day (high/low/precip carried forward,
    tagged 'persistence_projection').

    Args:
        lat, lon: forecast point (decimal degrees, WGS84).
        days: total days to return.
        client: optional httpx.Client; one is created if not provided.

    Returns:
        List of DailyForecast in chronological order. Empty list on failure.
    """
    own_client = client is None
    cl = client or httpx.Client(
        timeout=60,
        headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
    )
    try:
        # Step 1: points endpoint returns the grid forecast URL for this point.
        r = cl.get(f"{NWS_ROOT}/points/{lat:.4f},{lon:.4f}")
        r.raise_for_status()
        properties = r.json().get("properties") or {}
        forecast_url = properties.get("forecast")
        if not forecast_url:
            log.warning("NWS points: no forecast URL for %.4f,%.4f", lat, lon)
            return []

        # Step 2: forecast URL returns 14 periods (day + night) over ~7 days.
        r = cl.get(forecast_url)
        r.raise_for_status()
        periods = (r.json().get("properties") or {}).get("periods") or []
    except httpx.HTTPError as exc:
        log.warning("NWS forecast fetch failed for %.4f,%.4f: %s", lat, lon, exc)
        return []
    finally:
        if own_client:
            cl.close()

    # Periods alternate day/night with a startTime. Group by date.
    by_date: dict[date, dict] = {}
    for p in periods:
        try:
            start = datetime.fromisoformat(p["startTime"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        d = start.date()
        entry = by_date.setdefault(d, {})
        temp_f = p.get("temperature")
        temp_c = (temp_f - 32) * 5.0 / 9.0 if isinstance(temp_f, (int, float)) else None
        if p.get("isDaytime"):
            if temp_c is not None:
                entry["high_c"] = temp_c
            entry["short_forecast"] = p.get("shortForecast")
            prob = (p.get("probabilityOfPrecipitation") or {}).get("value")
            if isinstance(prob, (int, float)):
                entry["precip_probability_pct"] = int(prob)
        else:
            if temp_c is not None:
                entry["low_c"] = temp_c
            if "precip_probability_pct" not in entry:
                prob = (p.get("probabilityOfPrecipitation") or {}).get("value")
                if isinstance(prob, (int, float)):
                    entry["precip_probability_pct"] = int(prob)

    nws_days = [
        DailyForecast(
            forecast_date=d,
            high_c=by_date[d].get("high_c"),
            low_c=by_date[d].get("low_c"),
            precip_probability_pct=by_date[d].get("precip_probability_pct"),
            short_forecast=by_date[d].get("short_forecast"),
            source="NWS_NDFD_daily",
        )
        for d in sorted(by_date.keys())
    ]

    if not nws_days:
        return []

    if len(nws_days) >= days:
        return nws_days[:days]

    # Persistence projection beyond NWS horizon: repeat last day's values.
    last = nws_days[-1]
    projected: list[DailyForecast] = []
    next_date = last.forecast_date + timedelta(days=1)
    while len(nws_days) + len(projected) < days:
        projected.append(DailyForecast(
            forecast_date=next_date,
            high_c=last.high_c,
            low_c=last.low_c,
            precip_probability_pct=last.precip_probability_pct,
            short_forecast=last.short_forecast,
            source="persistence_projection",
        ))
        next_date = next_date + timedelta(days=1)
    return nws_days + projected
