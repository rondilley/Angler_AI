"""USGS NWIS water-temperature ingest. Parameter code 00010, continuous
sensor measurements. Persisted into reach_temperature with the
'NWIS_obs' source tag (observation at the gauge's reach, distinct from
'NWIS_interp' which would be spatial interpolation between gauges).

Why a separate module from usgs_nwis.py (discharge): the OGC continuous
endpoint is keyed per parameter, and the gauge subset that records water
temp is much smaller than the discharge set. Keeping the two paths
separate also keeps the reach_temperature provenance clean.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone

import httpx

from angler_ai.features.store import FeatureStore
from angler_ai.ingest._bulk import bulk_insert

log = logging.getLogger(__name__)

OGC_ROOT = "https://api.waterdata.usgs.gov/ogcapi/v0"
WATER_TEMP_PCODE = "00010"  # degrees Celsius at the sensor


def _get_with_backoff(
    client: httpx.Client, url: str, params: dict, *,
    max_retries: int = 3, base_delay: float = 1.5, max_delay: float = 30.0,
) -> httpx.Response:
    """GET with exponential backoff on 429. Returns the final Response.

    Honors Retry-After header but caps at max_delay so we don't block the
    pipeline for many minutes on a single transient block.
    """
    for attempt in range(max_retries):
        r = client.get(url, params=params)
        if r.status_code != 429:
            return r
        retry_after = r.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            delay = min(float(retry_after), max_delay)
        else:
            delay = min(base_delay * (2 ** attempt), max_delay)
        log.info("NWIS 429: backing off %.1fs (attempt %d/%d)", delay, attempt + 1, max_retries)
        time.sleep(delay)
    return r


def ingest_water_temp_for_huc8(
    store: FeatureStore,
    *,
    huc8: str,
    lookback_days: int = 14,
    max_gauges: int = 15,
) -> int:
    """Fetch observed water temperature at every NWIS stream gauge in `huc8`
    and persist into reach_temperature. Returns rows written.

    Strategy:
        1. Filter monitoring-locations by hydrologic_unit_code (HUC8 prefix).
        2. For each gauge that reports 00010, pull continuous measurements
           over the lookback window, daily-averaged.
        3. Resolve gauge -> COMID by joining the gauge's NHDPlus reachcode
           against our reaches table (use the gauge's NHDPlus reachcode if
           returned, else nearest reach by lat/lon proximity within HUC8).
        4. Insert one reach_temperature row per (comid, date) with source
           'NWIS_obs'. ON CONFLICT keep the original.
    """
    conn = store.connect()
    # We need a set of (gauge, comid) for the temperature data to land on.
    # NWIS monitoring-locations have a `hydrologic_unit_code` field giving the
    # HUC for that gauge - we can filter directly.
    headers = {
        "User-Agent": "angler-ai/0.1 (ron.dilley@gmail.com)",
        "Accept": "application/geo+json",
    }
    with httpx.Client(timeout=60, headers=headers) as client:
        gauges = _list_gauges_in_huc8(client, huc8)
        if not gauges:
            log.info("NWIS water-temp: no gauges found in HUC8 %s", huc8)
            return 0
        if max_gauges and len(gauges) > max_gauges:
            log.info("NWIS water-temp: capping %d gauges to %d (max_gauges)", len(gauges), max_gauges)
            gauges = gauges[:max_gauges]
        log.info("NWIS water-temp: %d candidate gauges in HUC8 %s", len(gauges), huc8)

        # Map gauges to a COMID by lat/lon proximity in the HUC8.
        # ST_Distance on GEOMETRY returns degrees in DuckDB spatial; that's
        # fine for nearest-neighbor *within* a single HUC8 (small extent).
        now = datetime.now(timezone.utc)
        since = now - timedelta(days=lookback_days)
        rows: list[tuple] = []
        for gauge_id, lat, lon in gauges:
            comid_row = conn.execute(
                """
                SELECT comid
                FROM reaches
                WHERE huc8 = ?
                ORDER BY ST_Distance(
                    geometry,
                    ST_Point(?, ?)
                ) ASC
                LIMIT 1
                """,
                [huc8, lon, lat],
            ).fetchone()
            if comid_row is None:
                continue
            comid = int(comid_row[0])
            try:
                samples = list(_continuous_water_temp(client, gauge_id, since, now))
            except httpx.HTTPError as exc:
                log.debug("Gauge %s water-temp fetch failed: %s", gauge_id, exc)
                continue
            # Politeness pause between per-gauge calls to keep NWIS happy.
            time.sleep(0.4)
            if not samples:
                continue
            # Daily average per gauge.
            by_day: dict[date, list[float]] = {}
            for ts, t_c in samples:
                by_day.setdefault(ts.date(), []).append(t_c)
            for d, vals in by_day.items():
                mean_c = sum(vals) / len(vals)
                # Standard deviation of the gauge readings within the day,
                # as a real uncertainty proxy.
                if len(vals) > 1:
                    mu = mean_c
                    var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
                    sd = var ** 0.5
                else:
                    sd = None
                rows.append((comid, d.isoformat(), mean_c, sd))

    if not rows:
        return 0
    ingested_at = datetime.now(timezone.utc).isoformat()
    with store.transaction() as conn:
        conn.executemany(
            """
            INSERT INTO reach_temperature
                (comid, date, temperature_c, uncertainty_c, source, ingested_at)
            VALUES (?, ?, ?, ?, 'NWIS_obs', ?)
            ON CONFLICT (comid, date, source) DO UPDATE SET
                temperature_c = EXCLUDED.temperature_c,
                uncertainty_c = EXCLUDED.uncertainty_c,
                ingested_at = EXCLUDED.ingested_at
            """,
            [(c, d, t, u, ingested_at) for c, d, t, u in rows],
        )
    log.info("NWIS water-temp: wrote %d reach-day rows for HUC8 %s", len(rows), huc8)
    return len(rows)


def _list_gauges_in_huc8(
    client: httpx.Client, huc8: str,
) -> list[tuple[str, float, float]]:
    """Yield (full_monitoring_location_id, lat, lon) for stream gauges in HUC8."""
    params = {
        "hydrologic_unit_code": huc8,
        "limit": 1000,
        "site_type_code": "ST",
        "agency_code": "USGS",
        "f": "json",
    }
    r = _get_with_backoff(client, f"{OGC_ROOT}/collections/monitoring-locations/items", params)
    r.raise_for_status()
    out: list[tuple[str, float, float]] = []
    for feat in r.json().get("features", []):
        props = feat.get("properties") or {}
        full_id = (props.get("id") or "").strip()
        if not full_id:
            continue
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates")
        if not coords or len(coords) < 2:
            continue
        try:
            lon = float(coords[0])
            lat = float(coords[1])
        except (TypeError, ValueError):
            continue
        out.append((full_id, lat, lon))
    return out


def _continuous_water_temp(
    client: httpx.Client,
    monitoring_location_id: str,
    since: datetime,
    until: datetime,
):
    """Yield (timestamp, water_temp_c) for a gauge over the time window."""
    params = {
        "monitoring_location_id": monitoring_location_id,
        "parameter_code": WATER_TEMP_PCODE,
        "datetime": f"{since.isoformat()}/{until.isoformat()}",
        "limit": 5000,
        "f": "json",
    }
    r = _get_with_backoff(client, f"{OGC_ROOT}/collections/continuous/items", params)
    if r.status_code == 404:
        return
    r.raise_for_status()
    for feat in r.json().get("features", []):
        props = feat.get("properties") or {}
        ts_str = props.get("time")
        value = props.get("value")
        if ts_str is None or value is None:
            continue
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        try:
            yield (ts, float(value))
        except (TypeError, ValueError):
            continue
