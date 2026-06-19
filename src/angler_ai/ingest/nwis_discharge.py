"""USGS NWIS discharge ingest scoped to a HUC8. Used to derive a real
flow-condition factor for the forecast pipeline.

Pulls parameter code 00060 (discharge, cfs) for every NWIS stream gauge in
the HUC8 over a 30-day window, computes a per-gauge recent mean and median,
and persists into reach_flow. Existing usgs_nwis.py is state-wide and
state-FIPS-driven; this module is HUC8-scoped to keep the western-waters
pipeline tight.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import httpx

from angler_ai.features.store import FeatureStore
from angler_ai.ingest._bulk import bulk_insert
from angler_ai.ingest.nwis_water_temp import _get_with_backoff, _list_gauges_in_huc8

log = logging.getLogger(__name__)

OGC_ROOT = "https://api.waterdata.usgs.gov/ogcapi/v0"
DISCHARGE_PCODE = "00060"  # cfs


def ingest_discharge_for_huc8(
    store: FeatureStore,
    *,
    huc8: str,
    lookback_days: int = 30,
    max_gauges: int = 8,
) -> int:
    """Pull discharge series for each USGS gauge in `huc8`. Returns row count.

    Joins each gauge to the nearest NHDPlus HR reach by ST_Distance so
    `reach_flow.comid` is populated immediately.
    """
    conn = store.connect()
    headers = {
        "User-Agent": "angler-ai/0.1 (ron.dilley@gmail.com)",
        "Accept": "application/geo+json",
    }
    with httpx.Client(timeout=60, headers=headers) as client:
        gauges = _list_gauges_in_huc8(client, huc8)
        if not gauges:
            log.info("NWIS discharge: no gauges in HUC8 %s", huc8)
            return 0
        if max_gauges and len(gauges) > max_gauges:
            gauges = gauges[:max_gauges]
        log.info("NWIS discharge: %d gauges in HUC8 %s", len(gauges), huc8)

        now = datetime.now(timezone.utc)
        since = now - timedelta(days=lookback_days)
        rows: list[tuple] = []
        for gauge_id, lat, lon in gauges:
            comid_row = conn.execute(
                """
                SELECT comid
                FROM reaches
                WHERE huc8 = ?
                ORDER BY ST_Distance(geometry, ST_Point(?, ?)) ASC
                LIMIT 1
                """,
                [huc8, lon, lat],
            ).fetchone()
            comid = int(comid_row[0]) if comid_row else None
            try:
                params = {
                    "monitoring_location_id": gauge_id,
                    "parameter_code": DISCHARGE_PCODE,
                    "datetime": f"{since.isoformat()}/{now.isoformat()}",
                    "limit": 5000,
                    "f": "json",
                }
                r = _get_with_backoff(
                    client, f"{OGC_ROOT}/collections/continuous/items", params,
                )
                if r.status_code == 404:
                    continue
                r.raise_for_status()
            except httpx.HTTPError as exc:
                log.debug("Gauge %s discharge fetch failed: %s", gauge_id, exc)
                continue
            time.sleep(0.4)  # politeness
            for feat in r.json().get("features", []):
                props = feat.get("properties") or {}
                ts_str = props.get("time")
                value = props.get("value")
                if ts_str is None or value is None:
                    continue
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    rows.append((comid, gauge_id, ts.isoformat(), float(value)))
                except (TypeError, ValueError):
                    continue

    if not rows:
        return 0
    ingested_at = datetime.now(timezone.utc).isoformat()
    with store.transaction() as conn:
        conn.execute(
            "DELETE FROM reach_flow WHERE source = 'USGS_NWIS' AND ts >= ?",
            [since.isoformat()],
        )
        bulk_insert(
            conn, "reach_flow", rows,
            columns=("comid", "gauge_id", "ts", "discharge_cfs"),
            extra_literals={"source": "USGS_NWIS", "ingested_at": ingested_at},
        )
    log.info("NWIS discharge: wrote %d samples for HUC8 %s", len(rows), huc8)
    return len(rows)


def discharge_summary_per_gauge(
    store: FeatureStore, *, huc8: str, recent_days: int = 7,
) -> dict[str, dict]:
    """For each gauge in the HUC8, return summary stats over the recent vs
    full lookback window. Returns dict keyed by gauge_id with:
        comid, recent_mean_cfs, baseline_median_cfs, ratio, n_recent, n_baseline.
    """
    conn = store.connect()
    rd = int(recent_days)  # interpolated literal; DuckDB INTERVAL takes no params
    rows = conn.execute(
        f"""
        WITH gauges_in_huc AS (
            SELECT DISTINCT f.gauge_id, f.comid
            FROM reach_flow f
            JOIN reaches r ON f.comid = r.comid
            WHERE r.huc8 = ? AND f.source = 'USGS_NWIS'
        ),
        recent AS (
            SELECT f.gauge_id,
                   AVG(f.discharge_cfs) AS recent_mean,
                   COUNT(*) AS n_recent
            FROM reach_flow f
            JOIN gauges_in_huc g ON f.gauge_id = g.gauge_id
            WHERE f.ts >= NOW() - INTERVAL {rd} DAY
            GROUP BY f.gauge_id
        ),
        baseline AS (
            SELECT f.gauge_id,
                   MEDIAN(f.discharge_cfs) AS baseline_median,
                   COUNT(*) AS n_baseline
            FROM reach_flow f
            JOIN gauges_in_huc g ON f.gauge_id = g.gauge_id
            GROUP BY f.gauge_id
        )
        SELECT g.gauge_id, g.comid, r.recent_mean, b.baseline_median,
               r.n_recent, b.n_baseline
        FROM gauges_in_huc g
        LEFT JOIN recent r ON r.gauge_id = g.gauge_id
        LEFT JOIN baseline b ON b.gauge_id = g.gauge_id
        """,
        [huc8],
    ).fetchall()
    out: dict[str, dict] = {}
    for gid, comid, rmean, bmed, n_r, n_b in rows:
        ratio = None
        if rmean is not None and bmed is not None and bmed > 0:
            ratio = float(rmean) / float(bmed)
        out[gid] = {
            "comid": int(comid) if comid is not None else None,
            "recent_mean_cfs": float(rmean) if rmean is not None else None,
            "baseline_median_cfs": float(bmed) if bmed is not None else None,
            "ratio": ratio,
            "n_recent": int(n_r) if n_r is not None else 0,
            "n_baseline": int(n_b) if n_b is not None else 0,
        }
    return out


def discharge_flow_factor(ratio: float | None) -> tuple[float, str]:
    """Map a recent/baseline discharge ratio to a flow factor in [0.5, 1.0].

    Bell 2022 review: fluvial salmonid catchability peaks near typical flow
    and falls off at both high (turbid, dispersed) and low (stressed,
    concentrated but hard to reach) extremes.

      - ratio in [0.8, 1.3]: 1.0  (typical)
      - ratio < 0.5:         0.7  (very low; fish stressed, slow feeding)
      - ratio > 2.0:         0.5  (high water; turbid, fish dispersed)
      - linear interp in between
    """
    if ratio is None:
        return (1.0, "not_modeled:no_discharge_data")
    if 0.8 <= ratio <= 1.3:
        return (1.0, "discharge_ratio:typical")
    if ratio < 0.5:
        return (0.7, "discharge_ratio:very_low")
    if ratio > 2.0:
        return (0.5, "discharge_ratio:very_high")
    if ratio < 0.8:
        # linear from (0.5, 0.7) to (0.8, 1.0)
        f = 0.7 + (ratio - 0.5) * (1.0 - 0.7) / (0.8 - 0.5)
        return (round(f, 3), "discharge_ratio:low")
    # ratio in (1.3, 2.0)
    f = 1.0 - (ratio - 1.3) * (1.0 - 0.5) / (2.0 - 1.3)
    return (round(f, 3), "discharge_ratio:high")
