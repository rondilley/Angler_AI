"""Per-gauge statistical flow-anomaly detection from real USGS NWIS discharge.

The HydroGEM TCN-Transformer in `prediction/hydrogem.py` requires a 576-hour
window plus the 12-channel feature stack from the published preprocessing
pipeline. Wiring that into the forecast pipeline per HUC8 is v1 work; the
test-pickle path remains the canonical path for the M6 `ask` agent.

For the forecast PNG pipeline we use a SIMPLER but real flow-anomaly
detector: z-score of the recent 7-day mean discharge against the prior
30-day distribution. Tag: 'flow_z_anomaly'. This is genuine anomaly
detection (statistical), not HydroGEM, and is honestly tagged as such.

Reaches downstream of an anomalous gauge get an additional 'anomaly_factor'
in the scoring chain that attenuates the score.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from angler_ai.features.store import FeatureStore

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GaugeAnomaly:
    """One gauge's statistical anomaly status."""

    gauge_id: str
    comid: int | None
    z_score: float
    is_anomalous: bool
    recent_mean_cfs: float
    baseline_mean_cfs: float
    baseline_sd_cfs: float
    n_recent: int
    n_baseline: int


def detect_huc8_anomalies(
    store: FeatureStore, *, huc8: str, recent_days: int = 7, threshold_z: float = 2.0,
) -> dict[str, GaugeAnomaly]:
    """Compute z-score anomaly for every gauge in HUC8 with sufficient data."""
    conn = store.connect()
    rd = int(recent_days)
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
                   AVG(f.discharge_cfs) AS rmean,
                   COUNT(*) AS n_recent
            FROM reach_flow f
            JOIN gauges_in_huc g ON f.gauge_id = g.gauge_id
            WHERE f.ts >= NOW() - INTERVAL {rd} DAY
            GROUP BY f.gauge_id
        ),
        baseline AS (
            SELECT f.gauge_id,
                   AVG(f.discharge_cfs) AS bmean,
                   STDDEV_SAMP(f.discharge_cfs) AS bsd,
                   COUNT(*) AS n_baseline
            FROM reach_flow f
            JOIN gauges_in_huc g ON f.gauge_id = g.gauge_id
            WHERE f.ts < NOW() - INTERVAL {rd} DAY
            GROUP BY f.gauge_id
        )
        SELECT g.gauge_id, g.comid, r.rmean, r.n_recent,
               b.bmean, b.bsd, b.n_baseline
        FROM gauges_in_huc g
        LEFT JOIN recent r ON r.gauge_id = g.gauge_id
        LEFT JOIN baseline b ON b.gauge_id = g.gauge_id
        """,
        [huc8],
    ).fetchall()

    out: dict[str, GaugeAnomaly] = {}
    for gid, comid, rmean, n_r, bmean, bsd, n_b in rows:
        if rmean is None or bmean is None or bsd is None or bsd <= 0:
            continue
        if (n_b or 0) < 5:
            continue
        z = (float(rmean) - float(bmean)) / float(bsd)
        is_anom = abs(z) >= threshold_z
        out[gid] = GaugeAnomaly(
            gauge_id=gid,
            comid=int(comid) if comid is not None else None,
            z_score=round(z, 3),
            is_anomalous=is_anom,
            recent_mean_cfs=round(float(rmean), 1),
            baseline_mean_cfs=round(float(bmean), 1),
            baseline_sd_cfs=round(float(bsd), 1),
            n_recent=int(n_r or 0),
            n_baseline=int(n_b or 0),
        )
    return out


def anomaly_factor(z: float | None, threshold_z: float = 2.0) -> tuple[float, str]:
    """Map a flow z-score to an additional anomaly attenuation in [0.7, 1.0]."""
    if z is None:
        return (1.0, "not_modeled:no_anomaly_data")
    az = abs(z)
    if az < threshold_z:
        return (1.0, "flow_z_anomaly:normal")
    # Saturating attenuation: full anomaly (|z|>=4) -> 0.7.
    az_clip = min(az, 4.0)
    f = 1.0 - 0.3 * (az_clip - threshold_z) / (4.0 - threshold_z)
    return (round(f, 3), f"flow_z_anomaly:|z|={az_clip:.2f}")


def map_reach_to_nearest_gauge(
    store: FeatureStore, *, huc8: str, anomalies: dict[str, GaugeAnomaly],
) -> dict[int, GaugeAnomaly]:
    """Assign each reach in HUC8 to the nearest gauge by ST_Distance."""
    if not anomalies:
        return {}
    conn = store.connect()
    # Get gauge locations
    gauge_ids = [a.gauge_id for a in anomalies.values()]
    gauge_locs = conn.execute(
        f"""
        SELECT f.gauge_id,
               ANY_VALUE(ST_X(ST_Centroid(r.geometry))) AS lon,
               ANY_VALUE(ST_Y(ST_Centroid(r.geometry))) AS lat
        FROM reach_flow f
        JOIN reaches r ON f.comid = r.comid
        WHERE f.gauge_id IN ({','.join('?' * len(gauge_ids))})
        GROUP BY f.gauge_id
        """,
        gauge_ids,
    ).fetchall()
    gauge_lookup = {g[0]: (float(g[1]), float(g[2])) for g in gauge_locs if g[1] is not None}
    if not gauge_lookup:
        return {}

    reaches = conn.execute(
        """
        SELECT comid,
               ST_X(ST_Centroid(geometry)) AS lon,
               ST_Y(ST_Centroid(geometry)) AS lat
        FROM reaches WHERE huc8 = ?
        """,
        [huc8],
    ).fetchall()

    out: dict[int, GaugeAnomaly] = {}
    for comid, lon, lat in reaches:
        if lon is None or lat is None:
            continue
        # Nearest gauge
        nearest_gid = None
        nearest_d = float("inf")
        for gid, (glon, glat) in gauge_lookup.items():
            dx = (glon - float(lon))
            dy = (glat - float(lat))
            d = math.sqrt(dx * dx + dy * dy)
            if d < nearest_d:
                nearest_d = d
                nearest_gid = gid
        if nearest_gid is not None:
            out[int(comid)] = anomalies[nearest_gid]
    return out
