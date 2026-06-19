"""Per-reach temperature resolver. Returns the best available real
temperature record + source tag, or honest `not_modeled` when no real
temperature record exists for a reach.

Source vocabulary (in resolution priority order):
  - 'NWIS_obs'        (USGS gauge observation at the reach; real measurement)
  - 'EcoSHEDS_TEMP'   (East, Letcher hierarchical Bayesian model; v1)
  - 'NorWeST'         (West, USFS shapefile scenarios; v1)
  - 'PG-GNN'          (physics-guided GNN nowcast; v1+)
  - 'NWIS_interp'     (USGS NWIS observed nearby interpolated; v0.5)
  - 'not_modeled'     (honest reason: no real source covers this reach)

No fabricated values. If `reach_temperature` is empty for a reach, the
resolver returns None and the source 'not_modeled'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from angler_ai.features.store import FeatureStore

# Resolution priority. Real sources listed first; never includes proxy entries.
# 'NWIS_obs' is direct gauge measurement at the reach: highest authority.
# 'NWIS_interp_air_adjusted' is IDW-anchored to a real observation but with
# a per-day delta from Mohseni-Stefan air-temp; below modeled sources but
# above pure air-only projection.
_PRIORITY: tuple[str, ...] = (
    "NWIS_obs",
    "EcoSHEDS_TEMP",
    "NorWeST",
    "PG-GNN",
    "NWIS_interp",
    "NWIS_interp_air_adjusted",
)


@dataclass(frozen=True, slots=True)
class ReachTemperature:
    """Resolved temperature for one reach. None values when not modeled."""

    comid: int
    temperature_c: float | None
    uncertainty_c: float | None
    source: str
    """One of _PRIORITY values, or 'not_modeled' when no real source covers
    the reach."""

    date: str | None = None


def resolve(store: FeatureStore, comid: int) -> ReachTemperature:
    """Resolve the best available temperature for one reach.

    Honors source priority; returns 'not_modeled' if no real source covers
    the reach. NEVER returns a fabricated value.
    """
    conn = store.connect()
    row = conn.execute(
        f"""
        SELECT temperature_c, uncertainty_c, source, date
        FROM reach_temperature
        WHERE comid = ?
          AND source IN ({",".join("?" * len(_PRIORITY))})
        ORDER BY array_position(?::VARCHAR[], source)
        LIMIT 1
        """,
        [comid, *_PRIORITY, list(_PRIORITY)],
    ).fetchone()
    if row is None:
        return ReachTemperature(
            comid=comid,
            temperature_c=None,
            uncertainty_c=None,
            source="not_modeled",
            date=None,
        )
    return ReachTemperature(
        comid=comid,
        temperature_c=float(row[0]) if row[0] is not None else None,
        uncertainty_c=float(row[1]) if row[1] is not None else None,
        source=row[2],
        date=row[3].isoformat() if hasattr(row[3], "isoformat") else row[3],
    )


def resolve_many(store: FeatureStore, comids: Iterable[int]) -> dict[int, ReachTemperature]:
    """Bulk resolution. Returns a dict keyed by COMID."""
    cids = list(comids)
    if not cids:
        return {}
    conn = store.connect()
    placeholders = ",".join("?" * len(cids))
    src_placeholders = ",".join("?" * len(_PRIORITY))
    rows = conn.execute(
        f"""
        WITH ranked AS (
            SELECT comid, temperature_c, uncertainty_c, source, date,
                   ROW_NUMBER() OVER (
                       PARTITION BY comid
                       ORDER BY array_position(?::VARCHAR[], source)
                   ) AS rn
            FROM reach_temperature
            WHERE comid IN ({placeholders})
              AND source IN ({src_placeholders})
        )
        SELECT comid, temperature_c, uncertainty_c, source, date
        FROM ranked WHERE rn = 1
        """,
        [list(_PRIORITY), *cids, *_PRIORITY],
    ).fetchall()
    out: dict[int, ReachTemperature] = {}
    for comid, t, u, src, date in rows:
        out[int(comid)] = ReachTemperature(
            comid=int(comid),
            temperature_c=float(t) if t is not None else None,
            uncertainty_c=float(u) if u is not None else None,
            source=src,
            date=date.isoformat() if hasattr(date, "isoformat") else date,
        )
    # Reaches with no record get the explicit not_modeled tag.
    for comid in cids:
        if comid not in out:
            out[comid] = ReachTemperature(
                comid=comid,
                temperature_c=None,
                uncertainty_c=None,
                source="not_modeled",
                date=None,
            )
    return out


# BTO ceiling enforcement helper.
def above_bto_ceiling(drainage_km2: float | None, ceiling_km2: float = 200.0) -> bool:
    """Return True if the reach is above the EcoSHEDS BTO drainage ceiling.

    EcoSHEDS BTO is restricted to lower-order headwater streams (drainage
    < 200 km^2). Reaches above the ceiling are NOT modeled by BTO; downstream
    output must surface 'BTO_not_applicable' rather than extrapolating.
    """
    if drainage_km2 is None:
        return False
    return drainage_km2 >= ceiling_km2
