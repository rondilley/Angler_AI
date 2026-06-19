"""Tool surface for the Analyst agent (FR-7.6).

These functions query the REAL DuckDB feature store; they never fabricate
results. When a tool can't answer (e.g., reach not loaded, species not in
BRT registry, gauge not in reach_flow), it raises ToolError so the agent
sees the failure rather than a fake answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from angler_ai.calibration.types import CalibratedProbability
from angler_ai.features.store import FeatureStore
from angler_ai.prediction.hydrogem import AnomalyStatus
from angler_ai.prediction.species_priors import species_priors_for_reach
from angler_ai.prediction.temperature import resolve as resolve_temperature

log = logging.getLogger(__name__)


class ToolError(Exception):
    """Raised when a tool cannot honestly answer the question."""


# ----------------------------------------------------------------------- types


@dataclass(frozen=True, slots=True)
class Reach:
    comid: int
    gnis_name: str | None
    state_fips: str | None
    huc8: str | None
    reachcode: str | None
    drainage_area_km2: float | None
    stream_order: int | None


@dataclass(frozen=True, slots=True)
class StockingEvent:
    comid: int | None
    state: str
    event_date: str
    species: str
    count: int | None
    water_body_name: str | None


@dataclass(frozen=True, slots=True)
class Regulation:
    state: str
    water_body_id: str | None
    comid: int | None
    species: str | None
    season_start: str | None
    season_end: str | None
    gear_restrictions: str | None
    bag_limit: int | None
    license_required: bool | None
    special_regulation: str | None
    source_url: str | None


# ----------------------------------------------------------------------- tools


def get_reach(store: FeatureStore, comid: int) -> Reach:
    """Look up one reach by NHDPlus HR COMID."""
    row = store.connect().execute(
        """
        SELECT comid, gnis_name, state_fips, huc8, reachcode,
               drainage_area_km2, stream_order
        FROM reaches WHERE comid = ?
        """,
        [comid],
    ).fetchone()
    if row is None:
        raise ToolError(f"COMID {comid} not in reaches table.")
    return Reach(
        comid=row[0], gnis_name=row[1], state_fips=row[2], huc8=row[3],
        reachcode=row[4], drainage_area_km2=row[5], stream_order=row[6],
    )


def get_reaches_in_huc8(store: FeatureStore, huc8: str, limit: int = 50) -> list[Reach]:
    """Return reaches within an 8-digit HUC8."""
    rows = store.connect().execute(
        """
        SELECT comid, gnis_name, state_fips, huc8, reachcode,
               drainage_area_km2, stream_order
        FROM reaches WHERE huc8 = ? LIMIT ?
        """,
        [huc8, int(limit)],
    ).fetchall()
    if not rows:
        raise ToolError(f"No reaches loaded for HUC8 {huc8}.")
    return [
        Reach(comid=r[0], gnis_name=r[1], state_fips=r[2], huc8=r[3],
              reachcode=r[4], drainage_area_km2=r[5], stream_order=r[6])
        for r in rows
    ]


def get_catch_probability(
    store: FeatureStore, comid: int, species_scientific: str,
) -> CalibratedProbability:
    """Return the CalibratedProbability for a species at a reach via BRT.

    Real value from species_priors -> CalibratedProbability with hyperstability
    correction and interval. Raises ToolError if no BRT prior exists.
    """
    priors = species_priors_for_reach(store, comid_hr=comid, species_scientific=species_scientific)
    if not priors:
        raise ToolError(
            f"No BRT prior for species {species_scientific!r} at COMID {comid}. "
            "The species may be outside the BRT native-range model, or the "
            "reach lacks a V2.1 join."
        )
    # MAX-over-matched-V2 follows the same aggregation pattern as the
    # species_priors_for_geometry path.
    return priors[0].probability


def get_top_species(
    store: FeatureStore, comid: int, top_k: int = 5,
) -> list[tuple[str, str | None, CalibratedProbability]]:
    """Top-K species at a reach by calibrated probability."""
    priors = species_priors_for_reach(store, comid_hr=comid, top_k=top_k)
    if not priors:
        raise ToolError(f"No BRT priors for COMID {comid}.")
    return [(p.species, p.common_name, p.probability) for p in priors]


def get_stocking_history(
    store: FeatureStore, *, water_body_name: str | None = None,
    state: str = "PA", lookback_days: int | None = None,
) -> list[StockingEvent]:
    """Stocking events for a water body (or all water bodies in state) within
    the lookback window."""
    where: list[str] = ["state = ?"]
    params: list[Any] = [state]
    if water_body_name:
        where.append("LOWER(water_body_name) LIKE ?")
        params.append(f"%{water_body_name.lower()}%")
    if lookback_days is not None:
        from datetime import date, timedelta
        threshold = (date.today() - timedelta(days=int(lookback_days))).isoformat()
        where.append("event_date >= ?")
        params.append(threshold)
    sql = (
        "SELECT comid, state, event_date, species, count, water_body_name "
        "FROM stocking_events WHERE " + " AND ".join(where) +
        " ORDER BY event_date DESC LIMIT 200"
    )
    rows = store.connect().execute(sql, params).fetchall()
    if not rows:
        scope = f"in {state}"
        if water_body_name:
            scope = f"for water body matching {water_body_name!r} in {state}"
        raise ToolError(f"No stocking events found {scope}.")
    return [
        StockingEvent(
            comid=r[0], state=r[1],
            event_date=r[2].isoformat() if hasattr(r[2], "isoformat") else r[2],
            species=r[3], count=r[4], water_body_name=r[5],
        )
        for r in rows
    ]


def get_regulations(
    store: FeatureStore, *, state: str, species: str | None = None,
    water_body_name: str | None = None,
) -> list[Regulation]:
    """Regulations in the given state, optionally filtered by species + water."""
    where: list[str] = ["state = ?"]
    params: list[Any] = [state]
    if species:
        where.append("(species IS NULL OR LOWER(species) LIKE ?)")
        params.append(f"%{species.lower()}%")
    if water_body_name:
        where.append("(water_body_id IS NULL OR LOWER(water_body_id) LIKE ?)")
        params.append(f"%{water_body_name.lower()}%")
    sql = (
        "SELECT state, water_body_id, comid, species, season_start, season_end, "
        "gear_restrictions, bag_limit, license_required, special_regulation, source_url "
        "FROM regulations WHERE " + " AND ".join(where) +
        " ORDER BY water_body_id LIMIT 50"
    )
    rows = store.connect().execute(sql, params).fetchall()
    if not rows:
        scope = f"state={state}"
        if species:
            scope += f", species={species}"
        if water_body_name:
            scope += f", water={water_body_name}"
        raise ToolError(f"No regulations found for {scope}.")
    return [
        Regulation(
            state=r[0], water_body_id=r[1], comid=r[2], species=r[3],
            season_start=r[4].isoformat() if hasattr(r[4], "isoformat") else r[4],
            season_end=r[5].isoformat() if hasattr(r[5], "isoformat") else r[5],
            gear_restrictions=r[6], bag_limit=r[7], license_required=r[8],
            special_regulation=r[9], source_url=r[10],
        )
        for r in rows
    ]


def get_temperature(store: FeatureStore, comid: int) -> dict:
    """Resolved temperature with explicit source. May return 'not_modeled'."""
    rt = resolve_temperature(store, comid)
    return {
        "comid": rt.comid,
        "temperature_c": rt.temperature_c,
        "uncertainty_c": rt.uncertainty_c,
        "source": rt.source,
        "date": rt.date,
    }


def get_attains_status(store: FeatureStore, comid: int) -> list[dict]:
    """EPA ATTAINS impaired-waters status rows for a COMID."""
    rows = store.connect().execute(
        "SELECT cycle_year, status, parameter, state_305b_url "
        "FROM attains_status WHERE comid = ? AND status IS NOT NULL "
        "ORDER BY cycle_year DESC LIMIT 10",
        [comid],
    ).fetchall()
    return [
        {"cycle_year": r[0], "status": r[1], "parameter": r[2],
         "state_305b_url": r[3]}
        for r in rows
    ]


def hydrogem_flow_anomaly(store: FeatureStore, gauge_id: str) -> AnomalyStatus:
    """Run HydroGEM on a USGS gauge's loaded flow window.

    For v0 this requires the gauge to have at least SEQUENCE_LENGTH (576)
    consecutive hourly samples in reach_flow. Our M2 NWIS loader pulled a
    24h snapshot, which is too short for the fixed-window model; this tool
    raises ToolError in that case rather than fabricating an answer.
    """
    rows = store.connect().execute(
        "SELECT ts, discharge_cfs FROM reach_flow "
        "WHERE gauge_id = ? AND source = 'USGS_NWIS' "
        "ORDER BY ts ASC",
        [gauge_id],
    ).fetchall()
    if not rows:
        raise ToolError(f"No flow samples loaded for gauge {gauge_id}.")
    raise ToolError(
        f"Gauge {gauge_id} has {len(rows)} samples; HydroGEM needs 576 "
        "consecutive hourly samples. Use M6 synthetic-anomaly test set for "
        "the smoke demo, or expand the NWIS lookback in v0.5."
    )


def hydrogem_synthetic_test(sample_index: int = 0) -> tuple[AnomalyStatus, dict]:
    """Run HydroGEM on a sample from the published synthetic-anomaly test set.

    The published `test_synthetic_mini.pkl` contains 4,500 USGS sequences with
    deliberately injected anomalies (debris effects, sensor faults, etc.) plus
    ground-truth segment labels. This tool runs the real HydroGEM model on a
    chosen sequence and returns the AnomalyStatus plus the ground-truth
    metadata so the Analyst can compare predicted vs known.

    Args:
        sample_index: zero-based index into the test set.

    Returns:
        (AnomalyStatus, ground_truth_meta) tuple. ground_truth_meta carries
        the synthetic segment(s) for honest comparison.
    """
    import pickle

    import numpy as np

    from angler_ai.config import default_paths
    from angler_ai.prediction.hydrogem import detect_anomaly

    test_path = (
        default_paths().cache_dir / "models" / "hydrogem" / "test_synthetic_mini.pkl"
    )
    if not test_path.exists():
        raise ToolError(
            f"HydroGEM synthetic test set not found at {test_path}. "
            "Download with `hf_hub_download('Ejokhan/HydroGEM', "
            "'test_synthetic_mini.pkl')`."
        )
    with test_path.open("rb") as fh:
        data = pickle.load(fh)
    if not 0 <= sample_index < len(data):
        raise ToolError(
            f"sample_index {sample_index} out of range; "
            f"test set has {len(data)} sequences."
        )
    x, _, meta = data[sample_index]
    if hasattr(x, "numpy"):
        x = x.numpy()
    x = np.asarray(x, dtype=np.float32)
    site_id = str(meta.get("site_id", "?"))
    status = detect_anomaly(x, gauge_id=site_id)
    truth = {
        "site_id": site_id,
        "segments": meta.get("segments", []),
        "timestamp_start": str(meta.get("timestamp_start")),
        "timestamp_end": str(meta.get("timestamp_end")),
    }
    return status, truth
