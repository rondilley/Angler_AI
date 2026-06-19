"""Tests for the temperature resolver (M5).

Key invariants:
- No fabricated values: when reach_temperature has no row for a COMID, the
  resolver returns temperature_c=None and source='not_modeled'.
- Source priority: real sources are returned in EcoSHEDS_TEMP > NorWeST >
  PG-GNN > NWIS_interp order.
- BTO ceiling helper correctly identifies reaches above the 200 km^2 threshold.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from angler_ai.features import open_store
from angler_ai.prediction.temperature import (
    above_bto_ceiling,
    resolve,
    resolve_many,
)


def _seed(store, comid: int, source: str, temp_c: float, date_str: str = "2024-08-15") -> None:
    now = datetime.now(timezone.utc).isoformat()
    store.connect().execute(
        """
        INSERT INTO reach_temperature
            (comid, date, temperature_c, uncertainty_c, source, ingested_at)
        VALUES (?, ?, ?, NULL, ?, ?)
        """,
        [comid, date_str, temp_c, source, now],
    )


def test_resolve_returns_not_modeled_for_empty_reach(tmp_path: Path) -> None:
    """FR-6 honesty: no real data -> source='not_modeled', value None."""
    db = tmp_path / "t.duckdb"
    with open_store(db) as store:
        store.initialize_schema()
        rt = resolve(store, comid=999999)
        assert rt.source == "not_modeled"
        assert rt.temperature_c is None
        assert rt.uncertainty_c is None
        assert rt.date is None


def test_resolve_returns_real_source_when_available(tmp_path: Path) -> None:
    db = tmp_path / "t.duckdb"
    with open_store(db) as store:
        store.initialize_schema()
        _seed(store, comid=12345, source="EcoSHEDS_TEMP", temp_c=18.7)
        rt = resolve(store, comid=12345)
        assert rt.source == "EcoSHEDS_TEMP"
        assert rt.temperature_c == 18.7


def test_resolve_source_priority(tmp_path: Path) -> None:
    """When multiple real sources cover a reach, EcoSHEDS_TEMP wins."""
    db = tmp_path / "t.duckdb"
    with open_store(db) as store:
        store.initialize_schema()
        _seed(store, comid=42, source="PG-GNN", temp_c=20.0, date_str="2024-08-10")
        _seed(store, comid=42, source="EcoSHEDS_TEMP", temp_c=18.5, date_str="2024-08-15")
        rt = resolve(store, comid=42)
        assert rt.source == "EcoSHEDS_TEMP"
        assert rt.temperature_c == 18.5


def test_resolve_many_handles_mixed_coverage(tmp_path: Path) -> None:
    db = tmp_path / "t.duckdb"
    with open_store(db) as store:
        store.initialize_schema()
        _seed(store, comid=1, source="EcoSHEDS_TEMP", temp_c=17.2)
        _seed(store, comid=2, source="NorWeST", temp_c=14.8)
        # comid=3 intentionally has no row
        results = resolve_many(store, [1, 2, 3])
        assert results[1].source == "EcoSHEDS_TEMP"
        assert results[2].source == "NorWeST"
        assert results[3].source == "not_modeled"
        assert results[3].temperature_c is None


def test_resolver_never_returns_proxy_source(tmp_path: Path) -> None:
    """A row tagged with a proxy source should NOT be picked up; only real
    sources from the documented vocabulary are honored."""
    db = tmp_path / "t.duckdb"
    with open_store(db) as store:
        store.initialize_schema()
        _seed(store, comid=7, source="EcoSHEDS_proxy_v0", temp_c=22.5)
        _seed(store, comid=7, source="NorWeST_proxy_v0", temp_c=21.0)
        rt = resolve(store, comid=7)
        # Both rows are "proxy" sources outside the real-source priority list.
        assert rt.source == "not_modeled"
        assert rt.temperature_c is None


def test_above_bto_ceiling_threshold() -> None:
    assert above_bto_ceiling(199.9) is False
    assert above_bto_ceiling(200.0) is True
    assert above_bto_ceiling(201.0) is True
    # Missing drainage: cannot apply ceiling -> False (don't suppress).
    assert above_bto_ceiling(None) is False
