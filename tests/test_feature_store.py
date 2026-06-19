"""Tests for the DuckDB feature store."""

from __future__ import annotations

from pathlib import Path

from angler_ai.features import open_store


def test_schema_initializes(tmp_path: Path) -> None:
    """Schema SQL is idempotent and creates all expected tables."""
    db_path = tmp_path / "test.duckdb"
    expected = {
        "reaches",
        "xwalk_v2_to_hr",
        "xwalk_necd_to_hr",
        "reach_temperature",
        "reach_flow",
        "reach_wq",
        "attains_status",
        "brt_priors",
        "stocking_events",
        "regulations",
        "sensitive_species",
        "tribal_mask",
        "model_selection_log",
        "calibration_log",
    }
    with open_store(db_path) as store:
        store.initialize_schema()
        # Idempotent: run twice.
        store.initialize_schema()
        rows = store.connect().execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    actual = {r[0] for r in rows}
    missing = expected - actual
    assert not missing, f"Schema missing tables: {missing}"
