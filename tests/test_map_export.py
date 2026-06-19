"""Tests for the M4 calibration + GeoJSON map export pipeline.

Locked invariants:
- Interval contains point on every feature (FR-6.4)
- Hyperstability beta is applied per CPUE-derived weight (FR-6.1)
- huc10_proximity reaches have wider intervals than reachcode_exact (M4 gate)
- GeoJSON FeatureCollection carries model + calibration provenance
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from angler_ai.calibration.types import CalibratedProbability, ProbabilityBasis
from angler_ai.prediction.map_export import (
    GeoJSONExport,
    _parse_coords,
    _wkt_to_geojson_geometry,
    build_feature_collection,
)
from angler_ai.prediction.species_priors import SpeciesPrior, _calibrate


def _make_prior(
    raw_p: float,
    method: str = "reachcode_exact",
    confidence: float = 1.0,
    prevalence: float | None = 0.3,
    species: str = "Salvelinus fontinalis",
    common: str = "brook trout",
    comid: int = 12345,
) -> SpeciesPrior:
    cp = _calibrate(raw_p=raw_p, v2_join_method=method,
                    v2_join_confidence=confidence,
                    prevalence=prevalence, species=species)
    return SpeciesPrior(
        comid=comid,
        species=species,
        common_name=common,
        probability=cp,
        v2_join_method=method,
    )


# ---------- calibration math ----------------------------------------------


def test_calibrate_records_hyperstability_constant_but_skips_application() -> None:
    """v0 behavior post multi-AI review STAT-01: the Charbonneau hyperstability
    correction (p^(1/0.23)) is a statistical category error when applied to a
    unit-interval BRT presence probability. Until M4 catch-data validation, the
    SpeciesPrior records the constant on the CalibratedProbability for
    provenance but the correction is NOT applied (cpue_weight=0). The basis
    sources tuple includes 'hyperstability:not_applied(...)' so consumers can
    see the intentional skip.
    """
    sp = _make_prior(raw_p=0.8)
    assert sp.probability.hyperstability_beta_applied == 0.23
    assert sp.probability.raw_point == 0.8
    # At v0 the calibrated point equals the raw BRT predict_prob.
    assert sp.probability.point == 0.8
    # Source tag must surface the intentional skip.
    sources = sp.probability.basis.sources
    assert any("hyperstability:not_applied" in s for s in sources), sources


def test_calibrate_interval_contains_point() -> None:
    for raw in (0.05, 0.25, 0.5, 0.75, 0.95):
        sp = _make_prior(raw_p=raw)
        cp = sp.probability
        assert cp.lower <= cp.point <= cp.upper, (
            f"interval violation at raw={raw}: {cp.lower} <= {cp.point} <= {cp.upper}"
        )


def test_huc10_proximity_intervals_wider_than_reachcode_exact() -> None:
    """M4 gate: lower-confidence joins must surface wider uncertainty."""
    raw = 0.6
    exact = _make_prior(raw_p=raw, method="reachcode_exact")
    proximity = _make_prior(raw_p=raw, method="huc10_proximity")
    w_exact = exact.probability.upper - exact.probability.lower
    w_prox = proximity.probability.upper - proximity.probability.lower
    assert w_prox > w_exact, (
        f"proximity width ({w_prox:.3f}) must exceed exact width ({w_exact:.3f})"
    )


def test_calibrate_basis_records_sources() -> None:
    sp = _make_prior(raw_p=0.4)
    sources = sp.probability.basis.sources
    assert "USGS_BRT_V2.0" in sources
    assert any("xwalk" in s for s in sources)
    assert any("hyperstability" in s for s in sources)


def test_calibrated_probability_resists_invalid_interval() -> None:
    """The dataclass itself prevents inverted intervals (FR-6.2)."""
    with pytest.raises(ValueError):
        CalibratedProbability(point=0.5, lower=0.7, upper=0.4)


# ---------- WKT -> GeoJSON conversion -------------------------------------


def test_parse_coords_simple() -> None:
    coords = _parse_coords("-77.0 41.0, -77.1 41.05")
    assert coords == [[-77.0, 41.0], [-77.1, 41.05]]


def test_wkt_to_geojson_linestring_single_path() -> None:
    wkt = "MULTILINESTRING((-77.0 41.0, -77.1 41.05))"
    geom = _wkt_to_geojson_geometry(wkt)
    assert geom == {"type": "LineString", "coordinates": [[-77.0, 41.0], [-77.1, 41.05]]}


def test_wkt_to_geojson_multilinestring() -> None:
    wkt = "MULTILINESTRING((-77.0 41.0, -77.1 41.05), (-78.0 40.0, -78.1 40.05))"
    geom = _wkt_to_geojson_geometry(wkt)
    assert geom is not None
    assert geom["type"] == "MultiLineString"
    assert len(geom["coordinates"]) == 2


def test_wkt_to_geojson_empty_or_invalid() -> None:
    assert _wkt_to_geojson_geometry("") is None
    assert _wkt_to_geojson_geometry("POLYGON((0 0, 1 0, 0 1, 0 0))") is None


# ---------- feature collection assembly -----------------------------------


def test_build_feature_collection_carries_calibration_fields() -> None:
    sp = _make_prior(raw_p=0.65)
    wkt = "MULTILINESTRING((-77 41, -77.1 41.05))"
    export = build_feature_collection(
        [(sp, wkt)],
        species_scientific="Salvelinus fontinalis",
        species_common="brook trout",
        date="2026-07-04",
    )
    assert export.feature_count == 1
    fc = export.raw_geojson
    assert fc["type"] == "FeatureCollection"
    assert fc["metadata"]["model"]["id"] == "USGS_BRT_V2.0"
    assert fc["metadata"]["calibration"]["hyperstability_beta"] == 0.23
    feat = fc["features"][0]
    props = feat["properties"]
    assert props["hyperstability_beta_applied"] == 0.23
    assert props["lower"] <= props["probability"] <= props["upper"]
    assert "USGS_BRT_V2.0" in props["basis"]["sources"]


def test_build_feature_collection_drops_invalid_geometry() -> None:
    sp = _make_prior(raw_p=0.5)
    export = build_feature_collection(
        [(sp, "")],
        species_scientific="x",
        species_common=None,
        date="2026-07-04",
    )
    assert export.feature_count == 0


def test_geojson_serializes_to_json(tmp_path: Path) -> None:
    sp = _make_prior(raw_p=0.42)
    wkt = "MULTILINESTRING((-77 41, -77.1 41.05))"
    export = build_feature_collection(
        [(sp, wkt)],
        species_scientific="Salvelinus fontinalis",
        species_common="brook trout",
        date="2026-07-04",
    )
    out = tmp_path / "m.geojson"
    from angler_ai.prediction.map_export import write_geojson
    write_geojson(export, out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["type"] == "FeatureCollection"
    assert len(loaded["features"]) == 1
