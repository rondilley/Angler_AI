"""Tests that the M5 map_export wiring carries temperature_source per Feature."""

from __future__ import annotations

from angler_ai.prediction.map_export import build_feature_collection
from angler_ai.prediction.species_priors import SpeciesPrior, _calibrate
from angler_ai.prediction.temperature import ReachTemperature


def _make_prior(comid: int = 100) -> tuple[SpeciesPrior, str]:
    cp = _calibrate(
        raw_p=0.5,
        v2_join_method="reachcode_exact",
        v2_join_confidence=1.0,
        prevalence=0.3,
        species="Salvelinus fontinalis",
    )
    sp = SpeciesPrior(
        comid=comid,
        species="Salvelinus fontinalis",
        common_name="brook trout",
        probability=cp,
        v2_join_method="reachcode_exact",
    )
    return sp, "MULTILINESTRING((-77 41, -77.1 41.05))"


def test_map_export_surfaces_not_modeled_when_no_temperature() -> None:
    """The honest default: no temperature data -> source='not_modeled'."""
    rows = [_make_prior(comid=100)]
    export = build_feature_collection(
        rows,
        species_scientific="Salvelinus fontinalis",
        species_common="brook trout",
        date="2026-07-04",
        temperatures=None,  # no resolver output
    )
    feat = export.raw_geojson["features"][0]
    props = feat["properties"]
    assert props["temperature_source"] == "not_modeled"
    assert props["temperature_c"] is None
    assert props["temperature_uncertainty_c"] is None


def test_map_export_surfaces_real_temperature_when_available() -> None:
    sp, wkt = _make_prior(comid=200)
    temps = {
        200: ReachTemperature(
            comid=200,
            temperature_c=17.4,
            uncertainty_c=0.9,
            source="EcoSHEDS_TEMP",
            date="2024-08-15",
        ),
    }
    export = build_feature_collection(
        [(sp, wkt)],
        species_scientific="Salvelinus fontinalis",
        species_common="brook trout",
        date="2026-07-04",
        temperatures=temps,
    )
    props = export.raw_geojson["features"][0]["properties"]
    assert props["temperature_source"] == "EcoSHEDS_TEMP"
    assert props["temperature_c"] == 17.4
    assert props["temperature_uncertainty_c"] == 0.9
    assert props["temperature_date"] == "2024-08-15"


def test_temperature_source_is_in_documented_vocabulary() -> None:
    """Source values must come from the M5-documented set; no surprise tags."""
    allowed = {
        "EcoSHEDS_TEMP", "NorWeST", "PG-GNN", "NWIS_interp", "not_modeled",
    }
    for src in allowed:
        sp, wkt = _make_prior(comid=300)
        temps = {300: ReachTemperature(
            comid=300, temperature_c=18.0, uncertainty_c=1.0,
            source=src, date="2024-08-15",
        )}
        export = build_feature_collection(
            [(sp, wkt)],
            species_scientific="Salvelinus fontinalis",
            species_common="brook trout",
            date="2026-07-04",
            temperatures=temps,
        )
        out_src = export.raw_geojson["features"][0]["properties"]["temperature_source"]
        assert out_src in allowed, f"unexpected source {out_src!r}"
