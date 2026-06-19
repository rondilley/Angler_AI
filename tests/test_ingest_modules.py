"""Unit tests for individual ingest modules. Network-facing logic is patched
so the suite stays offline. Live-network tests are marked `network`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from angler_ai.features import open_store
from angler_ai.ingest.nhdplus_hr import (
    DEFAULT_HUC8_SAMPLE,
    NHDPlusHRIngest,
    _polyline_to_wkt,
    _state_fips,
)
from angler_ai.ingest.pa_pfbc import PAPFBCTroutStockedIngest, _coerce_pfbc_date


# ----- PA PFBC -----------------------------------------------------------


def test_pfbc_service_regex_matches_both_naming_conventions() -> None:
    """Regex must match both 'TroutStocked2024' and 'TroutStockedSections_2026'."""
    rx = PAPFBCTroutStockedIngest._SERVICE_RE
    assert rx.search("TroutStocked2024") is not None
    assert rx.search("TroutStocked2025") is not None
    assert rx.search("TroutStockedSections_2026") is not None
    assert rx.search("TroutSomethingElse2024") is None


def test_pfbc_service_regex_extracts_year() -> None:
    rx = PAPFBCTroutStockedIngest._SERVICE_RE
    assert int(rx.search("TroutStocked2024").group(1)) == 2024  # type: ignore[union-attr]
    assert int(rx.search("TroutStockedSections_2026").group(1)) == 2026  # type: ignore[union-attr]


def test_pfbc_coerce_pfbc_date_epoch_ms() -> None:
    """Esri epoch-ms ints decode to ISO dates."""
    # 2024-04-01 00:00:00 UTC = 1711929600 sec = 1711929600000 ms
    out = _coerce_pfbc_date(1711929600000, year=2024)
    assert out == "2024-04-01"


def test_pfbc_coerce_pfbc_date_string() -> None:
    assert _coerce_pfbc_date("2024-04-12T00:00:00", year=2024) == "2024-04-12"
    assert _coerce_pfbc_date("2024-04-12", year=2024) == "2024-04-12"


def test_pfbc_coerce_pfbc_date_none() -> None:
    assert _coerce_pfbc_date(None, year=2024) is None
    assert _coerce_pfbc_date("not-a-date", year=2024) is None


def test_pfbc_normalize_drops_empty_water() -> None:
    """Features without a WtrName are dropped."""
    row = PAPFBCTroutStockedIngest._normalize_feature({"SecDate": 1711929600000}, 2024)
    assert row is None


def test_pfbc_normalize_returns_row_tuple() -> None:
    attrs = {"WtrName": "Pine Creek", "SecDate": 1711929600000}
    row = PAPFBCTroutStockedIngest._normalize_feature(attrs, 2024)
    assert row is not None
    assert row[1] == "PA"
    assert row[2] == "2024-04-01"
    assert row[5] == "Pine Creek"
    assert row[6] == "PA_PFBC_TROUT_STOCKED"


# ----- NHDPlus HR --------------------------------------------------------


def test_nhdplus_state_fips_known_states() -> None:
    assert _state_fips("PA") == "42"
    assert _state_fips("VA") == "51"
    assert _state_fips("ID") == "16"
    assert _state_fips("XX") is None


def test_nhdplus_default_huc8_per_v0_state() -> None:
    """Every v0 launch state must have a default smoke-test HUC8."""
    for state in ("PA", "VA", "ID"):
        assert state in DEFAULT_HUC8_SAMPLE
        assert len(DEFAULT_HUC8_SAMPLE[state]) == 8


def test_nhdplus_polyline_to_wkt_simple() -> None:
    geom = {"paths": [[[-77.0, 41.0], [-77.1, 41.05], [-77.2, 41.1]]]}
    wkt = _polyline_to_wkt(geom)
    assert wkt is not None
    assert wkt.startswith("MULTILINESTRING")
    assert "-77.0 41.0" in wkt


def test_nhdplus_polyline_to_wkt_handles_empty() -> None:
    assert _polyline_to_wkt(None) is None
    assert _polyline_to_wkt({"paths": []}) is None
    assert _polyline_to_wkt({"paths": [[[1.0, 2.0]]]}) is None  # single point


def test_nhdplus_polyline_handles_3d_and_4d_coords() -> None:
    """Esri polylines may carry M or Z dimension - keep only (x, y)."""
    # 3D (Z): [x, y, z]
    geom_3d = {"paths": [[[-77.0, 41.0, 100.0], [-77.1, 41.05, 110.0]]]}
    wkt = _polyline_to_wkt(geom_3d)
    assert wkt is not None
    assert "-77.0 41.0" in wkt
    assert "100" not in wkt  # Z stripped
    # 4D (Z + M): [x, y, z, m]
    geom_4d = {"paths": [[[-77.0, 41.0, 100.0, 0.0], [-77.1, 41.05, 110.0, 1.0]]]}
    wkt = _polyline_to_wkt(geom_4d)
    assert wkt is not None
    assert "-77.0 41.0" in wkt


def test_nhdplus_normalize_feature_extracts_vaas(tmp_path) -> None:
    """Spot-check: nhdplusid -> comid, reachcode -> huc8/10/12, arbolatesu kept."""
    feat = {
        "attributes": {
            "nhdplusid": 12345678,
            "reachcode": "02050206000123",
            "gnis_name": "West Branch Susquehanna River",
            "streamorde": 5,
            "totdasqkm": 250.5,
            "arbolatesu": 999.9,  # NOTE: lowercase; field name verified live (pass-2 fix)
        },
        "geometry": {"paths": [[[-77.0, 41.0], [-77.1, 41.05]]]},
    }
    row = NHDPlusHRIngest._normalize_feature(feat, "PA")
    assert row is not None
    assert row[0] == 12345678         # comid
    assert row[1] == "02050206000123" # reachcode (full 14-digit)
    assert row[2] == "West Branch Susquehanna River"
    assert row[3] == "42"             # PA FIPS
    assert row[5] == "02050206"       # huc8
    assert row[6] == "0205020600"     # huc10
    assert row[7] == "020502060001"   # huc12
    assert row[8] == 5                # stream order
    assert row[9] == 250.5            # drainage_km2
    assert row[10].startswith("MULTILINESTRING")
