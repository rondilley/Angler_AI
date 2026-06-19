"""GeoJSON map export for species probability over a geography.

Emits a FeatureCollection where each Feature is a reach (LineString or
MultiLineString) with properties carrying the CalibratedProbability fields
intact (point, lower, upper, basis, beta) per FR-9.3 and FR-6.4.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from angler_ai.prediction.species_priors import SpeciesPrior


@dataclass(slots=True)
class GeoJSONExport:
    """Holds the assembled FeatureCollection plus per-export metadata."""

    feature_count: int
    raw_geojson: dict


def build_feature_collection(
    rows: list[tuple[SpeciesPrior, str]],
    *,
    species_scientific: str,
    species_common: str | None,
    date: str,
    extra_metadata: dict | None = None,
    temperatures: dict | None = None,
) -> GeoJSONExport:
    """Construct a GeoJSON FeatureCollection.

    Args:
        rows: list of (SpeciesPrior, geometry_wkt) from species_priors_for_geometry.
        species_scientific: the queried species (Latin binomial).
        species_common: common name if known.
        date: ISO date of the prediction window.
        extra_metadata: optional top-level FeatureCollection metadata.

    Returns:
        GeoJSONExport. The contained `raw_geojson` is JSON-serializable.
    """
    features: list[dict] = []
    for sp, wkt in rows:
        geom = _wkt_to_geojson_geometry(wkt) if wkt else None
        if geom is None:
            continue
        cp = sp.probability
        # Temperature lookup: honest source even when not modeled.
        rt = temperatures.get(sp.comid) if temperatures else None
        properties = {
            "comid": sp.comid,
            "species_scientific": sp.species,
            "species_common": sp.common_name,
            "probability": cp.point,
            "lower": cp.lower,
            "upper": cp.upper,
            "interval_confidence": cp.interval_confidence,
            "raw_probability": cp.raw_point,
            "hyperstability_beta_applied": cp.hyperstability_beta_applied,
            "basis": {
                "cpue_derived_weight": cp.basis.cpue_derived_weight,
                "fisheries_independent_weight": cp.basis.fisheries_independent_weight,
                "sources": list(cp.basis.sources),
            },
            "v2_join_method": sp.v2_join_method,
            "date": date,
            # Temperature surface (M5): real value + source, or honest 'not_modeled'.
            "temperature_c": rt.temperature_c if rt else None,
            "temperature_uncertainty_c": rt.uncertainty_c if rt else None,
            "temperature_source": rt.source if rt else "not_modeled",
            "temperature_date": rt.date if rt else None,
        }
        feature = {
            "type": "Feature",
            "geometry": geom,
            "properties": properties,
        }
        features.append(feature)

    metadata = {
        "species_scientific": species_scientific,
        "species_common": species_common,
        "date": date,
        "model": {
            "id": "USGS_BRT_V2.0",
            "version": "USGS_BRT_V2.0",
            "doi": "https://doi.org/10.5066/P1UV25FW",
            "citation": (
                "Yu, S.L., Cooper, A.R., Ross, J., McKerrow, A.J., "
                "Wieferich, D.J., Infante, D.M. 2023. USGS SIR 2023-5088."
            ),
        },
        "calibration": {
            "hyperstability_beta": 0.23,
            "hyperstability_source": "Charbonneau et al. 2025 TAFS 154(4):339",
            "note": (
                "v0 wraps USGS BRT priors in hyperstability-aware "
                "CalibratedProbability. v1 will swap in SSN2 ssn_glm(binomial) "
                "trained against observed catch data."
            ),
        },
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    fc = {
        "type": "FeatureCollection",
        "metadata": metadata,
        "features": features,
    }
    return GeoJSONExport(feature_count=len(features), raw_geojson=fc)


_PATH_RE = re.compile(r"\(([^()]+)\)")


def _wkt_to_geojson_geometry(wkt: str) -> dict | None:
    """Convert MULTILINESTRING WKT to GeoJSON geometry.

    We do this by hand to avoid a shapely dependency on the export path.
    Tolerates surrounding whitespace and 2D coordinates.
    """
    s = wkt.strip()
    upper = s.upper()
    if upper.startswith("MULTILINESTRING"):
        # Strip prefix and outer parens
        body = s[len("MULTILINESTRING"):].strip()
        if body.startswith("("):
            body = body[1:]
        if body.endswith(")"):
            body = body[:-1]
        # Split into individual linestrings on '), ('
        parts = re.split(r"\),\s*\(", body)
        lines: list[list[list[float]]] = []
        for part in parts:
            part = part.strip().strip("()")
            coords = _parse_coords(part)
            if coords:
                lines.append(coords)
        if not lines:
            return None
        if len(lines) == 1:
            return {"type": "LineString", "coordinates": lines[0]}
        return {"type": "MultiLineString", "coordinates": lines}
    if upper.startswith("LINESTRING"):
        body = s[len("LINESTRING"):].strip().strip("()")
        coords = _parse_coords(body)
        if not coords:
            return None
        return {"type": "LineString", "coordinates": coords}
    return None


def _parse_coords(body: str) -> list[list[float]]:
    coords: list[list[float]] = []
    for pt in body.split(","):
        pt = pt.strip()
        if not pt:
            continue
        parts = pt.split()
        if len(parts) < 2:
            continue
        try:
            coords.append([float(parts[0]), float(parts[1])])
        except ValueError:
            continue
    return coords


def write_geojson(export: GeoJSONExport, out_path) -> None:
    """Write the FeatureCollection to disk as pretty JSON."""
    out_path = str(out_path)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(export.raw_geojson, fh, indent=2, sort_keys=False)
