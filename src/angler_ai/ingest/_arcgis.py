"""Shared ArcGIS REST helpers. Used by EPA ATTAINS, NHDPlus HR, PA PFBC, and any
future ArcGIS-published source. Handles maxRecordCount pagination via resultOffset.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

import httpx

log = logging.getLogger(__name__)


def query_layer(
    base_url: str,
    *,
    where: str = "1=1",
    out_fields: str = "*",
    geometry: dict[str, Any] | None = None,
    geometry_type: str | None = None,
    spatial_rel: str = "esriSpatialRelIntersects",
    out_sr: int = 4326,
    return_geometry: bool = True,
    max_record_count: int = 1000,
    max_pages: int | None = None,
    timeout_s: float = 60.0,
    client: httpx.Client | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield features from an ArcGIS REST FeatureLayer / MapServer layer.

    Pages via `resultOffset` until the layer reports no more records.

    Args:
        base_url: e.g. "https://gispub.epa.gov/.../MapServer/6"
        where: ArcGIS SQL where clause. Default "1=1" returns all.
        out_fields: comma-separated field list, or "*".
        geometry: optional Esri geometry dict for spatial filter.
        geometry_type: 'esriGeometryPoint' / 'esriGeometryEnvelope' / etc.
        spatial_rel: relationship verb.
        out_sr: output spatial reference. 4326 = WGS84 lat/lon.
        return_geometry: include the Shape in each feature.
        max_record_count: page size; ArcGIS will clamp to its server limit.
        timeout_s: per-request timeout.
        client: optional reusable httpx.Client.

    Yields:
        feature dicts with `attributes` (and `geometry` if return_geometry).
    """
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=timeout_s)
    try:
        offset = 0
        page = 0
        while True:
            params: dict[str, Any] = {
                "where": where,
                "outFields": out_fields,
                "returnGeometry": "true" if return_geometry else "false",
                "outSR": out_sr,
                "resultOffset": offset,
                "resultRecordCount": max_record_count,
                "f": "json",
            }
            if geometry is not None:
                params["geometry"] = httpx.QueryParams({"g": str(geometry)}).get("g") or ""
                # Easier: pass geometry as JSON-encoded string.
                import json as _json
                params["geometry"] = _json.dumps(geometry)
                params["geometryType"] = geometry_type or "esriGeometryEnvelope"
                params["spatialRel"] = spatial_rel
                params["inSR"] = 4326
            r = client.get(f"{base_url}/query", params=params)
            r.raise_for_status()
            data = r.json()
            features = data.get("features", [])
            if not features:
                return
            for feat in features:
                yield feat
            page += 1
            log.debug("arcgis page %d offset=%d -> %d features", page, offset, len(features))
            if not data.get("exceededTransferLimit") and len(features) < max_record_count:
                return
            if max_pages is not None and page >= max_pages:
                log.info(
                    "arcgis query reached max_pages=%d at offset=%d; stopping (more rows available upstream)",
                    max_pages, offset + len(features),
                )
                return
            offset += len(features)
    finally:
        if own_client:
            client.close()


def list_services(server_url: str, *, timeout_s: float = 30.0) -> list[dict[str, Any]]:
    """List MapServer / FeatureServer services under an ArcGIS REST root."""
    with httpx.Client(timeout=timeout_s) as client:
        r = client.get(server_url, params={"f": "json"})
        r.raise_for_status()
        return r.json().get("services", [])
