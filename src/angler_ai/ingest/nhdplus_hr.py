"""NHDPlus HR hydrography ingestion via USGS hydro.nationalmap.gov ArcGIS REST.
FR-4.2.

For v0 PA we pull NetworkNHDFlowline (layer 3) filtered to the HUC8s that
intersect Pennsylvania. The full state has roughly 250k reaches; the smoke
test scopes to a single HUC8 by default to keep the run under a minute.

VAA field name verified: the canonical NHDPlus column is `arbolatesu`
(lowercase, 10-char DBF truncation), NOT `arbolatesum`. Pass-2 peer-review fix.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from angler_ai.features.store import FeatureStore
from angler_ai.ingest._arcgis import query_layer
from angler_ai.ingest._bulk import bulk_insert
from angler_ai.ingest.base import IngestionModule, SourceMetadata

log = logging.getLogger(__name__)

NHDPLUS_HR_MAPSERVER = (
    "https://hydro.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/MapServer"
)
NETWORK_FLOWLINE_LAYER = 3


# HUC4 prefixes that cover each launch state. Used in the `where reachcode
# LIKE '04xx%'` filter. Includes some neighboring-state reaches; we tag state
# at the upsert step via centroid lat/lon.
STATE_HUC4_PREFIXES: dict[str, tuple[str, ...]] = {
    "PA": ("0205", "0204", "0207", "0501", "0502"),
    "VA": ("0207", "0208", "0301", "0508"),
    "ID": ("1701", "1702", "1704", "1705", "1706", "1707"),
    "MT": ("1002", "1003", "1004", "1005", "1006", "1007", "1009"),
    "WY": ("1002", "1004", "1007", "1008", "1009", "1018"),
}


# Smoke-test default: a single HUC8 around Lycoming County PA (West Branch
# Susquehanna) - the design doc's running example. Manageable size for M2 tests.
DEFAULT_HUC8_SAMPLE: dict[str, str] = {
    "PA": "02050206",  # Lower West Branch Susquehanna (Lycoming, Union)
    "VA": "02080201",  # Lower Rappahannock
    "ID": "17010204",  # Lower Kootenai
}


# VAA fields pulled from layer 3. Names match the ArcGIS service (already
# lowercase), so the lossless mapping to the DBF column `arbolatesu` is fine.
_FLOWLINE_OUT_FIELDS = ",".join((
    "nhdplusid", "reachcode", "gnis_id", "gnis_name", "lengthkm",
    "ftype", "streamorde", "totdasqkm",
    "hydroseq", "levelpathi", "uphydroseq", "dnhydroseq",
    "arbolatesu", "frommeas", "tomeas", "pathlength", "terminalpa",
))


class NHDPlusHRIngest(IngestionModule):
    metadata = SourceMetadata(
        source_id="NHDPlus_HR",
        display_name="NHDPlus High Resolution NetworkNHDFlowline",
        source_url=f"{NHDPLUS_HR_MAPSERVER}/{NETWORK_FLOWLINE_LAYER}",
        license="public-domain",
        refresh_cadence="annual or per new USGS vintage",
        discovery_pattern=(
            "Filter NetworkNHDFlowline by reachcode HUC8 prefix; default v0 "
            "smoke-test scope is one HUC8 per state."
        ),
        terms_of_use_url="https://www.usgs.gov/national-hydrography",
    )

    def ingest(
        self,
        store: FeatureStore,
        *,
        state: str = "PA",
        huc8: str | None = None,
        full_state: bool = False,
        **_: object,
    ) -> int:
        """Pull NHDFlowline rows for the requested geographic scope.

        Args:
            store: feature store.
            state: two-letter state code.
            huc8: explicit HUC8 to scope; takes precedence over `full_state`.
            full_state: if True, query every HUC4 prefix for the state. Slow.

        Returns:
            Row count inserted into `reaches`.
        """
        if huc8 is None and not full_state:
            huc8 = DEFAULT_HUC8_SAMPLE.get(state)
        if huc8 is not None:
            where_clauses = [f"reachcode LIKE '{huc8}%'"]
        else:
            prefixes = STATE_HUC4_PREFIXES.get(state, ())
            if not prefixes:
                log.warning("NHDPlus HR: no HUC4 prefixes known for state %s", state)
                return 0
            where_clauses = [f"reachcode LIKE '{p}%'" for p in prefixes]
        where = " OR ".join(where_clauses)
        log.info("NHDPlus HR query (%s): %s", state, where[:160])

        layer_url = f"{NHDPLUS_HR_MAPSERVER}/{NETWORK_FLOWLINE_LAYER}"
        rows: list[tuple] = []
        import httpx as _httpx

        skipped_no_geom = 0
        try:
            for feat in query_layer(
                layer_url,
                where=where,
                out_fields=_FLOWLINE_OUT_FIELDS,
                return_geometry=True,
                max_record_count=2000,
                max_pages=2,
                timeout_s=180.0,
            ):
                row = self._normalize_feature(feat, state)
                if row is None:
                    continue
                # Drop rows with no geometry: ST_GeomFromText rejects NULL.
                if row[9] is None:
                    skipped_no_geom += 1
                    continue
                rows.append(row)
        except (_httpx.ReadTimeout, _httpx.ConnectTimeout) as exc:
            log.warning(
                "NHDPlus HR upstream timeout after %d rows; persisting what we have: %s",
                len(rows), exc,
            )
        if skipped_no_geom:
            log.info("NHDPlus HR: dropped %d rows missing geometry", skipped_no_geom)

        now = datetime.now(timezone.utc).isoformat()
        with store.transaction() as conn:
            if huc8:
                conn.execute(
                    "DELETE FROM reaches WHERE source = 'NHDPlus_HR' AND huc8 = ?",
                    [huc8],
                )
            else:
                conn.execute(
                    "DELETE FROM reaches WHERE source = 'NHDPlus_HR' AND state_fips = ?",
                    [_state_fips(state)],
                )
            if rows:
                bulk_insert(
                    conn, "reaches", rows,
                    columns=(
                        "comid", "reachcode", "gnis_name", "state_fips", "county_fips",
                        "huc8", "huc10", "huc12", "stream_order", "drainage_area_km2",
                        "geometry",
                    ),
                    geometry_columns=("geometry",),
                    extra_literals={"source": "NHDPlus_HR", "ingested_at": now},
                )
        log.info("NHDPlus HR ingest: %d reaches inserted", len(rows))
        return len(rows)

    @staticmethod
    def _normalize_feature(feat: dict, state: str) -> tuple | None:
        attrs = feat.get("attributes", {})
        nhdplusid = attrs.get("nhdplusid")
        if nhdplusid is None:
            return None
        try:
            comid = int(nhdplusid)
        except (TypeError, ValueError):
            return None
        reachcode = (attrs.get("reachcode") or "").strip() or None
        huc8 = reachcode[:8] if reachcode and len(reachcode) >= 8 else None
        huc10 = reachcode[:10] if reachcode and len(reachcode) >= 10 else None
        huc12 = reachcode[:12] if reachcode and len(reachcode) >= 12 else None
        gnis_name = attrs.get("gnis_name")
        # Reach geometry: convert Esri polyline to WKT.
        geom_wkt = _polyline_to_wkt(feat.get("geometry"))
        stream_order = attrs.get("streamorde")
        drainage_km2 = attrs.get("totdasqkm")
        return (
            comid,
            reachcode,
            gnis_name,
            _state_fips(state),
            None,  # county_fips - derived in a later pass via spatial join
            huc8, huc10, huc12,
            int(stream_order) if stream_order is not None else None,
            float(drainage_km2) if drainage_km2 is not None else None,
            geom_wkt,
        )

    def last_ingested_at(self, store: FeatureStore) -> str | None:
        row = store.connect().execute(
            "SELECT MAX(ingested_at) FROM reaches WHERE source = 'NHDPlus_HR'",
        ).fetchone()
        return row[0] if row else None


# US Census state FIPS codes. v0 launch states + western-river-system support.
_STATE_FIPS = {
    "PA": "42", "VA": "51", "ID": "16",
    # Western trout-river demo states (added 2026-06-17 to support
    # Henrys Fork / Madison / Big Hole / Jefferson / Firehole queries).
    "MT": "30", "WY": "56",
    # Gunnison basin demo (added 2026-06-20 to support Uncompahgre / Cimarron /
    # Taylor / Upper+Lower Gunnison queries).
    "CO": "08",
}


def _state_fips(state: str) -> str | None:
    return _STATE_FIPS.get(state)


def _polyline_to_wkt(geom: dict | None) -> str | None:
    """Convert an Esri polyline geometry to WKT MULTILINESTRING.

    Tolerates 2D, 3D (Z), and 4D (Z+M) coordinate tuples by keeping only
    (x, y). Returns None on missing or unusable geometry.
    """
    if not geom or "paths" not in geom:
        return None
    paths = geom["paths"]
    if not paths:
        return None
    parts: list[str] = []
    for path in paths:
        if not path or len(path) < 2:
            continue
        try:
            coords = ", ".join(f"{pt[0]} {pt[1]}" for pt in path if len(pt) >= 2)
        except (TypeError, IndexError):
            continue
        if not coords:
            continue
        parts.append(f"({coords})")
    if not parts:
        return None
    return "MULTILINESTRING(" + ", ".join(parts) + ")"
