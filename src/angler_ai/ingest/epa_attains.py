"""EPA ATTAINS impaired-waters ingestion via the public ArcGIS REST geospatial
service. FR-4.4.

Layer 6 is the Assessments-by-Catchment table joining each impaired waterbody
to its NHDPlus catchment via `nhdplusid` (= NHDPlus HR COMID for HR data, or
NHDPlusV2.1 COMID for V2 data; EPA uses HR in current cycles). We filter to
the requested state with a `where state = 'PA'` clause and upsert into
`attains_status`. Idempotent per (source, state).

API key is "transitioning to required" per pass-2 peer-review correction. The
public MapServer at gispub.epa.gov continues to work without a key as of M2
implementation; we surface the API-key field in metadata for future enforcement.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from angler_ai.features.store import FeatureStore
from angler_ai.ingest._arcgis import query_layer
from angler_ai.ingest._bulk import bulk_insert
from angler_ai.ingest.base import IngestionModule, SourceMetadata

log = logging.getLogger(__name__)

ATTAINS_MAPSERVER = (
    "https://gispub.epa.gov/arcgis/rest/services/OW/ATTAINS_Assessment/MapServer"
)
ASSESSMENTS_BY_CATCHMENT_LAYER = 6


class EPAATTAINSIngest(IngestionModule):
    metadata = SourceMetadata(
        source_id="EPA_ATTAINS",
        display_name="EPA ATTAINS Assessment (Assessments by Catchment)",
        source_url=f"{ATTAINS_MAPSERVER}/{ASSESSMENTS_BY_CATCHMENT_LAYER}",
        license="public-domain",
        refresh_cadence="twice per state assessment cycle (event-driven)",
        discovery_pattern=(
            "Query MapServer layer 6 with where state = 'XX'; "
            "joins to NHDPlus HR COMID via nhdplusid."
        ),
        terms_of_use_url="https://www.epa.gov/waterdata",
    )

    def ingest(self, store: FeatureStore, *, state: str = "PA", **_: object) -> int:
        layer_url = f"{ATTAINS_MAPSERVER}/{ASSESSMENTS_BY_CATCHMENT_LAYER}"
        latest_cycle = _latest_cycle_for_state(layer_url, state)
        where = f"state = '{state}'"
        if latest_cycle is not None:
            where = f"{where} AND reportingcycle = {latest_cycle}"
        rows: list[tuple] = []
        for feat in query_layer(
            layer_url,
            where=where,
            out_fields="nhdplusid,state,reportingcycle,overallstatus,ircategory,assessmentunitidentifier,waterbodyreportlink",
            return_geometry=False,
            max_record_count=10000,
            max_pages=2,
        ):
            attrs = feat.get("attributes", {})
            nhdplusid = attrs.get("nhdplusid")
            if nhdplusid is None:
                continue
            try:
                comid = int(nhdplusid)
            except (TypeError, ValueError):
                continue
            rows.append((
                comid,
                int(attrs.get("reportingcycle") or 0),
                attrs.get("overallstatus"),
                attrs.get("ircategory"),
                attrs.get("waterbodyreportlink"),
            ))

        now = datetime.now(timezone.utc).isoformat()
        with store.transaction() as conn:
            conn.execute(
                "DELETE FROM attains_status WHERE source = ?",
                ["EPA_ATTAINS"],
            )
            if rows:
                bulk_insert(
                    conn, "attains_status", rows,
                    columns=("comid", "cycle_year", "status", "parameter", "state_305b_url"),
                    extra_literals={"source": "EPA_ATTAINS", "ingested_at": now},
                )
        log.info("ATTAINS ingest (%s): %d rows", state, len(rows))
        return len(rows)

    def last_ingested_at(self, store: FeatureStore) -> str | None:
        row = store.connect().execute(
            "SELECT MAX(ingested_at) FROM attains_status WHERE source = ?",
            ["EPA_ATTAINS"],
        ).fetchone()
        return row[0] if row else None


def _latest_cycle_for_state(layer_url: str, state: str) -> int | None:
    """Return the most recent reportingcycle integer for the given state, or
    None if the discovery query fails.

    EPA ATTAINS layer 6 is multi-cycle (decades of assessments). For v0 we
    surface the most recent cycle only; older cycles can be backfilled by
    re-running with a wider where clause when needed.
    """
    import httpx

    try:
        with httpx.Client(timeout=30) as client:
            r = client.get(
                f"{layer_url}/query",
                params={
                    "where": f"state = '{state}'",
                    "outFields": "reportingcycle",
                    "returnDistinctValues": "true",
                    "orderByFields": "reportingcycle DESC",
                    "resultRecordCount": 1,
                    "f": "json",
                },
            )
            r.raise_for_status()
            feats = r.json().get("features", [])
            if not feats:
                return None
            cycle = feats[0].get("attributes", {}).get("reportingcycle")
            return int(cycle) if cycle is not None else None
    except Exception:
        return None
