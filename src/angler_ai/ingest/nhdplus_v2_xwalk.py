"""NHDPlus V2.1 -> HR crosswalk via shared REACHCODE.

DR-1.2: USGS BRT v2.0 predictions are keyed by NHDPlusV2.1 COMID, but our
reaches table holds NHDPlus HR COMIDs. Both versions use the same 14-digit
NHDReachCode, so we build the crosswalk by joining on REACHCODE.

One V2 reach can split into multiple HR features (HR is higher resolution).
The crosswalk table xwalk_v2_to_hr supports 1:N: one V2 COMID maps to all the
HR COMIDs that share its REACHCODE.

Data source:
    EPA watersgeo NHDPlus_NP21 / NHDSnapshot_NP21 / Flowlines layer 0
    https://watersgeo.epa.gov/arcgis/rest/services/NHDPlus_NP21/NHDSnapshot_NP21/MapServer/0
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from angler_ai.features.store import FeatureStore
from angler_ai.ingest._arcgis import query_layer
from angler_ai.ingest._bulk import bulk_insert
from angler_ai.ingest.base import IngestionModule, SourceMetadata
from angler_ai.ingest.nhdplus_hr import DEFAULT_HUC8_SAMPLE, STATE_HUC4_PREFIXES

log = logging.getLogger(__name__)

V2_FLOWLINES_LAYER = (
    "https://watersgeo.epa.gov/arcgis/rest/services/NHDPlus_NP21/"
    "NHDSnapshot_NP21/MapServer/0"
)


class NHDPlusV2HRXwalkIngest(IngestionModule):
    """Pull NHDPlus V2 Flowlines for a state subset and build the V2 -> HR
    crosswalk by joining on REACHCODE against the local reaches table."""

    metadata = SourceMetadata(
        source_id="NHDPlus_V2_XWALK",
        display_name="NHDPlus V2.1 to HR Crosswalk via shared REACHCODE",
        source_url=V2_FLOWLINES_LAYER,
        license="public-domain",
        refresh_cadence="annual or per new vintage",
        discovery_pattern=(
            "Filter NHDPlus V2 Flowlines by REACHCODE HUC8 prefix matching the "
            "HR reaches loaded in M2; join on shared REACHCODE."
        ),
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
        """Build xwalk_v2_to_hr for the state. Returns matched-row count."""
        if huc8 is None and not full_state:
            huc8 = DEFAULT_HUC8_SAMPLE.get(state)
        if huc8 is not None:
            where_clauses = [f"REACHCODE LIKE '{huc8}%'"]
        else:
            prefixes = STATE_HUC4_PREFIXES.get(state, ())
            if not prefixes:
                log.warning("V2 xwalk: no HUC4 prefixes for state %s", state)
                return 0
            where_clauses = [f"REACHCODE LIKE '{p}%'" for p in prefixes]
        where = " OR ".join(where_clauses)
        log.info("NHDPlus V2 query (%s): %s", state, where[:120])

        # Pull V2 flowlines: (COMID, REACHCODE)
        v2_rows: list[tuple[int, str]] = []
        for feat in query_layer(
            V2_FLOWLINES_LAYER,
            where=where,
            out_fields="COMID,REACHCODE",
            return_geometry=False,
            max_record_count=1000,
            max_pages=20,
            timeout_s=120.0,
        ):
            attrs = feat.get("attributes", {})
            comid_v2 = attrs.get("COMID")
            reachcode = (attrs.get("REACHCODE") or "").strip()
            if comid_v2 is None or not reachcode:
                continue
            try:
                v2_rows.append((int(comid_v2), reachcode))
            except (TypeError, ValueError):
                continue
        log.info("NHDPlus V2 flowlines fetched: %d", len(v2_rows))

        if not v2_rows:
            return 0

        # Join V2 -> HR on exact REACHCODE match. Both NHDPlus versions register
        # against the same 14-digit NHDReachCode. HR can split a V2 reach into
        # multiple high-resolution features, so the relationship is 1:N.
        now = datetime.now(timezone.utc).isoformat()
        with store.transaction() as conn:
            import pyarrow as pa
            v2_table = pa.table({
                "comid_v2": [r[0] for r in v2_rows],
                "reachcode": [r[1] for r in v2_rows],
            })
            conn.register("_v2_stage", v2_table)
            # Additive across calls so multiple HUC8 ingests accumulate.
            # ON CONFLICT keeps the FIRST mapping for any (comid_v2, comid_hr).
            # Tier 1: exact reachcode match (confidence 1.0). Major reaches.
            conn.execute(
                """
                INSERT INTO xwalk_v2_to_hr (comid_v2, comid_hr, confidence, method, ingested_at)
                SELECT DISTINCT v.comid_v2, r.comid, 1.0, 'reachcode_exact', ?
                FROM _v2_stage v
                JOIN reaches r ON v.reachcode = r.reachcode
                ON CONFLICT (comid_v2, comid_hr) DO NOTHING
                """,
                [now],
            )
            exact = conn.execute(
                "SELECT COUNT(*) FROM xwalk_v2_to_hr WHERE method = 'reachcode_exact'"
            ).fetchone()[0]
            # Tier 2: HUC10 proximity (confidence 0.5). For HR reaches finer
            # than V2 resolution that have no exact match, fall back to a V2
            # reach in the same HUC10 with the highest stream order proxy
            # (use the V2 reach with the LOWEST comid as a deterministic
            # tiebreaker - true nearest-line spatial join is a future
            # enhancement requiring geometry on V2).
            conn.execute(
                """
                INSERT INTO xwalk_v2_to_hr (comid_v2, comid_hr, confidence, method, ingested_at)
                SELECT DISTINCT ON (r.comid)
                    v.comid_v2, r.comid, 0.5, 'huc10_proximity', ?
                FROM _v2_stage v
                JOIN reaches r
                  ON SUBSTRING(v.reachcode, 1, 10) = r.huc10
                WHERE NOT EXISTS (
                    SELECT 1 FROM xwalk_v2_to_hr x
                    WHERE x.comid_hr = r.comid AND x.method = 'reachcode_exact'
                )
                ORDER BY r.comid, v.comid_v2
                ON CONFLICT (comid_v2, comid_hr) DO NOTHING
                """,
                [now],
            )
            count = conn.execute(
                "SELECT COUNT(*) FROM xwalk_v2_to_hr"
            ).fetchone()[0]
            proximity = count - exact
            log.info(
                "xwalk_v2_to_hr (%s): %d total (%d reachcode_exact, %d huc10_proximity)",
                state, count, exact, proximity,
            )
            conn.unregister("_v2_stage")
        return int(count)

    def last_ingested_at(self, store: FeatureStore) -> str | None:
        row = store.connect().execute(
            "SELECT MAX(ingested_at) FROM xwalk_v2_to_hr",
        ).fetchone()
        return row[0] if row else None
