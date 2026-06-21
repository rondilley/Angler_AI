"""USGS NAS (Nonindigenous Aquatic Species) ingestion via the v2 occurrence API.

FR-4.x extension. Provides a presence-only fallback prior for non-native
species in HUC8s where USGS BRT v2.0 has no row (BRT v2.0 only models
native distributions, so all non-natives outside their native range get
zero priors).

NAS gives us per-record introduction reports (species + HUC + year + status
+ locality + reference). We collapse to a per-(huc8, scientific_name) summary
suitable for a binary fallback decision in species_priors.

Establishment filter: only contribute records whose `status` is in
{'established', 'stocked', 'collected', 'regularly observed'} (case-insensitive
on the NAS lowercased values). Drop 'eradicated', 'failed', 'extirpated'.
Rationale: a 1986 eradicated record is materially different evidence from
a 2024 established record - we do not want to hand 0.35 priors to populations
known to be gone.

Federal public-domain API; no API key required.

API docs: https://nas.er.usgs.gov/api/documentation.aspx
Base URL: https://nas.er.usgs.gov/api/v2
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from angler_ai.features.store import FeatureStore
from angler_ai.ingest._bulk import bulk_insert
from angler_ai.ingest.base import IngestionModule, SourceMetadata

log = logging.getLogger(__name__)

NAS_BASE_URL = "https://nas.er.usgs.gov/api/v2"
SEARCH_ENDPOINT = f"{NAS_BASE_URL}/occurrence/search"

# Only contribute records whose status indicates current or repeated presence.
# NAS status vocabulary is lowercase free-text; we lowercase + strip on
# comparison. See ingest filter test below.
_PRESENCE_STATUSES: frozenset[str] = frozenset({
    "established",
    "stocked",
    "collected",
    "regularly observed",
})

# Fishes-only at v0. NAS includes amphibians, reptiles, mussels, plants etc.
# The forecast pipeline scores fish.
_TARGET_GROUP = "Fishes"


class USGSNASIngest(IngestionModule):
    metadata = SourceMetadata(
        source_id="USGS_NAS_V1.0",
        display_name="USGS Nonindigenous Aquatic Species occurrence summary",
        source_url=NAS_BASE_URL,
        license="public-domain",
        refresh_cadence="continuous (NAS curates ongoing)",
        discovery_pattern=(
            "Query /occurrence/search per HUC8; filter to fish records with "
            "presence-indicating status; aggregate per (huc8, scientific_name)."
        ),
        terms_of_use_url="https://nas.er.usgs.gov/api/documentation.aspx",
    )

    def ingest(
        self,
        store: FeatureStore,
        *,
        state: str = "CO",
        huc8: str | None = None,
        timeout_s: float = 60.0,
        **_: object,
    ) -> int:
        """Populate `nas_occurrences` for one HUC8.

        Args:
            store: feature store.
            state: state code (informational; the actual scope is huc8).
            huc8: 8-digit HUC. Required - NAS is queried per HUC8.
            timeout_s: per-request HTTP timeout.

        Returns:
            Row count inserted into `nas_occurrences`.
        """
        if huc8 is None:
            log.warning("USGS NAS: --huc8 is required; nothing to ingest")
            return 0

        records = self._fetch_huc8(huc8, timeout_s=timeout_s)
        if not records:
            log.info("USGS NAS: no records returned for HUC8 %s", huc8)
            return 0

        # Filter to fish + presence-indicating status.
        kept: list[dict] = []
        for rec in records:
            if rec.get("group") != _TARGET_GROUP:
                continue
            status = str(rec.get("status") or "").strip().lower()
            if status not in _PRESENCE_STATUSES:
                continue
            sci = rec.get("scientificName")
            if not sci:
                continue
            kept.append(rec)
        if not kept:
            log.info(
                "USGS NAS: %d records, 0 fish with presence-indicating status "
                "for HUC8 %s",
                len(records), huc8,
            )
            return 0

        # Aggregate per (huc8, scientific_name).
        # Most-recent record wins for `status` and `common_name`.
        agg: dict[str, dict] = {}
        for rec in kept:
            sci = rec["scientificName"]
            year = rec.get("year")
            try:
                year_int = int(year) if year not in (None, "", "null") else None
            except (TypeError, ValueError):
                year_int = None
            entry = agg.setdefault(sci, {
                "scientific_name": sci,
                "common_name": rec.get("commonName"),
                "status": str(rec.get("status") or "").strip().lower(),
                "year_first": year_int,
                "year_last": year_int,
                "n_records": 0,
                "latest_year_seen": year_int,
            })
            entry["n_records"] += 1
            if year_int is not None:
                if entry["year_first"] is None or year_int < entry["year_first"]:
                    entry["year_first"] = year_int
                if entry["year_last"] is None or year_int > entry["year_last"]:
                    entry["year_last"] = year_int
                if entry["latest_year_seen"] is None or year_int >= entry["latest_year_seen"]:
                    entry["latest_year_seen"] = year_int
                    entry["status"] = str(rec.get("status") or "").strip().lower()
                    entry["common_name"] = rec.get("commonName") or entry["common_name"]

        rows: list[tuple] = []
        for sci, e in agg.items():
            rows.append((
                huc8,
                sci,
                e["common_name"],
                e["status"],
                e["year_first"],
                e["year_last"],
                e["n_records"],
            ))

        now = datetime.now(timezone.utc).isoformat()
        with store.transaction() as conn:
            conn.execute(
                "DELETE FROM nas_occurrences WHERE huc8 = ? AND source = ?",
                [huc8, self.metadata.source_id],
            )
            if rows:
                bulk_insert(
                    conn, "nas_occurrences", rows,
                    columns=(
                        "huc8", "scientific_name", "common_name", "status",
                        "year_first_observed", "year_last_observed", "n_records",
                    ),
                    extra_literals={
                        "source": self.metadata.source_id,
                        "ingested_at": now,
                    },
                )
        log.info(
            "USGS NAS: HUC8 %s: %d total records -> %d fish-presence species summarized",
            huc8, len(records), len(rows),
        )
        return len(rows)

    def _fetch_huc8(self, huc8: str, *, timeout_s: float) -> list[dict]:
        """Pull all NAS occurrence records for one HUC8.

        The NAS v2 API returns the full result set in one response for
        HUC8-scoped queries (counts are typically <100). We do not paginate.
        """
        params = {"huc8": huc8}
        try:
            with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
                resp = client.get(SEARCH_ENDPOINT, params=params)
            resp.raise_for_status()
            payload = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("USGS NAS: HUC8 %s fetch failed: %s", huc8, exc)
            return []
        results = payload.get("results") or []
        if not isinstance(results, list):
            log.warning("USGS NAS: HUC8 %s unexpected payload shape", huc8)
            return []
        return results

    def last_ingested_at(self, store: FeatureStore) -> str | None:
        row = store.connect().execute(
            "SELECT MAX(ingested_at) FROM nas_occurrences WHERE source = ?",
            [self.metadata.source_id],
        ).fetchone()
        return row[0] if row else None
