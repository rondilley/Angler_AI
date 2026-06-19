"""Pennsylvania PFBC TroutStocked ingestion. FR-4.8.

PFBC publishes one MapServer per stocking year. Two naming conventions seen:
  - TroutStocked<YEAR>            (2024, 2025)
  - TroutStockedSections_<YEAR>   (2026 and presumably forward)

We discover available services at runtime rather than hardcoding URLs. The
`SecDate` field on each feature is the stocking date; the geometry is a
polyline along the stocked stream section.

Reaches are joined to NHDPlus HR COMID after NHDPlus HR is loaded (M2). For
the v0 path we attach `comid=None` when the join cannot be made and let the
caller upgrade later. This lets the smoke test run before reaches are present.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from angler_ai.features.store import FeatureStore
from angler_ai.ingest._arcgis import list_services, query_layer
from angler_ai.ingest._bulk import bulk_insert
from angler_ai.ingest.base import IngestionModule, SourceMetadata

log = logging.getLogger(__name__)

PFBC_REST_ROOT = "https://fbweb.pa.gov/arcgis/rest/services/PFBC_Map_Services"


class PAPFBCTroutStockedIngest(IngestionModule):
    metadata = SourceMetadata(
        source_id="PA_PFBC_TROUT_STOCKED",
        display_name="Pennsylvania PFBC Trout-Stocked Sections",
        source_url=PFBC_REST_ROOT,
        license="unspecified-state-agency",
        refresh_cadence="weekly during stocking season",
        discovery_pattern=(
            "List services under PFBC_Map_Services; match both "
            "TroutStocked<YEAR> and TroutStockedSections_<YEAR> naming."
        ),
        terms_of_use_url="https://www.fishandboat.com/",
    )

    _SERVICE_RE = re.compile(r"(?:TroutStocked|TroutStockedSections_?)(\d{4})$")

    def discover_services(self) -> dict[int, str]:
        """Return {year: MapServer URL} for every available TroutStocked* service."""
        services = list_services(PFBC_REST_ROOT)
        found: dict[int, str] = {}
        for svc in services:
            name = svc.get("name", "")
            if svc.get("type") != "MapServer":
                continue
            short = name.split("/", 1)[-1]
            m = self._SERVICE_RE.search(short)
            if m:
                year = int(m.group(1))
                found[year] = f"{PFBC_REST_ROOT}/{short}/MapServer"
        log.info("Discovered PFBC TroutStocked services for years: %s", sorted(found.keys()))
        return found

    def ingest(self, store: FeatureStore, **kwargs: object) -> int:
        """Pull every discovered year, normalize, upsert into stocking_events
        and regulations.

        Returns the count of stocking_events rows written.
        """
        years_kw = kwargs.get("years")
        services = self.discover_services()
        if years_kw:
            services = {y: u for y, u in services.items() if y in years_kw}
        if not services:
            log.warning("No PFBC TroutStocked services discovered.")
            return 0

        rows: list[tuple] = []
        regs: dict[tuple, tuple] = {}
        for year, layer_root in sorted(services.items()):
            layer_url = f"{layer_root}/0"
            for feat in query_layer(layer_url, out_fields="*", return_geometry=False, max_record_count=1000):
                attrs = feat.get("attributes", {})
                row = self._normalize_feature(attrs, year)
                if row is not None:
                    rows.append(row)
                    # Derive a minimal stocking-aware regulation row per water.
                    water = row[5]  # water_body_name (position 5 in _normalize_feature tuple)
                    if water:
                        regs[(water, "rainbow")] = (
                            "PA", water, None, "rainbow",
                            None, None, None, None, True,
                            "stocked_trout",
                            f"https://fishandboat.com/Fish/TroutStocking/Pages/default.aspx",
                            "PA_PFBC_TROUT_STOCKED",
                        )

        now = datetime.now(timezone.utc).isoformat()
        with store.transaction() as conn:
            # Idempotent: clear PA rows from PFBC then bulk-insert current state.
            conn.execute(
                "DELETE FROM stocking_events WHERE source = ?",
                ["PA_PFBC_TROUT_STOCKED"],
            )
            if rows:
                # rows are (comid, state, event_date, species, count, water_body_name, source)
                # 'source' is the 7th element of the tuple; bulk_insert drops it
                # in favor of the extra_literal of the same name.
                rows_for_bulk = [r[:6] for r in rows]
                bulk_insert(
                    conn, "stocking_events", rows_for_bulk,
                    columns=("comid", "state", "event_date", "species", "count", "water_body_name"),
                    extra_literals={"source": "PA_PFBC_TROUT_STOCKED", "ingested_at": now},
                )
            conn.execute(
                "DELETE FROM regulations WHERE source = ?",
                ["PA_PFBC_TROUT_STOCKED"],
            )
            if regs:
                reg_rows = [r[:11] for r in regs.values()]
                bulk_insert(
                    conn, "regulations", reg_rows,
                    columns=(
                        "state", "water_body_id", "comid", "species",
                        "season_start", "season_end", "gear_restrictions", "bag_limit",
                        "license_required", "special_regulation", "source_url",
                    ),
                    extra_literals={"source": "PA_PFBC_TROUT_STOCKED", "ingested_at": now},
                )

        log.info("PFBC ingest: %d stocking_events, %d regulations rows", len(rows), len(regs))
        return len(rows)

    @staticmethod
    def _normalize_feature(attrs: dict, year: int) -> tuple | None:
        """Coerce one PFBC feature into a stocking_events row tuple."""
        water = attrs.get("WtrName") or attrs.get("WtrName_1")
        sec_date = attrs.get("SecDate")
        if not water:
            return None
        event_date_str = _coerce_pfbc_date(sec_date, year)
        # Species: PFBC stocks rainbow, brown, brook, and occasionally other strains.
        # The TroutStocked layers do not always carry per-event species. Default to
        # 'rainbow' with a note; M3 acceptance gate refines per WRDS classification.
        species = "rainbow"
        # comid is unset; M2 NHDPlus HR loader + a join pass populates it.
        return (
            None,                       # comid
            "PA",                       # state
            event_date_str,             # event_date
            species,                    # species
            None,                       # count (PFBC doesn't expose per-event)
            water,                      # water_body_name
            "PA_PFBC_TROUT_STOCKED",    # source
        )

    def last_ingested_at(self, store: FeatureStore) -> str | None:
        conn = store.connect()
        row = conn.execute(
            "SELECT MAX(ingested_at) FROM stocking_events WHERE source = ?",
            ["PA_PFBC_TROUT_STOCKED"],
        ).fetchone()
        return row[0] if row else None


def _coerce_pfbc_date(value: object, year: int) -> str | None:
    """PFBC SecDate is sometimes an Esri epoch-ms int and sometimes a string."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc).date().isoformat()
        except (OSError, ValueError):
            return None
    if isinstance(value, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%m/%d/%Y"):
            try:
                return datetime.strptime(value[: len(fmt) + 5], fmt).date().isoformat()
            except ValueError:
                continue
    return None
