"""NorWeST modeled stream temperature ingestion (West states). FR-4.5.

v0 status: NotImplementedError. The full ingestion requires:
  - Download NorWeST shapefile per Processing Unit (PU) from
    https://research.fs.usda.gov/rmrs/projects/norwest
  - Each PU shapefile carries reach geometry + 36 scenario columns
    (S1_93_11 = mean August 1993-2011 baseline, plus future scenarios)
  - NHD-registered geometries: spatial-join to NHDPlus HR by REACHCODE
  - Upsert per-COMID temperature with source='NorWeST'

NO FABRICATED VALUES. v0 returns 0 rows; reaches surface source='not_modeled'
when queried until the real NorWeST shapefiles are wired up.
"""

from __future__ import annotations

import logging

from angler_ai.features.store import FeatureStore
from angler_ai.ingest.base import IngestionModule, SourceMetadata

log = logging.getLogger(__name__)

# NorWeST coverage states (research pass 1; AZ unconfirmed in pass 2).
NORWEST_STATES = ("ID", "MT", "NV", "OR", "UT", "WA", "WY", "CA", "CO", "NM")


class NorWeSTIngest(IngestionModule):
    metadata = SourceMetadata(
        source_id="NorWeST",
        display_name="USFS RMRS NorWeST mean-August stream temperature",
        source_url="https://research.fs.usda.gov/rmrs/projects/norwest",
        license="public-domain",
        refresh_cadence="per release",
    )

    def ingest(self, store: FeatureStore, *, state: str = "ID", **_: object) -> int:
        if state not in NORWEST_STATES:
            log.warning("NorWeST: %s outside western model coverage", state)
            return 0
        log.info(
            "NorWeST v0: NotImplementedError. Per-PU shapefile ingest is v1 "
            "work. Reaches will surface source='not_modeled' until real "
            "shapefiles are wired up."
        )
        return 0

    def last_ingested_at(self, store: FeatureStore) -> str | None:
        row = store.connect().execute(
            "SELECT MAX(ingested_at) FROM reach_temperature WHERE source = 'NorWeST'",
        ).fetchone()
        return row[0] if row else None
