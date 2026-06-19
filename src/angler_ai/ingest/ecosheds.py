"""EcoSHEDS Northeast Stream Temperature Model + NECD + BTO ingestion. FR-4.6.

v0 status: NotImplementedError. The full ingestion requires:
  - NECD catchment shapefile (https://github.com/EcoSHEDS/necd) ~hundreds of MB
  - EcoSHEDS Letcher hierarchical Bayesian model output (stream temperature
    predictions, gigabytes for daily). Distribution channels: db.ecosheds.org
    (API not publicly documented) and bookdown release artifacts.
  - Spatial-join NECD reaches to NHDPlus HR by REACHCODE / geometry
  - 200 km^2 drainage ceiling on BTO outputs.

NO FABRICATED VALUES. v0 returns 0 rows from these modules; the temperature
resolver (`prediction.temperature`) honestly reports `source='not_modeled'`
for reaches without ingested temperature data. The species_priors / map_export
output surfaces that explicitly so users know what is and is not modeled.
"""

from __future__ import annotations

import logging

from angler_ai.features.store import FeatureStore
from angler_ai.ingest.base import IngestionModule, SourceMetadata

log = logging.getLogger(__name__)

ECOSHEDS_STATES = ("ME", "NH", "VT", "MA", "RI", "CT", "NY", "NJ", "PA", "DE", "MD", "VA")
BTO_DRAINAGE_CEILING_KM2 = 200.0


class EcoSHEDSTempIngest(IngestionModule):
    """EcoSHEDS Northeast stream temperature ingestion. v0 stub.

    v1 plan:
      1. Download NECD catchments shapefile via `EcoSHEDS/necd` releases.
      2. Download Letcher model predictions from db.ecosheds.org or release
         artifacts (API channel needs separate research).
      3. Build xwalk_necd_to_hr (DR-1.3) by joining on NHDReachCode or
         spatial intersection.
      4. Upsert per-COMID temperature with source='EcoSHEDS_TEMP' and the
         real Letcher per-reach uncertainty.
    """

    metadata = SourceMetadata(
        source_id="EcoSHEDS_TEMP",
        display_name="EcoSHEDS Northeast Stream Temperature Model",
        source_url="https://ecosheds.github.io/northeast-temp-model/",
        license="public-domain",
        refresh_cadence="per release",
    )

    def ingest(self, store: FeatureStore, *, state: str = "PA", **_: object) -> int:
        if state not in ECOSHEDS_STATES:
            log.warning("EcoSHEDS: %s outside Northeast model coverage", state)
            return 0
        log.info(
            "EcoSHEDS v0: NotImplementedError. NECD shapefile + Letcher model "
            "predictions are v1 work. Reaches will surface source='not_modeled' "
            "until real data lands."
        )
        return 0

    def last_ingested_at(self, store: FeatureStore) -> str | None:
        row = store.connect().execute(
            "SELECT MAX(ingested_at) FROM reach_temperature WHERE source = 'EcoSHEDS_TEMP'",
        ).fetchone()
        return row[0] if row else None


class EcoSHEDSBTOIngest(IngestionModule):
    """Brook Trout Occupancy ingestion. v0 stub.

    v1 plan: ingest the Letcher BTO model output (+0/+2/+4/+6 deg C climate
    scenarios) tied to NECD reaches with drainage < 200 km^2.
    """

    metadata = SourceMetadata(
        source_id="EcoSHEDS_BTO",
        display_name="EcoSHEDS Northeast Brook Trout Occupancy",
        source_url="https://ecosheds.github.io/northeast-bto-model/",
        license="public-domain",
        refresh_cadence="per release",
    )

    def ingest(self, store: FeatureStore, **_: object) -> int:
        log.info("EcoSHEDS BTO v0: NotImplementedError. Letcher BTO output is v1.")
        return 0

    def last_ingested_at(self, store: FeatureStore) -> str | None:
        return None


class NECDIngest(IngestionModule):
    """Northeast Catchment Delineation. v0 stub.

    v1 plan: download NECD catchments shapefile from
    https://github.com/EcoSHEDS/necd and populate xwalk_necd_to_hr.
    """

    metadata = SourceMetadata(
        source_id="EcoSHEDS_NECD",
        display_name="Northeast Catchment Delineation",
        source_url="https://github.com/EcoSHEDS/necd",
        license="public-domain",
        refresh_cadence="static",
    )

    def ingest(self, store: FeatureStore, **_: object) -> int:
        log.info("NECD v0: NotImplementedError. Catchment shapefile + xwalk is v1.")
        return 0

    def last_ingested_at(self, store: FeatureStore) -> str | None:
        return None
