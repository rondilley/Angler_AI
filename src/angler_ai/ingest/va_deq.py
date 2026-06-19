"""Virginia DEQ trout streams ingestion. v0 launch state (PA + VA + ID).

Aligned to 9VAC-25-260 (state administrative code). Multi-format publication:
ArcGIS REST, CSV, Shapefile, GeoJSON, KML.
"""

from __future__ import annotations

from angler_ai.features.store import FeatureStore
from angler_ai.ingest.base import IngestionModule, SourceMetadata


class VADEQTroutStreamsIngest(IngestionModule):
    metadata = SourceMetadata(
        source_id="VA_DEQ_TROUT_STREAMS",
        display_name="Virginia DEQ Trout Streams and Rivers",
        source_url="https://data.virginia.gov/dataset/trout-streams-rivers",
        license="unspecified-state-agency",
        refresh_cadence="per state assessment cycle",
        discovery_pattern=(
            "Fetch the data.virginia.gov dataset metadata for current "
            "ArcGIS REST URL; fall back to data.virginia.gov CSV if REST fails."
        ),
        terms_of_use_url="https://www.deq.virginia.gov/",
    )

    def ingest(self, store: FeatureStore, **kwargs: object) -> int:
        raise NotImplementedError(
            "M2/M3 milestone: fetch VA DEQ trout streams (Stockable vs Natural per "
            "9VAC-25-260), join to NHDPlus HR COMID via VA SCRT/IDs, upsert "
            "regulations + stocking_events as applicable."
        )

    def last_ingested_at(self, store: FeatureStore) -> str | None:
        return None
