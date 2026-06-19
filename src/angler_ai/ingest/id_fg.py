"""Idaho Department of Fish and Game ingestion. v0 launch state (PA + VA + ID).

Coverage probing happens at M2 (research pass B noted `data-idfggis.opendata.arcgis.com`
exists but was unreliable in the verification budget). If IDFG ArcGIS Hub turns
out to be insufficient, fall back to MT FWP or CO CPW.
"""

from __future__ import annotations

from angler_ai.features.store import FeatureStore
from angler_ai.ingest.base import IngestionModule, SourceMetadata


class IDFGStreamFisheriesIngest(IngestionModule):
    metadata = SourceMetadata(
        source_id="ID_FG_STREAM_FISHERIES",
        display_name="Idaho Department of Fish and Game stream fisheries",
        source_url="https://data-idfggis.opendata.arcgis.com",
        license="unspecified-state-agency",
        refresh_cadence="TBD - probe at M2",
        discovery_pattern=(
            "Catalog IDFG ArcGIS Hub for trout-stocking + regulations + access-point "
            "layers; if not machine-readable, fall back to MT FWP gis-mtfwp.hub.arcgis.com "
            "or CO CPW geodata-cpw.hub.arcgis.com."
        ),
        terms_of_use_url="https://idfg.idaho.gov/",
    )

    def ingest(self, store: FeatureStore, **kwargs: object) -> int:
        raise NotImplementedError(
            "M2 milestone: probe IDFG ArcGIS Hub for trout-stream layers and "
            "stocking schedules; ingest into stocking_events + regulations + reaches."
        )

    def last_ingested_at(self, store: FeatureStore) -> str | None:
        return None
