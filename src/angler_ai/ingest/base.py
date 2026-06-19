"""IngestionModule protocol. Every data source implements this."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from angler_ai.features.store import FeatureStore


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    """Source provenance + licensing declared per ingestion module (NFR-5.1).

    Recorded in the data manifest (DR-2.4).
    """

    source_id: str
    """Stable id, e.g. 'EPA_WQP', 'USGS_BRT_V2.0', 'PA_PFBC_TROUT_STOCKED'."""

    display_name: str
    source_url: str
    license: str
    """License string. 'public-domain', 'unspecified-state-agency', 'Qwen', etc."""

    refresh_cadence: str
    """Free-text cadence, e.g. '15-min', 'weekly', 'twice per state assessment cycle'."""

    discovery_pattern: str | None = None
    """For year-stamped or otherwise-changing endpoints (e.g. PA PFBC
    TroutStocked<year>), describe the discovery approach. FR-4.8."""

    terms_of_use_url: str | None = None
    """Per CR-3.4: link the source terms-of-use when license is silent."""


@runtime_checkable
class IngestionModule(Protocol):
    """Protocol every ingestion source implements.

    Implementations are idempotent (NFR-2.1) and append idempotent upserts
    to the feature store.
    """

    metadata: SourceMetadata

    def ingest(self, store: FeatureStore, **kwargs: object) -> int:
        """Run ingestion. Returns the count of rows inserted/updated."""
        ...

    def last_ingested_at(self, store: FeatureStore) -> str | None:
        """Return ISO timestamp of last successful ingest, or None."""
        ...
