"""Data Ingestion (DI) - federal + state data normalized to NHDPlus HR COMID.

Every ingestion module conforms to the `IngestionModule` protocol in `base.py`
and declares its source URL, refresh cadence, license, and discovery pattern
(NFR-5.1, FR-4.11).
"""

from angler_ai.ingest.base import IngestionModule, SourceMetadata
from angler_ai.ingest.dispatch import IngestSummary, run

__all__ = ["IngestionModule", "IngestSummary", "SourceMetadata", "run"]
