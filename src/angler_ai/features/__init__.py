"""Feature Store (FS) - DuckDB backing all reach-level joins.

See design 5.5, DR-4.1. Spatial extension used for geometry-aware queries.
"""

from angler_ai.features.store import FeatureStore, open_store

__all__ = ["FeatureStore", "open_store"]
