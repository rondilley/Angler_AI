"""Ingest dispatcher. Maps a `source` argument to one or more IngestionModule
instances and routes the call to each. Also writes the data_manifest.json
(DR-2.4): every dataset's source id, license, refresh cadence, last successful
refresh, and row count.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from angler_ai.features.store import FeatureStore
from angler_ai.ingest.base import IngestionModule
from angler_ai.ingest.epa_attains import EPAATTAINSIngest
from angler_ai.ingest.epa_wqp import EPAWaterQualityPortalIngest
from angler_ai.ingest.nhdplus_hr import NHDPlusHRIngest
from angler_ai.ingest.nhdplus_v2_xwalk import NHDPlusV2HRXwalkIngest
from angler_ai.ingest.pa_pfbc import PAPFBCTroutStockedIngest
from angler_ai.ingest.usgs_brt import USGSBRTFluvialFishIngest
from angler_ai.ingest.usgs_nwis import USGSNWISIngest

log = logging.getLogger(__name__)


# Ordered so the foundational reaches table is populated first; downstream
# joins (PFBC -> COMID, ATTAINS -> COMID, NWIS -> COMID) can attach after.
_DEFAULT_ORDER: tuple[tuple[str, type[IngestionModule]], ...] = (
    ("nhdplus", NHDPlusHRIngest),
    ("v2_xwalk", NHDPlusV2HRXwalkIngest),
    ("brt", USGSBRTFluvialFishIngest),
    ("attains", EPAATTAINSIngest),
    ("nwis", USGSNWISIngest),
    ("wqp", EPAWaterQualityPortalIngest),
    ("pa_pfbc", PAPFBCTroutStockedIngest),
)


@dataclass(slots=True)
class IngestSummary:
    """One IngestionModule run result, persisted in the data manifest."""

    source_id: str
    display_name: str
    license: str
    refresh_cadence: str
    state: str
    rows_written: int
    last_refresh: str
    ok: bool
    error: str | None = None


def run(
    store: FeatureStore,
    *,
    source: str = "all",
    state: str = "PA",
    manifest_path: Path | None = None,
    extra_kwargs: dict[str, object] | None = None,
) -> list[IngestSummary]:
    """Dispatch the requested source(s) for the given state.

    Args:
        store: feature store.
        source: 'all' or one of the source ids ('nhdplus', 'attains', 'nwis',
            'wqp', 'pa_pfbc').
        state: two-letter state code.
        manifest_path: where to write the data manifest. Skipped if None.

    Returns:
        Per-module IngestSummary list.
    """
    summaries: list[IngestSummary] = []
    targets = _DEFAULT_ORDER if source == "all" else tuple(
        (k, c) for k, c in _DEFAULT_ORDER if k == source
    )
    if not targets:
        raise ValueError(
            f"Unknown source {source!r}. Pick from "
            f"{tuple(k for k, _ in _DEFAULT_ORDER)} or 'all'."
        )

    # v0 smoke-test bounds: small enough to finish in minutes, large enough to
    # demonstrably populate every table. Future ingest paths will scale up.
    PER_MODULE_KWARGS: dict[str, dict[str, object]] = {
        "wqp": {"start_date": "2025-06-01", "max_retries": 2},
        "nwis": {"max_gauges": 20},
    }

    for key, cls in targets:
        module = cls()
        meta = module.metadata
        log.info("Ingesting %s for %s ...", meta.source_id, state)
        started = datetime.now(timezone.utc)
        try:
            module_kwargs = dict(PER_MODULE_KWARGS.get(key, {}))
            if extra_kwargs:
                module_kwargs.update(extra_kwargs)
            rows = module.ingest(store, state=state, **module_kwargs)
            summary = IngestSummary(
                source_id=meta.source_id,
                display_name=meta.display_name,
                license=meta.license,
                refresh_cadence=meta.refresh_cadence,
                state=state,
                rows_written=int(rows),
                last_refresh=started.isoformat(),
                ok=True,
            )
            log.info("%s OK: %d rows", meta.source_id, rows)
        except Exception as exc:  # noqa: BLE001 - we surface and continue
            log.exception("%s failed: %s", meta.source_id, exc)
            summary = IngestSummary(
                source_id=meta.source_id,
                display_name=meta.display_name,
                license=meta.license,
                refresh_cadence=meta.refresh_cadence,
                state=state,
                rows_written=0,
                last_refresh=started.isoformat(),
                ok=False,
                error=str(exc)[:300],
            )
        summaries.append(summary)

    if manifest_path is not None:
        _update_manifest(manifest_path, summaries)
    return summaries


def _update_manifest(manifest_path: Path, summaries: list[IngestSummary]) -> None:
    """Merge per-run dataset summaries into the data manifest. Idempotent per
    (source_id, state)."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        data = {"models": {}, "datasets": {}}
    datasets = data.setdefault("datasets", {})
    for s in summaries:
        key = f"{s.source_id}::{s.state}"
        datasets[key] = asdict(s)
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    log.info("Data manifest updated: %s (%d datasets)", manifest_path, len(datasets))
