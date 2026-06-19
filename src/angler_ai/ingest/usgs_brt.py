"""USGS BRT v2.0 fluvial fish SDM priors ingestion.

FR-4.7. Pulls 415 native + 4 non-native species predictions per NHDPlusV2.1
COMID from the 112M-row parquet at:
    https://doi.org/10.5066/P1UV25FW
    file: fluvial_fish_brt_predictions_v2_0.parquet.gzip (~1 GB)

Predictions are keyed by V2.1 COMID. The HR-side query path joins via
xwalk_v2_to_hr (built separately by NHDPlusV2HRXwalkIngest).

v0 ingestion path: stream the parquet via DuckDB, filter to V2 COMIDs that
the crosswalk knows about for the target state, upsert into brt_priors.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import httpx

from angler_ai.config import default_paths
from angler_ai.features.store import FeatureStore
from angler_ai.ingest.base import IngestionModule, SourceMetadata

log = logging.getLogger(__name__)

ITEM_ID = "6760bf81d34e03058f220b48"
BASE_URL = f"https://www.sciencebase.gov/catalog/file/get/{ITEM_ID}"
PREDICTIONS_FILENAME = "fluvial_fish_brt_predictions_v2_0.parquet.gzip"
FISH_LIST_FILENAME = "fish_list_v2_0.csv"
DOI_URL = "https://doi.org/10.5066/P1UV25FW"


class USGSBRTFluvialFishIngest(IngestionModule):
    metadata = SourceMetadata(
        source_id="USGS_BRT_V2.0",
        display_name="USGS Fluvial Fish BRT v2.0 (Yu et al. 2024)",
        source_url=DOI_URL,
        license="public-domain",
        refresh_cadence="annual or per new vintage",
        discovery_pattern=(
            "Filter the 112M-row predictions parquet to V2.1 COMIDs present in "
            "xwalk_v2_to_hr; upsert with model_version='USGS_BRT_V2.0'."
        ),
        terms_of_use_url=DOI_URL,
    )

    def ingest(
        self,
        store: FeatureStore,
        *,
        state: str = "PA",
        cache_dir: Path | None = None,
        download: bool = True,
        min_probability: float = 0.05,
        **_: object,
    ) -> int:
        """Filter predictions to PA-relevant V2 COMIDs and upsert.

        Args:
            store: feature store.
            state: state code (informational; filtering is via xwalk_v2_to_hr).
            cache_dir: where the parquet lives. Defaults to
                ${data_dir}/raw/usgs_brt/.
            download: if True, fetch parquet + fish list when missing.
            min_probability: drop predictions below this probability to keep
                the table small (default 5%; full data has many sub-1% rows).

        Returns:
            Row count inserted into brt_priors.
        """
        paths = default_paths()
        cache = cache_dir or (paths.raw_data / "usgs_brt")
        cache.mkdir(parents=True, exist_ok=True)
        parquet_path = cache / PREDICTIONS_FILENAME
        fish_list_path = cache / FISH_LIST_FILENAME

        if download:
            if not fish_list_path.exists():
                _download(FISH_LIST_FILENAME, fish_list_path)
            if not parquet_path.exists():
                _download(PREDICTIONS_FILENAME, parquet_path)

        # Load species metadata table (small, ~419 rows).
        species_count = _load_species_table(store, fish_list_path)
        log.info("USGS BRT: %d species metadata rows loaded", species_count)

        # Pull the set of V2 COMIDs we care about for `state` from the xwalk.
        conn = store.connect()
        comid_rows = conn.execute(
            "SELECT DISTINCT comid_v2 FROM xwalk_v2_to_hr"
        ).fetchall()
        if not comid_rows:
            log.warning(
                "USGS BRT ingest: xwalk_v2_to_hr is empty. Run nhdplus_v2_xwalk first."
            )
            return 0
        v2_comids = [r[0] for r in comid_rows]
        log.info("USGS BRT: %d V2 COMIDs to look up in predictions", len(v2_comids))

        # DuckDB can read the gzipped parquet directly and execute the filter.
        now = datetime.now(timezone.utc).isoformat()
        with store.transaction() as txn:
            txn.execute(
                "DELETE FROM brt_priors WHERE model_version = 'USGS_BRT_V2.0'"
            )
            # Stage V2 COMIDs as an arrow table for an efficient IN-join.
            import pyarrow as pa
            comid_table = pa.table({"comid_v2": v2_comids})
            txn.register("_v2_comids", comid_table)
            inserted = txn.execute(
                f"""
                INSERT INTO brt_priors (comid, species, probability, auc, model_version, ingested_at)
                SELECT
                    p.comid AS comid,
                    p.scientific_name AS species,
                    p.predict_prob AS probability,
                    NULL AS auc,
                    'USGS_BRT_V2.0' AS model_version,
                    ? AS ingested_at
                FROM read_parquet('{parquet_path.as_posix()}') AS p
                JOIN _v2_comids c ON c.comid_v2 = p.comid
                WHERE p.predict_prob >= ?
                """,
                [now, float(min_probability)],
            )
            count = txn.execute(
                "SELECT COUNT(*) FROM brt_priors WHERE model_version = 'USGS_BRT_V2.0'"
            ).fetchone()[0]
            txn.unregister("_v2_comids")
        log.info(
            "USGS BRT ingest (%s): %d brt_priors rows inserted (min p=%.2f)",
            state, count, min_probability,
        )
        return int(count)

    def last_ingested_at(self, store: FeatureStore) -> str | None:
        row = store.connect().execute(
            "SELECT MAX(ingested_at) FROM brt_priors WHERE model_version = 'USGS_BRT_V2.0'",
        ).fetchone()
        return row[0] if row else None


def _load_species_table(store: FeatureStore, fish_list_path: Path) -> int:
    """Populate brt_species from fish_list_v2_0.csv. Idempotent."""
    import csv

    now = datetime.now(timezone.utc).isoformat()
    rows: list[tuple] = []
    with fish_list_path.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append((
                    int(r.get("itis_tsn") or 0) or None,
                    r["scientific_name"],
                    r.get("common_name"),
                    int(r["presences"]) if r.get("presences") else None,
                    int(r["absences"]) if r.get("absences") else None,
                    float(r["prevalence"]) if r.get("prevalence") else None,
                    r.get("order"),
                    r.get("family"),
                ))
            except (KeyError, ValueError):
                continue
    with store.transaction() as conn:
        conn.execute("DELETE FROM brt_species")
        if rows:
            from angler_ai.ingest._bulk import bulk_insert
            bulk_insert(
                conn, "brt_species", rows,
                columns=(
                    "itis_tsn", "scientific_name", "common_name",
                    "presences", "absences", "prevalence",
                    "taxonomic_order", "family",
                ),
                extra_literals={"ingested_at": now},
            )
    return len(rows)


def _download(filename: str, dest: Path) -> None:
    """Stream-download a file from ScienceBase."""
    url = f"{BASE_URL}?name={filename}"
    log.info("Downloading %s -> %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as r:
        r.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in r.iter_bytes(chunk_size=1 << 22):
                fh.write(chunk)
    log.info("Saved %s (%.0f MB)", dest, dest.stat().st_size / (1024 ** 2))
