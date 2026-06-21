"""NorWeST modeled stream temperature ingestion (West states). FR-4.5.

Reads the per-processing-unit shapefile published by USFS RMRS and
populates `reach_temp_baseline` with the mean-August 1993-2011 scenario
(`S1_93_11`). Used as the per-reach `T_water_baseline` anchor for the
Mohseni-Stefan air-to-water projection in `water_temp_model.py`.

NorWeST is climatology, NOT a daily forecast. It does NOT participate in
the `temperature.py` resolver chain; it only feeds the baseline anchor.
See plan notes at C:/Users/rondi/.claude/plans/cozy-tickling-coral.md.

Data acquisition (manual, one-time per processing unit):
  1. Visit https://research.fs.usda.gov/rmrs/projects/norwest
  2. Download the "Colorado" processing-unit modeled-summer-temperature
     shapefile zip.
  3. Unzip into `${ANGLER_AI_DATA}/raw/norwest/<PU_name>/`.
  4. Run `angler-ai ingest --source norwest --norwest-shapefile <path/to/shp>`
     OR drop the .shp + sidecar files in the conventional path below and
     the ingester auto-discovers.

Conventional path (auto-discovered):
  ${data_dir}/raw/norwest/Colorado/NorWeST_PredictedStreams_Colorado.shp

If no shapefile is present the ingester logs an explicit warning and
returns 0 rows. Downstream `water_temp_model.py` falls back to its
existing Mohseni-Stefan stratified defaults - no silent degradation, no
fabricated values.

Shapefile fields used (per RDS-2016-0033 metadata):
  - COMID       (NHDPlusV2 COMID; we join via xwalk_v2_to_hr to get HR COMID)
  - S1_93_11    (mean August 1993-2011 modeled water temperature, deg C)

Other scenario columns (S2_*, S3_*, ... future climate) are ignored at v0.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from angler_ai.config import default_paths
from angler_ai.features.store import FeatureStore
from angler_ai.ingest._bulk import bulk_insert
from angler_ai.ingest.base import IngestionModule, SourceMetadata

log = logging.getLogger(__name__)

# Processing units NorWeST covers. Used to map state -> PU when the caller
# supplies only a state.
_PU_BY_STATE: dict[str, str] = {
    "CO": "Colorado",
    "NM": "NewMexico",
    "UT": "Utah",
    "WY": "Colorado",  # WY headwaters of the Colorado basin fall into the CO PU
    "AZ": "Arizona",
    "ID": "MissouriHeadwaters",  # rough mapping; ID also overlaps SnakeBear
    "MT": "MissouriHeadwaters",
    "NV": "Lahontan",
    "OR": "PacificNW",
    "WA": "PacificNW",
    "CA": "CentralCalifornia",
}

NORWEST_STATES = tuple(_PU_BY_STATE.keys())

# Scenario column carrying the mean August 1993-2011 modeled temperature.
# The metadata lists scenarios as "S#_YEAR" or "S#_STARTYEAR-ENDYEAR";
# the v1.x historical baseline is consistently "S1_93_11".
_BASELINE_SCENARIO = "S1_93_11"

# Month tag we write into reach_temp_baseline. NorWeST is August climatology.
_BASELINE_MONTH = 8


def _conventional_shapefile_path(pu_name: str) -> Path:
    """Default location the ingester checks if no path is supplied."""
    paths = default_paths()
    return (
        paths.raw_data
        / "norwest"
        / pu_name
        / f"NorWeST_PredictedStreams_{pu_name}.shp"
    )


class NorWeSTIngest(IngestionModule):
    metadata = SourceMetadata(
        source_id="NorWeST",
        display_name=(
            "USFS RMRS NorWeST mean-August stream temperature baseline "
            "(scenario S1_93_11)"
        ),
        source_url="https://research.fs.usda.gov/rmrs/projects/norwest",
        license="public-domain",
        refresh_cadence="per release (annual or less frequent)",
        discovery_pattern=(
            "Per-processing-unit shapefile; manually downloaded one-time "
            "and read via DuckDB ST_Read. v0 uses S1_93_11 baseline only; "
            "scenario columns (S2_*, S3_*) deferred to v1."
        ),
        terms_of_use_url="https://doi.org/10.2737/RDS-2016-0033",
    )

    def ingest(
        self,
        store: FeatureStore,
        *,
        state: str = "CO",
        shapefile: str | None = None,
        **_: object,
    ) -> int:
        """Populate `reach_temp_baseline` from a NorWeST PU shapefile.

        Args:
            store: feature store.
            state: state code; selects the conventional processing unit.
            shapefile: explicit path to a .shp file. If None, auto-discover
                via `_conventional_shapefile_path(state's PU)`.

        Returns:
            Row count inserted into `reach_temp_baseline`.
        """
        if state not in _PU_BY_STATE:
            log.warning("NorWeST: state %s outside western model coverage", state)
            return 0
        pu_name = _PU_BY_STATE[state]

        shp_path = Path(shapefile) if shapefile else _conventional_shapefile_path(pu_name)
        if not shp_path.exists():
            log.warning(
                "NorWeST: shapefile not found at %s. Download the %s "
                "processing-unit shapefile from "
                "https://research.fs.usda.gov/rmrs/projects/norwest and "
                "place under %s, then re-run ingest. Falling through with "
                "0 rows; downstream water_temp_model.py will use "
                "stratified Mohseni-Stefan defaults.",
                shp_path, pu_name, shp_path.parent,
            )
            return 0

        log.info("NorWeST: reading shapefile %s", shp_path)
        conn = store.connect()
        # DuckDB spatial extension reads shapefile via ST_Read. We pull only
        # the columns we need; this avoids loading the full attribute table
        # (some scenario columns are dozens of MB).
        try:
            rows = conn.execute(
                f"""
                SELECT TRY_CAST(COMID AS BIGINT) AS comid_v2,
                       TRY_CAST({_BASELINE_SCENARIO} AS DOUBLE) AS baseline_c
                FROM ST_Read(?)
                WHERE COMID IS NOT NULL
                  AND {_BASELINE_SCENARIO} IS NOT NULL
                """,
                [str(shp_path)],
            ).fetchall()
        except Exception as exc:  # noqa: BLE001 - report and abort cleanly
            log.error(
                "NorWeST: ST_Read failed on %s: %s. The shapefile may be a "
                "different vintage with renamed columns. Expected COMID and "
                "%s columns.",
                shp_path, exc, _BASELINE_SCENARIO,
            )
            return 0
        if not rows:
            log.warning("NorWeST: 0 rows extracted from %s", shp_path)
            return 0
        log.info("NorWeST: %d V2 reaches with %s baseline", len(rows), _BASELINE_SCENARIO)

        # Join via xwalk_v2_to_hr to land on HR COMIDs the rest of the
        # feature store uses. One V2 reach can map to multiple HR reaches.
        v2_to_hr = self._fetch_xwalk(store)
        if not v2_to_hr:
            log.warning(
                "NorWeST: xwalk_v2_to_hr is empty. Run "
                "`angler-ai ingest --source v2_xwalk --huc8 ...` first."
            )
            return 0

        hr_rows: list[tuple] = []
        unmapped = 0
        for v2_comid, baseline_c in rows:
            if v2_comid is None or baseline_c is None:
                continue
            hr_comids = v2_to_hr.get(int(v2_comid))
            if not hr_comids:
                unmapped += 1
                continue
            for hr_comid in hr_comids:
                hr_rows.append((hr_comid, _BASELINE_MONTH, float(baseline_c)))
        if unmapped:
            log.info(
                "NorWeST: %d V2 reaches lacked an HR mapping in xwalk_v2_to_hr "
                "(likely outside the loaded NHDPlus_HR scope)",
                unmapped,
            )

        if not hr_rows:
            log.warning(
                "NorWeST: 0 rows landed on HR COMIDs. Check that "
                "xwalk_v2_to_hr covers the same processing unit."
            )
            return 0

        now = datetime.now(timezone.utc).isoformat()
        with store.transaction() as conn:
            # Idempotent: drop prior NorWeST rows for these HR comids before
            # re-inserting.
            conn.execute(
                "DELETE FROM reach_temp_baseline WHERE source = ?",
                [self.metadata.source_id],
            )
            bulk_insert(
                conn, "reach_temp_baseline", hr_rows,
                columns=("comid", "month", "baseline_temp_c"),
                extra_literals={
                    "source": self.metadata.source_id,
                    "ingested_at": now,
                },
            )
        log.info(
            "NorWeST: wrote %d reach_temp_baseline rows (HR COMIDs, "
            "month=%d, source=%s)",
            len(hr_rows), _BASELINE_MONTH, self.metadata.source_id,
        )
        return len(hr_rows)

    @staticmethod
    def _fetch_xwalk(store: FeatureStore) -> dict[int, list[int]]:
        """Return {v2_comid: [hr_comid, ...]} from xwalk_v2_to_hr."""
        rows = store.connect().execute(
            "SELECT comid_v2, comid_hr FROM xwalk_v2_to_hr"
        ).fetchall()
        out: dict[int, list[int]] = {}
        for v2, hr in rows:
            out.setdefault(int(v2), []).append(int(hr))
        return out

    def last_ingested_at(self, store: FeatureStore) -> str | None:
        row = store.connect().execute(
            "SELECT MAX(ingested_at) FROM reach_temp_baseline WHERE source = ?",
            [self.metadata.source_id],
        ).fetchone()
        return row[0] if row else None
