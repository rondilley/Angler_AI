"""EPA Water Quality Portal ingestion via the legacy `/data/Result/search` CSV
endpoint. FR-4.3.

WQP returns CSV bulk responses; we pull a small parameter set for the requested
state and recent window, then upsert into `reach_wq`. The newer wqx3 endpoint
exists but has been unstable; v0 sticks with the legacy endpoint and degrades
gracefully on 5xx.

Characteristics pulled for v0:
- Temperature, water
- Dissolved oxygen (DO)
- pH
- Specific conductance
- Turbidity
"""

from __future__ import annotations

import csv
import io
import logging
import time
from datetime import datetime, timezone

import httpx

from angler_ai.features.store import FeatureStore
from angler_ai.ingest._bulk import bulk_insert
from angler_ai.ingest.base import IngestionModule, SourceMetadata

log = logging.getLogger(__name__)

WQP_ENDPOINT = "https://www.waterqualitydata.us/data/Result/search"
DEFAULT_CHARACTERISTICS = (
    "Temperature, water",
    "Dissolved oxygen (DO)",
    "pH",
    "Specific conductance",
    "Turbidity",
)


class EPAWaterQualityPortalIngest(IngestionModule):
    metadata = SourceMetadata(
        source_id="EPA_WQP",
        display_name="EPA / USGS Water Quality Portal (legacy CSV endpoint)",
        source_url=WQP_ENDPOINT,
        license="public-domain",
        refresh_cadence="weekly batch",
        discovery_pattern=(
            "POST/GET CSV per (state, characteristicName) with a startDateLo "
            "window; retries with exponential backoff on 5xx."
        ),
    )

    def ingest(
        self,
        store: FeatureStore,
        *,
        state: str = "PA",
        start_date: str = "2024-01-01",
        characteristics: tuple[str, ...] = DEFAULT_CHARACTERISTICS,
        max_retries: int = 3,
        **_: object,
    ) -> int:
        rows: list[tuple] = []
        state_code = f"US:{_state_fips(state)}" if _state_fips(state) else None
        if not state_code:
            log.warning("EPA WQP: no FIPS code for state %s", state)
            return 0

        with httpx.Client(timeout=120, follow_redirects=True) as client:
            for char in characteristics:
                fetched = _fetch_characteristic(
                    client, state_code, char, start_date, max_retries,
                )
                if not fetched:
                    continue
                for r in _parse_csv(fetched, char):
                    rows.append(r)

        now = datetime.now(timezone.utc).isoformat()
        with store.transaction() as conn:
            conn.execute(
                "DELETE FROM reach_wq WHERE source = 'EPA_WQP' AND sample_date >= ?",
                [start_date],
            )
            if rows:
                # rows are (sample_date, parameter, value, unit, org_id); comid is unset.
                bulk_insert(
                    conn, "reach_wq", rows,
                    columns=("sample_date", "parameter", "value", "unit", "org_id"),
                    extra_literals={"source": "EPA_WQP", "ingested_at": now},
                )
        log.info("EPA WQP ingest (%s, since %s): %d rows", state, start_date, len(rows))
        return len(rows)

    def last_ingested_at(self, store: FeatureStore) -> str | None:
        row = store.connect().execute(
            "SELECT MAX(ingested_at) FROM reach_wq WHERE source = 'EPA_WQP'",
        ).fetchone()
        return row[0] if row else None


def _fetch_characteristic(
    client: httpx.Client,
    state_code: str,
    characteristic: str,
    start_date: str,
    max_retries: int,
) -> str | None:
    # WQP legacy endpoint accepts dates as MM-DD-YYYY.
    if len(start_date) == 10 and start_date[4] == "-":
        try:
            y, m, d = start_date.split("-")
            start_param = f"{m}-{d}-{y}"
        except ValueError:
            start_param = start_date
    else:
        start_param = start_date
    params = {
        "statecode": state_code,
        "characteristicName": characteristic,
        "startDateLo": start_param,
        "mimeType": "csv",
        "sorted": "no",
    }
    backoff = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            r = client.get(WQP_ENDPOINT, params=params)
            if r.status_code == 200:
                return r.text
            log.warning(
                "WQP attempt %d for %r returned %d; body[:120]=%s",
                attempt, characteristic, r.status_code, r.text[:120],
            )
        except httpx.HTTPError as exc:
            log.warning("WQP attempt %d for %r raised: %s", attempt, characteristic, exc)
        if attempt == max_retries:
            return None
        time.sleep(backoff)
        backoff *= 2
    return None


def _parse_csv(csv_text: str, characteristic: str) -> list[tuple]:
    """Parse a WQP CSV body into (sample_date, parameter, value, unit, org_id) rows."""
    rows: list[tuple] = []
    reader = csv.DictReader(io.StringIO(csv_text))
    for r in reader:
        date_str = (r.get("ActivityStartDate") or "").strip()
        val_str = (r.get("ResultMeasureValue") or "").strip()
        if not date_str or not val_str:
            continue
        try:
            value = float(val_str)
        except ValueError:
            continue
        unit = (r.get("ResultMeasure/MeasureUnitCode") or "").strip() or None
        org_id = (r.get("OrganizationIdentifier") or "").strip() or None
        rows.append((date_str, characteristic, value, unit, org_id))
    return rows


_STATE_FIPS = {"PA": "42", "VA": "51", "ID": "16"}


def _state_fips(state: str) -> str | None:
    return _STATE_FIPS.get(state)
