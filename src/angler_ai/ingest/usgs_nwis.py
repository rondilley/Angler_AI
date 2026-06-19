"""USGS NWIS streamflow ingestion via the Water Data OGC API. FR-4.1.

Two paths:
- Bulk ingest: snapshot the recent ~24h of discharge for every gauge in a
  state's bbox into `reach_flow`.
- On-demand: query a single gauge for the freshest value at query time.

The OGC API endpoint family:
- monitoring-locations: gauge metadata (id, name, lat/lon, agency)
- continuous: instantaneous (~15-min) sensor measurements

Discharge is parameter code 00060 (cubic feet per second).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Iterator

import httpx

from angler_ai.features.store import FeatureStore
from angler_ai.ingest._bulk import bulk_insert
from angler_ai.ingest.base import IngestionModule, SourceMetadata

log = logging.getLogger(__name__)

OGC_ROOT = "https://api.waterdata.usgs.gov/ogcapi/v0"
DISCHARGE_PCODE = "00060"
GAGE_HEIGHT_PCODE = "00065"

_STATE_FIPS = {"PA": "42", "VA": "51", "ID": "16"}


class USGSNWISIngest(IngestionModule):
    metadata = SourceMetadata(
        source_id="USGS_NWIS",
        display_name="USGS Water Data OGC API (NWIS streamflow)",
        source_url=OGC_ROOT,
        license="public-domain",
        refresh_cadence="15-min instantaneous; cached snapshot every ingest",
        discovery_pattern=(
            "monitoring-locations filtered by state bbox; "
            "continuous query per gauge for last 24h of discharge."
        ),
    )

    def ingest(
        self,
        store: FeatureStore,
        *,
        state: str = "PA",
        lookback_hours: int = 24,
        max_gauges: int = 50,
        **_: object,
    ) -> int:
        """Pull recent discharge for up to `max_gauges` stream gauges in `state`.

        v0 caps the number of gauges to keep the smoke test fast. Production
        ingest at M3+ will expand or do this per query.
        """
        fips = _STATE_FIPS.get(state)
        if fips is None:
            log.warning("USGS NWIS ingest: no FIPS code known for state %s", state)
            return 0
        with httpx.Client(timeout=60) as client:
            gauges = list(_list_stream_gauges(client, fips, max_gauges))
            log.info("USGS NWIS: %d candidate gauges in %s bbox", len(gauges), state)
            now = datetime.now(timezone.utc)
            since = now - timedelta(hours=lookback_hours)
            rows: list[tuple] = []
            for gauge_id, gauge_name in gauges:
                try:
                    samples = list(_continuous_discharge(client, gauge_id, since, now))
                except httpx.HTTPError as exc:
                    log.debug("Gauge %s discharge fetch failed: %s", gauge_id, exc)
                    continue
                if not samples:
                    continue
                for ts, discharge_cfs in samples:
                    rows.append((
                        None,                # comid (NHDPlus join happens after reaches load)
                        gauge_id,
                        ts.isoformat(),
                        discharge_cfs,
                        None,                # gauge_height_ft not pulled in v0
                    ))

        ingested_at = datetime.now(timezone.utc).isoformat()
        with store.transaction() as conn:
            conn.execute(
                "DELETE FROM reach_flow WHERE source = 'USGS_NWIS' AND ts >= ?",
                [since.isoformat()],
            )
            if rows:
                bulk_insert(
                    conn, "reach_flow", rows,
                    columns=("comid", "gauge_id", "ts", "discharge_cfs", "gauge_height_ft"),
                    extra_literals={"source": "USGS_NWIS", "ingested_at": ingested_at},
                )
        log.info("USGS NWIS ingest: %d discharge samples across %d gauges", len(rows), len(gauges))
        return len(rows)

    def last_ingested_at(self, store: FeatureStore) -> str | None:
        row = store.connect().execute(
            "SELECT MAX(ingested_at) FROM reach_flow WHERE source = 'USGS_NWIS'",
        ).fetchone()
        return row[0] if row else None


def _list_stream_gauges(
    client: httpx.Client,
    state_fips: str,
    limit: int,
) -> Iterator[tuple[str, str]]:
    """Yield (full_monitoring_location_id, name) for USGS stream gauges in `state_fips`.

    Uses the OGC `state_code` filter (NOT `district_code` - same value but
    `state_code` is the documented monitoring-locations attribute) plus
    `agency_code=USGS` to drop synoptic non-USGS sites that often lack
    continuous discharge.
    """
    params = {
        "state_code": state_fips,
        "limit": min(limit, 1000),
        "site_type_code": "ST",  # ST = stream
        "agency_code": "USGS",
        "f": "json",
    }
    r = client.get(f"{OGC_ROOT}/collections/monitoring-locations/items", params=params)
    r.raise_for_status()
    for feat in r.json().get("features", []):
        props = feat.get("properties") or {}
        # Use the full id (includes agency prefix like 'USGS-01554000') so the
        # continuous endpoint matches without us reconstructing it.
        full_id = (props.get("id") or "").strip()
        if not full_id:
            continue
        yield (full_id, (props.get("monitoring_location_name") or "").strip())


def _continuous_discharge(
    client: httpx.Client,
    monitoring_location_id: str,
    since: datetime,
    until: datetime,
) -> Iterator[tuple[datetime, float]]:
    """Yield (timestamp, discharge_cfs) for a gauge over the time window."""
    params = {
        "monitoring_location_id": monitoring_location_id,
        "parameter_code": DISCHARGE_PCODE,
        "datetime": f"{since.isoformat()}/{until.isoformat()}",
        "limit": 1000,
        "f": "json",
    }
    r = client.get(f"{OGC_ROOT}/collections/continuous/items", params=params)
    if r.status_code == 404:
        return
    r.raise_for_status()
    for feat in r.json().get("features", []):
        props = feat.get("properties") or {}
        ts_str = props.get("time")
        value = props.get("value")
        if ts_str is None or value is None:
            continue
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        try:
            yield (ts, float(value))
        except (TypeError, ValueError):
            continue
