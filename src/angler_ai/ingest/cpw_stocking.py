"""Colorado Parks and Wildlife (CPW) trout stocking ingestion.

v0 status: honest NotImplementedError. The CPW weekly stocking report at
https://cpw.state.co.us/fishing/stocking-report is HTML-only and exposes
only (body_of_water, region, report_date) - NO species and NO count fields,
so no usable stocking prior can be derived from it.

A Socrata catalog query against data.colorado.gov on 2026-06-20 returned
zero CPW stocking datasets. CPW Fishery Survey Summaries are published as
per-water PDFs with no machine-readable schema.

v1 path requires one of:
  (a) FOIA the CPW master stocking table (species + count + lifestage + date
      per stocking event), or
  (b) Build a per-water PDF scraper for the Fishery Survey Summary documents
      and extract species composition / electrofishing CPUE, or
  (c) Negotiate a data-share agreement with CPW.

Until one of those lands, this module is an honest NotImplementedError. The
Tier 1 non-native species prior (`Oncorhynchus mykiss`, `Salvelinus fontinalis`)
is built from USGS NAS occurrence records instead - presence-only but federal,
public-domain, and programmatically accessible.
"""

from __future__ import annotations

import logging

from angler_ai.features.store import FeatureStore
from angler_ai.ingest.base import IngestionModule, SourceMetadata

log = logging.getLogger(__name__)


class CPWStockingIngest(IngestionModule):
    metadata = SourceMetadata(
        source_id="CPW_STOCKING_v0",
        display_name="Colorado Parks and Wildlife stocking report",
        source_url="https://cpw.state.co.us/fishing/stocking-report",
        license="unspecified-state-agency",
        refresh_cadence="weekly (during season)",
        discovery_pattern=(
            "v0: NotImplementedError. Public CPW weekly report has only "
            "(body_of_water, region, report_date); no species, no count. "
            "Socrata data.colorado.gov has no CPW dataset. v1 path is FOIA "
            "of CPW master stocking table or PDF scrape of Fishery Survey "
            "Summaries."
        ),
        terms_of_use_url="https://cpw.state.co.us/fishing/stocking-report",
    )

    def ingest(self, store: FeatureStore, **_: object) -> int:
        raise NotImplementedError(
            "CPW stocking ingestion is honestly deferred to v1. The CPW "
            "weekly report is HTML-only with no species or count fields, "
            "and data.colorado.gov has no CPW stocking dataset. See module "
            "docstring for v1 paths. For non-native species priors at v0, "
            "the USGS NAS ingester provides federal presence-only coverage."
        )

    def last_ingested_at(self, store: FeatureStore) -> str | None:
        return None
