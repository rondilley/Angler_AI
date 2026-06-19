"""Tribal-waters mask. CR-2.1/2.2.

The mask is populated from public USGS / state datasets only (CR-2.3). We do
NOT scrape, hot-link, or redistribute CRITFC ArcGIS Online layers. Queries
that intersect tribally-managed waters surface a sovereignty notice and a
redirect to the tribe's published data resources.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TribalRegion:
    """A tribal-jurisdiction region in the mask."""

    region_id: str
    tribe_name: str
    redirect_url: str
    source: str
    """Where we sourced the boundary (e.g. 'USGS_Reservations', 'BIA_LAR_2024')."""


# Seed list of redirect URLs for the major tribally-managed water authorities.
# Boundaries themselves are loaded at M2 from USGS / BIA public datasets.
TRIBAL_REDIRECTS: dict[str, str] = {
    "critfc": "https://critfc.org/fish-and-watersheds/fishery-science/data-resources/",
    "nwifc": "https://nwifc.org/",
}


def lookup(comid: int) -> TribalRegion | None:
    """Return the TribalRegion if `comid` falls within a masked area, else None."""
    raise NotImplementedError("M2 milestone: load USGS / BIA boundary layers, spatial join to NHDPlus HR.")
