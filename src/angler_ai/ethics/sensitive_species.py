"""Sensitive-species policy table. Seeded from the bundled CSV at first run."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class SuppressLevel(str, Enum):
    """Coarse-graining level for a sensitive species."""

    REACH = "reach"
    HUC12 = "huc12"
    HUC10 = "huc10"
    HUC8 = "huc8"

    def is_finer_than(self, other: SuppressLevel) -> bool:
        order = {SuppressLevel.HUC8: 0, SuppressLevel.HUC10: 1, SuppressLevel.HUC12: 2, SuppressLevel.REACH: 3}
        return order[self] > order[other]


@dataclass(frozen=True, slots=True)
class SensitiveSpeciesEntry:
    """One row in the sensitive_species table."""

    species_id: str
    common_name: str
    scientific_name: str
    status: str
    """e.g. 'ESA-threatened', 'ESA-candidate', 'state-soc'."""

    suppress_level: SuppressLevel
    rationale: str
    source_doc: str
    """Requirement id, e.g. 'CR-1.1'."""


_SEED_CSV = Path(__file__).parent / "data" / "sensitive_species_seed.csv"


def load_seed() -> tuple[SensitiveSpeciesEntry, ...]:
    """Load the bundled seed table. Used to populate the DB at first run."""
    if not _SEED_CSV.exists():
        return ()
    with _SEED_CSV.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return tuple(
            SensitiveSpeciesEntry(
                species_id=row["species_id"],
                common_name=row["common_name"],
                scientific_name=row["scientific_name"],
                status=row["status"],
                suppress_level=SuppressLevel(row["suppress_level"]),
                rationale=row["rationale"],
                source_doc=row["source_doc"],
            )
            for row in reader
        )
