"""USGS BRT v2.0 species-presence priors (FR-5.2)."""

from __future__ import annotations

from angler_ai.calibration.types import CalibratedProbability


def get_species_priors(comid: int, top_k: int = 10) -> list[tuple[str, CalibratedProbability]]:
    """Return top-K species presence priors for a reach.

    USGS BRT was trained on fisheries-independent data, so
    `cpue_derived_weight=0.0` in the CalibratedProbability basis. The interval
    is constructed from the BRT cross-validation deviance.
    """
    raise NotImplementedError("M3 milestone: query brt_priors via xwalk_v2_to_hr; build CalibratedProbability.")


def get_species_probability(comid: int, species: str) -> CalibratedProbability | None:
    """Return BRT prior for a single species at a reach, or None if unmodeled."""
    raise NotImplementedError("M3 milestone.")
