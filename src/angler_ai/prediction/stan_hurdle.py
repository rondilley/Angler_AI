"""Bayesian hurdle-gamma harvest probability (FR-5.5).

arXiv 2503.17293 framework (Gulf of Mexico gag grouper, 2025 NAJFM submission).
Adapted for trout openers, salmon runs, put-and-take stocking windows. Uses
CmdStanPy or PyStan.
"""

from __future__ import annotations

from angler_ai.calibration.types import CalibratedProbability


def harvest_probability(
    state: str,
    species: str,
    season_window: tuple[str, str],
    effort_compression: float | None = None,
) -> CalibratedProbability:
    """Predict probability of meeting a per-trip harvest threshold.

    Returns a CalibratedProbability with the 80% posterior predictive interval
    converted to the standard 95% via re-sampling, or the native 80% with
    `interval_confidence=0.8`.
    """
    raise NotImplementedError(
        "M6 milestone: Stan model per arXiv 2503.17293, posterior-predictive "
        "sampling, route through Calibration Layer."
    )
