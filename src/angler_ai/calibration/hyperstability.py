"""Hyperstability correction for CPUE-derived probabilities (FR-6.1).

Charbonneau et al. 2025 TAFS quantified hyperstability at beta = 0.23 across 14
BC steelhead streams over 47 years using fisheries-independent baselines. A 50%
true population decline produces only a 40% CPUE decline; a 90% decline
produces only a 77% CPUE decline. Mechanism mirrors Erisman et al. 2011
"illusion of plenty" across walleye, bass, and trout.

Reference value is 0.23. Users may override via config but the override is
logged. There is no global "disable hyperstability" path (CR-1.5 by analogy
and CR-4.2).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HyperstabilityConstant:
    """A documented hyperstability constant with provenance."""

    beta: float
    """The exponent in CPUE = q * N^beta. Lower beta = stronger hyperstability."""

    source: str
    """Citation, e.g. 'Charbonneau et al. 2025 TAFS, BC steelhead 14 streams 1972-2019'."""

    geography: str
    """Where the constant was measured, e.g. 'British Columbia steelhead streams'."""

    standard_error: float | None = None


CHARBONNEAU_2025_BC_STEELHEAD = HyperstabilityConstant(
    beta=0.23,
    source="Charbonneau et al. 2025 TAFS 154(4):339",
    geography="British Columbia steelhead streams, 14 sites 1972-2019",
    standard_error=0.05,
)


def apply_hyperstability(
    raw_p: float,
    cpue_weight: float,
    constant: HyperstabilityConstant = CHARBONNEAU_2025_BC_STEELHEAD,
) -> float:
    """Downweight a CPUE-derived raw probability.

    For the CPUE-derived component, apply the hyperstability correction
    `p_corrected = p_raw ^ (1 / beta)` clamped to [0, 1]. For the
    fisheries-independent component, leave the raw value unchanged. Combine
    by weighted average.

    NOTE: This is the v0 implementation. M4 will validate the formulation
    against held-out PA data and revise. The cpue_weight argument enables
    blended sources without losing the calibration signal.

    Args:
        raw_p: Raw probability in [0, 1].
        cpue_weight: Fraction (0..1) of raw_p sourced from CPUE-derived signals.
        constant: Hyperstability constant. Default is BC steelhead 0.23.

    Returns:
        Calibrated probability in [0, 1]. Always logged with raw -> calibrated.
    """
    if not 0.0 <= raw_p <= 1.0:
        raise ValueError(f"raw_p must be in [0, 1], got {raw_p}")
    if not 0.0 <= cpue_weight <= 1.0:
        raise ValueError(f"cpue_weight must be in [0, 1], got {cpue_weight}")
    if cpue_weight == 0.0:
        log.debug("apply_hyperstability: cpue_weight=0; raw_p unchanged at %s", raw_p)
        return raw_p
    # CPUE = q * N^beta  =>  estimate N proportional to CPUE^(1/beta).
    # We apply the correction on the unit interval as a monotone transform.
    cpue_component = math.pow(max(raw_p, 1e-9), 1.0 / constant.beta)
    cpue_component = min(max(cpue_component, 0.0), 1.0)
    fi_component = raw_p
    p_cal = cpue_weight * cpue_component + (1.0 - cpue_weight) * fi_component
    p_cal = min(max(p_cal, 0.0), 1.0)
    log.debug(
        "hyperstability: raw=%.4f cpue_weight=%.2f beta=%.3f -> calibrated=%.4f source=%s",
        raw_p, cpue_weight, constant.beta, p_cal, constant.source,
    )
    return p_cal
