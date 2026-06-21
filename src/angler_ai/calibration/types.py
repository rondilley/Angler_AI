"""Calibration types. CalibratedProbability is the structural invariant.

Every Prediction-Layer probability returned to a user-facing surface must be a
CalibratedProbability. This enforces FR-6.2 (interval mandatory) and FR-6.4
(no downstream surface can strip the interval) at the type level.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ProbabilityBasis:
    """How the raw probability was assembled. Surfaced to user (FR-6.3)."""

    cpue_derived_weight: float
    """Fraction (0.0-1.0) of the raw probability sourced from catch-per-unit-effort
    data. Drives whether hyperstability correction was applied."""

    fisheries_independent_weight: float
    """Fraction (0.0-1.0) from fisheries-independent surveys (electrofishing, etc.)."""

    sources: tuple[str, ...] = ()
    """e.g. ('SSN2_binomial', 'USGS_BRT_V2.0', 'PG-GNN'). Each tag must
    correspond to a Prediction-Layer model_id."""

    interval_kind: str = "sampling"
    """Semantic of the lower/upper interval on the CalibratedProbability:
       - 'sampling': uncertainty around a point estimate from finite survey
         data (BRT survey count, SSN2 binomial GLM, etc.). Typical width
         <= 0.10.
       - 'spatial_unmodeled': presence confirmed at HUC8 scale but reach-level
         suitability not modeled. Width >= 0.20 by construction. Used for
         NAS-derived non-native fallback priors.
       - 'climatological': baseline from long-term climatology (NorWeST mean
         August). Currently not surfaced as a SpeciesPrior interval.

    Downstream surfaces (map caption, narrative, JSON) MUST surface this so
    a brown-trout (sampling, 0.05 wide) and rainbow-trout (spatial_unmodeled,
    0.40 wide) side-by-side reads correctly. FR-6.4 invariant."""

    def __post_init__(self) -> None:
        total = self.cpue_derived_weight + self.fisheries_independent_weight
        if not 0.0 <= total <= 1.0 + 1e-6:
            raise ValueError(
                f"Probability basis weights must sum to <= 1.0, got {total}"
            )
        if self.interval_kind not in ("sampling", "spatial_unmodeled", "climatological"):
            raise ValueError(
                f"interval_kind must be one of "
                f"('sampling', 'spatial_unmodeled', 'climatological'), "
                f"got {self.interval_kind!r}"
            )


@dataclass(frozen=True, slots=True)
class CalibratedProbability:
    """A probability with a calibrated interval and full basis attribution.

    Constructed only by the Calibration Layer. No public API may return a bare
    float as a probability to user-facing surfaces; this dataclass is the invariant.
    """

    point: float
    """Calibrated point estimate. Range [0.0, 1.0]."""

    lower: float
    """Lower bound of the interval. Range [0.0, 1.0]."""

    upper: float
    """Upper bound of the interval. Range [0.0, 1.0]."""

    interval_confidence: float = 0.95
    """Stated confidence level, e.g. 0.95 for a 95% interval."""

    raw_point: float | None = None
    """Pre-calibration raw probability. None if no correction was applied."""

    hyperstability_beta_applied: float | None = None
    """The beta used in the hyperstability correction (Charbonneau 2025 default = 0.23).
    None if no correction was applied (basis fully fisheries-independent)."""

    basis: ProbabilityBasis = field(default_factory=lambda: ProbabilityBasis(0.0, 1.0))

    def __post_init__(self) -> None:
        for label, v in (("point", self.point), ("lower", self.lower), ("upper", self.upper)):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{label} must be in [0.0, 1.0], got {v}")
        if not (self.lower <= self.point <= self.upper):
            raise ValueError(
                f"Calibrated point {self.point} not within [{self.lower}, {self.upper}]"
            )
        if not 0.0 < self.interval_confidence < 1.0:
            raise ValueError(
                f"interval_confidence must be in (0, 1), got {self.interval_confidence}"
            )
