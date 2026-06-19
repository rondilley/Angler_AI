"""Calibration Layer (CL) - hyperstability-aware probability with structural intervals.

CalibratedProbability is the only return type from the Prediction Layer.
Downstream surfaces (CLI, HTTP API, map exports) cannot strip the interval.
This is enforced by the type system (FR-6.4).

See FR-6, design 5.7, research/04 (Charbonneau et al. 2025 TAFS, beta=0.23).
"""

from angler_ai.calibration.hyperstability import HyperstabilityConstant, apply_hyperstability
from angler_ai.calibration.types import CalibratedProbability, ProbabilityBasis

__all__ = [
    "CalibratedProbability",
    "HyperstabilityConstant",
    "ProbabilityBasis",
    "apply_hyperstability",
]
