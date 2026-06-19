"""SSN2 reach-level catch probability (FR-5.1).

v0 status: STUB. The v0 calibration path uses USGS BRT priors directly,
wrapped in CalibratedProbability via `prediction.species_priors`. This
module preserves the future SSN2 R-bridge signatures so v1 can swap in a
true spatial-stream-network model without changing callers.

v1 plan:
    - R 4.3+ installed locally
    - SSN2 (USEPA, https://github.com/USEPA/SSN2) installed via install.packages
    - SSNbler installed for NHDPlus HR preprocessing
    - Either rpy2 or subprocess R bridge
    - Train ssn_glm(family='binomial') against observed catch data
      (state electrofishing CPUE + Anglers Atlas / MyCatch where licensed)
    - predict() returns native 95% prediction intervals over the stream
      network, which we feed into CalibratedProbability directly
    - hyperstability correction stays the M4 default (BC steelhead beta=0.23)
"""

from __future__ import annotations

import logging

from angler_ai.calibration.types import CalibratedProbability

log = logging.getLogger(__name__)


def predict_reach_catch_probability(
    comid: int,
    species: str,
    date: str,  # ISO date
    feature_store_path: str,
) -> CalibratedProbability:
    """Predict per-reach catch probability for a species/date.

    v0: NotImplementedError - callers should use
    `prediction.species_priors.species_priors_for_reach` which is BRT-based.

    v1 will implement the SSN2 ssn_glm(binomial) path with R, returning the
    same CalibratedProbability shape.
    """
    raise NotImplementedError(
        "v1 milestone. v0 uses prediction.species_priors (BRT-based). "
        "v1 swap requires R + SSN2 + SSNbler + labeled training data."
    )
