"""Prediction interval construction. Stub - M4 implements conformal/isotonic
once a verified recipe is available for SSN2 outputs (open question carried in
research/04, design 9).
"""

from __future__ import annotations


def naive_beta_interval(point: float, n_effective: float, confidence: float = 0.95) -> tuple[float, float]:
    """Beta-distribution interval as a placeholder until conformal lands at M4.

    Treats `point` as a sample mean from `n_effective` Bernoulli draws.
    Replace with conformal prediction or isotonic regression at M4 once the
    research question on SSN2-compatible calibration recipes is resolved.

    Args:
        point: Calibrated point estimate in [0, 1].
        n_effective: Effective sample size driving the interval width.
        confidence: Stated coverage. Default 0.95.

    Returns:
        (lower, upper) bounds in [0, 1].
    """
    if n_effective <= 0:
        return (0.0, 1.0)
    # Wilson-style approximation; replace at M4.
    from math import sqrt
    z = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645
    denom = 1.0 + z * z / n_effective
    centre = (point + z * z / (2.0 * n_effective)) / denom
    margin = z * sqrt((point * (1.0 - point) + z * z / (4.0 * n_effective)) / n_effective) / denom
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    if lower > point:
        lower = max(0.0, point - 1e-6)
    if upper < point:
        upper = min(1.0, point + 1e-6)
    return (lower, upper)
