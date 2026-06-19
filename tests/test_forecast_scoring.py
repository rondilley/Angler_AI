"""Tests for the multi-factor suitability index scoring chain.

Locks in the Tier 1 invariants from the 2026-06-17 multi-AI review:
  - DailyScore.suitability_index is in [0, 1]
  - DailyScore propagates a 95% interval (suitability_lower/upper) so
    FR-6.4 (no surface may strip the CalibratedProbability interval) is
    preserved end-to-end
  - seasonal_factor is clamped at 1.0 in the product so the chain cannot
    produce values > 1.0
  - The interval bounds bracket the suitability_index
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

from angler_ai.calibration.types import (
    CalibratedProbability, ProbabilityBasis,
)
from angler_ai.ingest.noaa_forecast import DailyForecast
from angler_ai.prediction.forecast_scoring import score_reach_daily
from angler_ai.prediction.species_priors import SpeciesPrior
from angler_ai.prediction.water_temp_model import ProjectedDailyTemp


def _make_prior(point=0.95, lower=0.85, upper=0.99) -> SpeciesPrior:
    cp = CalibratedProbability(
        point=point, lower=lower, upper=upper, interval_confidence=0.95,
        raw_point=point, hyperstability_beta_applied=0.23,
        basis=ProbabilityBasis(
            cpue_derived_weight=0.0, fisheries_independent_weight=1.0,
            sources=("USGS_BRT_V2.0",),
        ),
    )
    return SpeciesPrior(
        comid=42, species="Salmo trutta", common_name="brown trout",
        probability=cp, v2_join_method="reachcode_exact",
    )


def test_suitability_index_is_in_unit_interval() -> None:
    """The 5-factor product must never escape [0, 1] (multi-AI STAT-03)."""
    prior = _make_prior(point=0.99, lower=0.95, upper=0.999)
    # Force peak seasonal month (May for brown trout, multiplier 1.10).
    score = score_reach_daily(
        prior=prior,
        daily_temp=ProjectedDailyTemp(
            comid=42, forecast_date=date(2026, 5, 15),
            temperature_c=13.9, source="NWIS_obs",
        ),
        daily_forecast=DailyForecast(
            forecast_date=date(2026, 5, 15),
            high_c=22.0, low_c=10.0, precip_probability_pct=10,
            short_forecast="Sunny", source="NWS_NDFD_daily",
        ),
        daily_discharge_factor=None,
        gauge_anomaly=None,
        score_date=date(2026, 5, 15),
    )
    assert 0.0 <= score.suitability_index <= 1.0
    assert 0.0 <= score.suitability_lower <= 1.0
    assert 0.0 <= score.suitability_upper <= 1.0


def test_interval_propagated_through_factor_chain() -> None:
    """FR-6.4: the lower/upper bounds must travel with the point estimate
    so no downstream surface can strip them. The propagated interval must
    bracket the index."""
    prior = _make_prior(point=0.5, lower=0.3, upper=0.7)
    score = score_reach_daily(
        prior=prior,
        daily_temp=ProjectedDailyTemp(
            comid=42, forecast_date=date(2026, 6, 18),
            temperature_c=15.0, source="NWIS_interp",
        ),
        daily_forecast=DailyForecast(
            forecast_date=date(2026, 6, 18),
            high_c=18.0, low_c=8.0, precip_probability_pct=20,
            short_forecast="Sunny", source="NWS_NDFD_daily",
        ),
        daily_discharge_factor=None,
        gauge_anomaly=None,
        score_date=date(2026, 6, 18),
    )
    assert score.suitability_lower <= score.suitability_index <= score.suitability_upper, (
        f"interval violation: {score.suitability_lower} <= {score.suitability_index} "
        f"<= {score.suitability_upper}"
    )
    assert score.interval_confidence == 0.95


def test_seasonal_factor_above_1_is_capped() -> None:
    """STAT-03: seasonal_factor up to 1.10 must not push the product > 1.0.
    The seasonal_source tag must record that capping occurred."""
    prior = _make_prior(point=1.0, lower=1.0, upper=1.0)
    # May for brown trout has monthly multiplier 1.10.
    score = score_reach_daily(
        prior=prior,
        daily_temp=ProjectedDailyTemp(
            comid=42, forecast_date=date(2026, 5, 15),
            temperature_c=13.9, source="NWIS_obs",
        ),
        daily_forecast=None,
        daily_discharge_factor=None,
        gauge_anomaly=None,
        score_date=date(2026, 5, 15),
    )
    assert score.factors.seasonal_factor == 1.0
    assert "capped_from_1.10" in score.factors.seasonal_source
    assert score.suitability_index <= 1.0


def test_deprecated_score_alias_returns_suitability_index() -> None:
    """Back-compat: `DailyScore.score` is an alias kept until M9."""
    prior = _make_prior()
    score = score_reach_daily(
        prior=prior, daily_temp=None, daily_forecast=None,
        daily_discharge_factor=None, gauge_anomaly=None,
        score_date=date(2026, 6, 18),
    )
    assert score.score == score.suitability_index


def test_base_p_source_says_hyperstability_not_applied() -> None:
    """Multi-AI STAT-01: hyperstability is intentionally not applied to BRT
    presence probability at v0. The base_source string must surface this."""
    prior = _make_prior()
    score = score_reach_daily(
        prior=prior, daily_temp=None, daily_forecast=None,
        daily_discharge_factor=None, gauge_anomaly=None,
        score_date=date(2026, 6, 18),
    )
    assert "hyperstability not_applied" in score.factors.base_source
