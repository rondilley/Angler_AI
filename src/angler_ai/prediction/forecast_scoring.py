"""Forecast scoring: combine BRT calibrated prior x species thermal niche x
flow modifier x seasonal-activity factor x flow-anomaly factor into a
per-day RELATIVE SUITABILITY INDEX per reach per species.

NOMENCLATURE (post multi-AI review, 2026-06-17):
This output is NOT a calibrated catch probability. It is a relative
suitability index in [0, 1] that ranks reach-day combinations within a
single map. The five factors are heterogeneous (Gaussian density,
discharge ratio, 3-bin precip step, monthly heuristic, saturating z-score)
and are multiplied as if independent even though some pairs are correlated
(thermal+seasonal both temperature-driven, flow+anomaly both discharge-
driven). Until M4 catch-data validation produces a defensible probability
calibration, downstream surfaces MUST call this a "suitability index", not
a probability. The interval is propagated from the underlying BRT
CalibratedProbability so FR-6.4 invariants are preserved end-to-end.

The scoring chain at v0:
    base_p          USGS BRT v2.0 presence probability (calibrated; v0 has
                    cpue_weight=0 so equals the raw BRT predict_prob)
    x thermal       species-niche bell evaluated at PER-DAY modeled water temp
                    (NWIS_obs > NWIS_interp > NWIS_air_projected > not_modeled)
    x flow          NWS+OpenMeteo precip-probability turbidity proxy
                    OR real discharge ratio if NWIS_obs flow ingested
    x seasonal      species monthly activity multiplier from published phenology
                    (CLAMPED TO 1.0 to keep the product in [0, 1])
    x anomaly       additional attenuation when the nearest gauge shows a
                    flow z-score anomaly

Every factor is paired with a source tag. Missing data returns 1.0
(neutral) tagged 'not_modeled' - we never substitute a value. The
suitability_index is clamped to [0, 1] explicitly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

from angler_ai.calibration.types import CalibratedProbability
from angler_ai.ingest.noaa_forecast import DailyForecast
from angler_ai.prediction.flow_anomaly import GaugeAnomaly, anomaly_factor
from angler_ai.prediction.seasonal_activity import seasonal_factor_for_date
from angler_ai.prediction.species_priors import SpeciesPrior
from angler_ai.prediction.thermal_niches import ThermalNiche, get_niche
from angler_ai.prediction.water_temp_model import ProjectedDailyTemp

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FactorBreakdown:
    """Per-factor numeric value + source tag. Surfaces 'not_modeled' explicitly."""

    base_p: float
    base_source: str

    thermal_factor: float
    thermal_source: str
    thermal_input_c: float | None

    flow_factor: float
    flow_source: str
    precip_probability_pct: int | None

    seasonal_factor: float
    seasonal_source: str

    anomaly_factor: float
    anomaly_source: str


@dataclass(frozen=True, slots=True)
class DailyScore:
    """One day's relative suitability index for one reach + species.

    The suitability_index is a clamped multiplicative composite in [0, 1].
    It is NOT a calibrated catch probability. The 95% interval is
    propagated from the underlying BRT CalibratedProbability via the same
    deterministic factor chain so FR-6.4 (no surface may strip the
    interval) is preserved at the type level. Field name uses
    `suitability_index` not `score` per multi-AI review STAT-02 / PV-6.

    For backward compatibility callers can access .score as an alias for
    .suitability_index; new code should use the suitability_index name.
    """

    comid: int
    species: str
    score_date: date
    suitability_index: float
    """Clamped multiplicative composite in [0, 1]. NOT a probability."""

    suitability_lower: float
    """Lower bound of the propagated 95% interval, clamped to [0, 1]."""

    suitability_upper: float
    """Upper bound of the propagated 95% interval, clamped to [0, 1]."""

    interval_confidence: float
    """95% by construction; surfaced so downstream knows the level."""

    factors: FactorBreakdown

    @property
    def score(self) -> float:
        """Deprecated alias for `suitability_index`. Will be removed at M9."""
        return self.suitability_index


def flow_factor_from_forecast(
    forecast: DailyForecast | None,
) -> tuple[float, str, int | None]:
    """3-bin turbidity proxy from precip-probability when discharge data is absent."""
    if forecast is None or forecast.precip_probability_pct is None:
        return (1.0, "not_modeled", None)
    p = forecast.precip_probability_pct
    if p <= 30:
        return (1.0, forecast.source, p)
    if p <= 70:
        return (0.85, forecast.source, p)
    return (0.6, forecast.source, p)


def thermal_factor_from_temp_c(
    temp_c: float | None, temp_source: str, niche: ThermalNiche | None,
) -> tuple[float, str, float | None]:
    if niche is None:
        return (1.0, "not_modeled:no_niche", None)
    if temp_c is None or temp_source == "not_modeled":
        return (1.0, "not_modeled:no_temperature", None)
    factor = niche.suitability(temp_c)
    return (factor, f"niche:{niche.scientific_name} x temp:{temp_source}", temp_c)


def _clamp_unit(x: float) -> float:
    """Clamp a value to [0.0, 1.0]."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def score_reach_daily(
    *,
    prior: SpeciesPrior,
    daily_temp: ProjectedDailyTemp | None,
    daily_forecast: DailyForecast | None,
    daily_discharge_factor: tuple[float, str] | None,
    gauge_anomaly: GaugeAnomaly | None,
    score_date: date,
) -> DailyScore:
    """Compute the daily suitability index for one (reach, species, day).

    Uses the per-day projected water temp (NWIS_obs / interp / air_projected),
    real discharge factor if supplied (overrides the precip-prob proxy), the
    seasonal activity factor from published phenology (clamped at 1.0 to keep
    the product in [0, 1]), and the flow z-score anomaly attenuation if the
    gauge is flagged. The 95% calibrated interval is propagated through the
    same factor chain (FR-6.4 invariant).
    """
    niche = get_niche(prior.species)
    if daily_temp is not None:
        thermal_f, thermal_src, thermal_in = thermal_factor_from_temp_c(
            daily_temp.temperature_c, daily_temp.source, niche,
        )
    else:
        thermal_f, thermal_src, thermal_in = (1.0, "not_modeled:no_temperature", None)

    if daily_discharge_factor is not None:
        flow_f, flow_src = daily_discharge_factor
        precip_p: int | None = None
    else:
        flow_f, flow_src, precip_p = flow_factor_from_forecast(daily_forecast)

    seasonal_f_raw, seasonal_src, _ = seasonal_factor_for_date(
        prior.species, score_date.month,
    )
    # Clamp seasonal at 1.0 so the multiplicative chain stays in [0, 1].
    # (multi-AI review STAT-03: seasonal_factor up to 1.10 could push the
    # product above 1.0 without a clamp). The source tag preserves the
    # raw value so consumers can see when capping happened.
    if seasonal_f_raw > 1.0:
        seasonal_f = 1.0
        seasonal_src = seasonal_src + f"|capped_from_{seasonal_f_raw:.2f}"
    else:
        seasonal_f = seasonal_f_raw

    if gauge_anomaly is not None:
        anom_f, anom_src = anomaly_factor(gauge_anomaly.z_score)
    else:
        anom_f, anom_src = (1.0, "not_modeled:no_anomaly_data")

    cp = prior.probability
    factor_product = thermal_f * flow_f * seasonal_f * anom_f
    suitability_index = _clamp_unit(cp.point * factor_product)
    suitability_lower = _clamp_unit(cp.lower * factor_product)
    suitability_upper = _clamp_unit(cp.upper * factor_product)
    # Order-preserving: clamp can collapse near 0/1 but never invert.
    if suitability_lower > suitability_index:
        suitability_lower = suitability_index
    if suitability_upper < suitability_index:
        suitability_upper = suitability_index

    return DailyScore(
        comid=prior.comid,
        species=prior.species,
        score_date=score_date,
        suitability_index=suitability_index,
        suitability_lower=suitability_lower,
        suitability_upper=suitability_upper,
        interval_confidence=cp.interval_confidence,
        factors=FactorBreakdown(
            base_p=cp.point,
            base_source="USGS_BRT_V2.0 (presence probability; hyperstability not_applied at v0)",
            thermal_factor=thermal_f,
            thermal_source=thermal_src,
            thermal_input_c=thermal_in,
            flow_factor=flow_f,
            flow_source=flow_src,
            precip_probability_pct=precip_p,
            seasonal_factor=seasonal_f,
            seasonal_source=seasonal_src,
            anomaly_factor=anom_f,
            anomaly_source=anom_src,
        ),
    )


def score_reach_over_window(
    *,
    prior: SpeciesPrior,
    per_day_temps: dict[date, ProjectedDailyTemp],
    forecast_window: list[DailyForecast],
    per_day_discharge_factor: dict[date, tuple[float, str]] | None = None,
    gauge_anomaly: GaugeAnomaly | None = None,
) -> list[DailyScore]:
    """Score one reach across each day in the forecast window.

    For days where no per-day temp is available but other days in the
    window have one, use the nearest-by-date temp so the entire window's
    thermal factor is anchored on real (observed/interpolated/projected)
    data rather than silently defaulting to 1.0 on the gap days.
    """
    sorted_temp_dates = sorted(per_day_temps.keys()) if per_day_temps else []
    out: list[DailyScore] = []
    for fd in forecast_window:
        d = fd.forecast_date
        dt = per_day_temps.get(d)
        if dt is None and sorted_temp_dates:
            # Nearest by date (carry-forward / carry-back)
            nearest = min(sorted_temp_dates, key=lambda x: abs((d - x).days))
            dt = per_day_temps.get(nearest)
        disc = None if per_day_discharge_factor is None else per_day_discharge_factor.get(d)
        out.append(score_reach_daily(
            prior=prior,
            daily_temp=dt,
            daily_forecast=fd,
            daily_discharge_factor=disc,
            gauge_anomaly=gauge_anomaly,
            score_date=d,
        ))
    return out


def max_score_over_window(
    *,
    prior: SpeciesPrior,
    per_day_temps: dict[date, ProjectedDailyTemp],
    forecast_window: list[DailyForecast],
    per_day_discharge_factor: dict[date, tuple[float, str]] | None = None,
    gauge_anomaly: GaugeAnomaly | None = None,
) -> DailyScore | None:
    daily = score_reach_over_window(
        prior=prior,
        per_day_temps=per_day_temps,
        forecast_window=forecast_window,
        per_day_discharge_factor=per_day_discharge_factor,
        gauge_anomaly=gauge_anomaly,
    )
    if not daily:
        return None
    return max(daily, key=lambda d: d.suitability_index)
