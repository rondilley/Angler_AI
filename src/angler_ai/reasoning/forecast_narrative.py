"""LLM-generated narrative summary for a forecast PNG.

Takes the deterministic forecast outputs (top-scoring reaches, factor
breakdowns, source provenance) and asks the locally-loaded llama.cpp model
to produce a short narrative grounded in those numbers. The LLM does NOT
generate the probabilities; it summarizes the numeric outputs already
produced by the scoring layer.

The narrative is constrained by a strict system prompt: cite numbers
exactly, never invent species or reaches, surface the source tags. We
keep this lightweight (one chat turn, max 600 tokens) so it adds <5 s
per (water, species) on a typical RTX 5090.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from angler_ai.reasoning.agents import LLMRunner

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DayFactorSummary:
    """Per-day watershed-aggregate factor breakdown for the narrative."""

    date: str
    forecast_source: str
    median_score: float
    top_score: float
    mean_thermal_factor: float
    mean_water_temp_c: float | None
    flow_factor: float
    flow_source: str
    seasonal_factor: float
    mean_anomaly_factor: float


@dataclass(frozen=True, slots=True)
class NarrativeRequest:
    water_name: str
    species_scientific: str
    species_common: str
    huc8: str
    n_reaches: int
    n_temp_resolved: int
    n_anomalous_gauges: int
    top_score: float
    median_score: float
    forecast_window_start: str
    forecast_window_end: str
    sources_used: list[str]
    niche_citation_short: str
    factor_summary: str
    """One-line description of the strongest factor at the top reach
    (e.g. 'thermal=0.92 from NWIS_obs 13.2C, flow=1.0, seasonal=1.05')."""

    best_day: DayFactorSummary | None = None
    worst_day: DayFactorSummary | None = None
    species_t_optimum_c: float | None = None
    species_t_preferred_upper_c: float | None = None
    species_t_lethal_upper_c: float | None = None


_SYSTEM_PROMPT = """You are an honest fishing-intelligence analyst. You write \
short, factual summaries grounded entirely in the numbers and source tags \
the user provides. RULES:

1. Never invent species, reach counts, scores, dates, or citations.
2. Cite the actual data sources by name when stating a number.
3. Be brief: 6-9 sentences total. No bullet points. No emojis.
4. If a factor is 'not_modeled', say so explicitly; do not pretend.
5. Refer to the output as a "relative suitability index" or "suitability \
index" - NEVER as a "catch probability" or "probability of catch". This \
is a ranking in [0,1], not a calibrated probability.
6. If a numeric thermal_factor is present, do NOT say "thermal not modeled" \
- those two statements are contradictory and will be rejected.
7. Distinguish thermal_factor (the [0,1] multiplier) from base_p (the BRT \
presence probability) - they are separate fields in the factor breakdown.

STRUCTURE - REQUIRED:
The narrative MUST name the BEST day and the WORST day in the forecast \
window with their exact dates and median suitability values, and MUST \
explain WHY each is best or worst by referencing the specific factor \
differences:
  - water temperature on each day vs the species thermal niche \
    (T_optimum, T_preferred_upper, T_lethal_upper)
  - flow factor on each day and its source tag
  - seasonal factor for the month
  - anomaly factor

End with one sentence on the honest LIMIT (e.g., "Note: thermal \
suitability was modeled for X of Y reaches.")."""


def _format_day_summary(label: str, d: DayFactorSummary | None) -> str:
    if d is None:
        return f"{label}: not available"
    t_str = f"{d.mean_water_temp_c:.1f} C" if d.mean_water_temp_c is not None else "unknown"
    return (
        f"{label}: {d.date} ({d.forecast_source}) | median index {d.median_score:.3f}, "
        f"top {d.top_score:.3f} | mean water temp {t_str}, "
        f"thermal_factor {d.mean_thermal_factor:.2f}, flow_factor {d.flow_factor:.2f} "
        f"({d.flow_source}), seasonal_factor {d.seasonal_factor:.2f}, "
        f"anomaly_factor {d.mean_anomaly_factor:.2f}"
    )


def build_user_prompt(req: NarrativeRequest) -> str:
    niche_summary = []
    if req.species_t_optimum_c is not None:
        niche_summary.append(f"T_optimum={req.species_t_optimum_c:.1f}C")
    if req.species_t_preferred_upper_c is not None:
        niche_summary.append(f"T_preferred_upper={req.species_t_preferred_upper_c:.1f}C")
    if req.species_t_lethal_upper_c is not None:
        niche_summary.append(f"T_lethal_upper={req.species_t_lethal_upper_c:.1f}C")
    niche_line = (
        f"Species thermal niche: {', '.join(niche_summary)}."
        if niche_summary else "Species thermal niche: not in registry."
    )

    lines = [
        f"Water: {req.water_name} (HUC8 {req.huc8}).",
        f"Species: {req.species_common} ({req.species_scientific}).",
        f"Forecast window: {req.forecast_window_start} to {req.forecast_window_end}.",
        f"Reaches scored: {req.n_reaches}.",
        f"Reaches with observed/interpolated water temperature: {req.n_temp_resolved} of {req.n_reaches}.",
        f"USGS gauges flagged as flow-anomalous: {req.n_anomalous_gauges}.",
        f"Suitability index range (NOT a probability) over the WHOLE window: top reach {req.top_score:.3f}, median {req.median_score:.3f} on a 0-1 relative scale.",
        f"Window top-reach factor breakdown: {req.factor_summary}.",
        niche_line,
        f"Species thermal niche source: {req.niche_citation_short}.",
        "",
        "BEST AND WORST DAY DETAIL (use these directly in the narrative):",
        _format_day_summary("BEST_DAY", req.best_day),
        _format_day_summary("WORST_DAY", req.worst_day),
        "",
        f"Sources used: {', '.join(req.sources_used)}.",
        "",
        "Write a 6-9 sentence summary. You MUST: (a) name the BEST day "
        "and the WORST day with their exact dates and median index values, "
        "(b) explain WHY each is best/worst by comparing mean water temp "
        "against the species thermal niche, and by citing any change in "
        "flow_factor, seasonal_factor, or anomaly_factor between the two days, "
        "(c) close with one sentence stating the modeling LIMIT. Refer to "
        "the output as a 'suitability index', NEVER as a 'catch probability'.",
    ]
    return "\n".join(lines)


def generate_narrative(
    llm: LLMRunner, request: NarrativeRequest, *, max_tokens: int = 400,
) -> str:
    """Generate the narrative string. Returns empty string on failure."""
    user_prompt = build_user_prompt(request)
    try:
        text = llm.chat(
            system=_SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=max_tokens,
            temperature=0.3,
        )
    except Exception as exc:  # noqa: BLE001 - we surface and continue
        log.warning("Narrative generation failed for %s/%s: %s",
                    request.water_name, request.species_common, exc)
        return ""
    return text.strip()
