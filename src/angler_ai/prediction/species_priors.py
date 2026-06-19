"""Reach-level species probability with calibrated intervals (M4).

v0 implementation: BRT priors as the underlying probability substrate, wrapped
in CalibratedProbability with mandatory hyperstability correction (FR-6.1) and
prediction intervals derived from V2-join confidence + species-level prevalence
as data-sparsity proxies.

v1 will swap the BRT-based prior for a true SSN2 ssn_glm(family=binomial)
fitted against observed catch data; the CalibratedProbability return type and
public function signatures stay stable so the swap is internal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from angler_ai.calibration.hyperstability import (
    CHARBONNEAU_2025_BC_STEELHEAD,
    apply_hyperstability,
)
from angler_ai.calibration.intervals import naive_beta_interval
from angler_ai.calibration.types import CalibratedProbability, ProbabilityBasis
from angler_ai.features.store import FeatureStore

log = logging.getLogger(__name__)


# BRT predictions are unit-interval PRESENCE probabilities from a binomial
# Boosted Regression Tree fit to electrofishing/seine survey data. The
# Charbonneau 2025 hyperstability constant (beta=0.23) is calibrated for
# the abundance-from-CPUE inversion CPUE = q * N^beta, NOT for unit-interval
# presence probabilities. Applying p^(1/beta) = p^4.347 to a presence
# probability is a statistical category error: a raw BRT prior of 0.5 would
# become 0.049 (98% reduction), without any published justification.
#
# v0 multi-AI review (2026-06-17) flagged this as STAT-01 critical. Until
# M4 catch-data validation produces a defensible calibration, we set the
# CPUE-derived weight to ZERO, leaving the raw BRT probability unchanged.
# The CalibratedProbability still carries the basis tag so downstream
# consumers know hyperstability was considered and explicitly skipped.
#
# v1 path: replace BRT prior with SSN2 ssn_glm(family=binomial) trained on
# observed CPUE; at that point a CPUE-derived weight > 0 will be justified.
BRT_CPUE_DERIVED_WEIGHT = 0.0


@dataclass(frozen=True, slots=True)
class SpeciesPrior:
    """One row for the map exporter / species CLI."""

    comid: int
    species: str
    common_name: str | None
    probability: CalibratedProbability
    v2_join_method: str
    """'reachcode_exact', 'huc10_proximity', etc. - from xwalk_v2_to_hr.method"""


def species_priors_for_reach(
    store: FeatureStore,
    comid_hr: int,
    species_scientific: str | None = None,
    top_k: int | None = None,
) -> list[SpeciesPrior]:
    """Return calibrated species priors for one HR reach.

    Args:
        store: FeatureStore.
        comid_hr: NHDPlus HR COMID.
        species_scientific: filter to one species; if None, return top_k by
            probability across all species at this reach.
        top_k: limit ranking. Ignored if species_scientific is set.

    Returns:
        List of SpeciesPrior in descending raw probability order.
    """
    conn = store.connect()
    if species_scientific:
        query = """
            SELECT p.comid AS v2_comid, p.species, p.probability,
                   x.method, x.confidence, s.common_name, s.prevalence
            FROM xwalk_v2_to_hr x
            JOIN brt_priors p ON p.comid = x.comid_v2
            LEFT JOIN brt_species s ON s.scientific_name = p.species
            WHERE x.comid_hr = ?
              AND p.species = ?
              AND p.model_version = 'USGS_BRT_V2.0'
            ORDER BY p.probability DESC
        """
        params: list = [comid_hr, species_scientific]
    else:
        limit = int(top_k or 10)
        query = """
            SELECT v2_comid, species, probability, method, confidence,
                   common_name, prevalence
            FROM (
                SELECT p.comid AS v2_comid, p.species, p.probability,
                       x.method, x.confidence,
                       s.common_name, s.prevalence,
                       ROW_NUMBER() OVER (PARTITION BY p.species
                                          ORDER BY p.probability DESC) AS rn
                FROM xwalk_v2_to_hr x
                JOIN brt_priors p ON p.comid = x.comid_v2
                LEFT JOIN brt_species s ON s.scientific_name = p.species
                WHERE x.comid_hr = ?
                  AND p.model_version = 'USGS_BRT_V2.0'
            ) t
            WHERE rn = 1
            ORDER BY probability DESC
            LIMIT ?
        """
        params = [comid_hr, limit]

    out: list[SpeciesPrior] = []
    for v2_comid, species, raw_p, method, confidence, common_name, prevalence in conn.execute(query, params).fetchall():
        cp = _calibrate(
            raw_p=float(raw_p),
            v2_join_method=method,
            v2_join_confidence=float(confidence),
            prevalence=float(prevalence) if prevalence is not None else None,
            species=species,
        )
        out.append(SpeciesPrior(
            comid=comid_hr,
            species=species,
            common_name=common_name,
            probability=cp,
            v2_join_method=method,
        ))
    return out


def species_priors_for_geometry(
    store: FeatureStore,
    species_scientific: str,
    *,
    huc8: str | None = None,
    state: str | None = None,
    limit: int | None = None,
) -> list[tuple[SpeciesPrior, str]]:
    """Return calibrated priors for one species across many reaches.

    Returns:
        List of (SpeciesPrior, geometry_wkt). Geometry is the reach's
        MULTILINESTRING WKT.
    """
    conn = store.connect()
    where = []
    params: list = []
    if huc8:
        where.append("r.huc8 = ?")
        params.append(huc8)
    if state:
        from angler_ai.ingest.nhdplus_hr import _state_fips
        fips = _state_fips(state)
        if fips:
            where.append("r.state_fips = ?")
            params.append(fips)
    where_sql = (" AND " + " AND ".join(where)) if where else ""
    sql = f"""
        SELECT r.comid, ST_AsText(r.geometry) AS geom_wkt,
               t.species, t.max_p, t.method, t.confidence,
               s.common_name, s.prevalence
        FROM reaches r
        JOIN (
            SELECT x.comid_hr, p.species,
                   MAX(p.probability) AS max_p,
                   ANY_VALUE(x.method) AS method,
                   ANY_VALUE(x.confidence) AS confidence
            FROM xwalk_v2_to_hr x
            JOIN brt_priors p ON p.comid = x.comid_v2
            WHERE p.species = ?
              AND p.model_version = 'USGS_BRT_V2.0'
            GROUP BY x.comid_hr, p.species
        ) t ON t.comid_hr = r.comid
        LEFT JOIN brt_species s ON s.scientific_name = t.species
        WHERE r.source = 'NHDPlus_HR'
          {where_sql}
        ORDER BY t.max_p DESC
    """
    params = [species_scientific] + params
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows: list[tuple[SpeciesPrior, str]] = []
    for comid, geom_wkt, species, max_p, method, confidence, common_name, prevalence in conn.execute(sql, params).fetchall():
        cp = _calibrate(
            raw_p=float(max_p),
            v2_join_method=method,
            v2_join_confidence=float(confidence),
            prevalence=float(prevalence) if prevalence is not None else None,
            species=species,
        )
        sp = SpeciesPrior(
            comid=comid,
            species=species,
            common_name=common_name,
            probability=cp,
            v2_join_method=method,
        )
        rows.append((sp, geom_wkt or ""))
    return rows


def _calibrate(
    raw_p: float,
    v2_join_method: str,
    v2_join_confidence: float,
    prevalence: float | None,
    species: str,
) -> CalibratedProbability:
    """Derive a calibrated probability with a 95% prediction interval.

    Args:
        raw_p: BRT predict_prob.
        v2_join_method: 'reachcode_exact' or 'huc10_proximity' from xwalk.
        v2_join_confidence: 1.0 for exact, 0.5 for proximity.
        prevalence: per-species BRT prevalence (presences / total). Used as
            an effective-sample-size proxy: lower prevalence => fewer
            positive observations => wider interval.
        species: scientific name; surfaces in basis.sources.

    Returns:
        CalibratedProbability with:
            - point: raw BRT predict_prob (v0 - see BRT_CPUE_DERIVED_WEIGHT)
            - lower, upper: 95% interval from Wilson approximation, widened
              when v2 join is proximity-only
            - raw_point: BRT predict_prob
            - hyperstability_beta_applied: 0.23 (constant on record, but the
              correction is currently NOT applied at cpue_weight=0)
            - basis: ProbabilityBasis with cpue_derived_weight=0.0 at v0
    """
    constant = CHARBONNEAU_2025_BC_STEELHEAD
    point = apply_hyperstability(
        raw_p=raw_p,
        cpue_weight=BRT_CPUE_DERIVED_WEIGHT,
        constant=constant,
    )

    # Effective-sample-size proxy:
    #   - reachcode_exact: 200 baseline (the underlying BRT survey count per
    #     V2 reach is ~120-400; 200 is a defensible middle).
    #   - huc10_proximity: 50 (we're inferring across HR reaches that V2
    #     does not directly model; widen the interval to reflect that).
    #   - Further reduce by prevalence: rare species have fewer positive
    #     observations driving the model fit.
    base_n = 200 if v2_join_method == "reachcode_exact" else 50
    if prevalence is not None and prevalence > 0:
        n_effective = base_n * max(prevalence, 0.05)
    else:
        n_effective = base_n * 0.5

    lower, upper = naive_beta_interval(
        point=point,
        n_effective=n_effective,
        confidence=0.95,
    )

    # v0 honest tagging: cpue_derived_weight=0 means the Charbonneau
    # hyperstability correction is on record but NOT applied to the BRT
    # presence probability (multi-AI review STAT-01 - applying p^(1/0.23)
    # to a unit-interval presence probability is a category error). The
    # source tag reflects the actual treatment.
    hyper_tag = (
        f"hyperstability:not_applied(reason=BRT_presence_probability_v0)"
        if BRT_CPUE_DERIVED_WEIGHT == 0.0
        else f"hyperstability:{constant.source}"
    )
    basis = ProbabilityBasis(
        cpue_derived_weight=BRT_CPUE_DERIVED_WEIGHT,
        fisheries_independent_weight=1.0 - BRT_CPUE_DERIVED_WEIGHT,
        sources=(
            "USGS_BRT_V2.0",
            f"NHDPlusV2.1_xwalk:{v2_join_method}",
            hyper_tag,
        ),
    )
    return CalibratedProbability(
        point=point,
        lower=lower,
        upper=upper,
        interval_confidence=0.95,
        raw_point=raw_p,
        hyperstability_beta_applied=constant.beta,
        basis=basis,
    )
