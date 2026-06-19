"""Prediction Layer (PL). All outputs return CalibratedProbability (FR-6.4).

Components:
- ssn2: R SSN2 binomial GLM (reach-level catch probability)
- brt_priors: USGS BRT v2.0 species-presence priors
- pg_gnn: physics-guided GNN nowcasts (temperature, flow)
- hydrogem: USGS NWIS anomaly detection
- stan_hurdle: Bayesian hurdle-gamma season-/event-conditioned harvest probability
"""
