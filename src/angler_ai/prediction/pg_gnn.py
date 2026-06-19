"""Physics-guided recurrent GNN for water temperature + flow nowcasts (FR-5.3).

v0 status: NotImplementedError. The real implementation requires:
  - Jia 2021 PG-RGN reference architecture (arXiv 2009.12575) plus the
    He 2024 fair-graph-sampling refinement (arXiv 2412.16523)
  - PyTorch implementation over NHDPlus HR graph + USGS NWIS gauge subset
  - Training against observed stream temperatures (NorWeST / EcoSHEDS labels
    where available) with physics-loss term from PRMS-SNTemp
  - Reach-level predict()

NO FABRICATED VALUES. v0 raises NotImplementedError and the temperature
resolver surfaces 'not_modeled' for reaches not covered by a real source.
"""

from __future__ import annotations


def nowcast_temperature(comid: int, date: str) -> float:
    """Predict temperature on a reach lacking direct measurement. v0: not impl."""
    raise NotImplementedError(
        "v1 milestone. Requires PyTorch implementation of Jia 2021 PG-RGN + "
        "training against NorWeST/EcoSHEDS observed labels."
    )


def nowcast_flow(comid: int, date: str) -> float:
    """Predict flow on a reach lacking a USGS NWIS gauge. v0: not impl."""
    raise NotImplementedError(
        "v1 milestone. Same architecture as nowcast_temperature with discharge as target."
    )
