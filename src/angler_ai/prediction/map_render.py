"""Render a per-reach scored prediction as a colored PNG map.

Color ramp: matplotlib RdYlGn applied to the [0, 1] daily_score.
Reaches are drawn as line collections from their NHDPlus MULTILINESTRING
geometry. The output is a single PNG file with title + colorbar + caption
identifying the data sources used (USGS BRT v2.0 + species niche citation +
NWS forecast + NWIS water-temp source if any).

Honest behavior:
  - Reaches with no BRT prior are omitted (they would not appear at all
    in the input score list); the renderer does not invent them.
  - The caption surfaces how many reaches had real water-temp data vs
    'not_modeled', so a reader can see at a glance how much of the surface
    was actually conditioned on temperature.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend; never opens a window
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from angler_ai.prediction.forecast_scoring import DailyScore

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScoredReach:
    """One reach with its geometry WKT and its best-day score."""

    comid: int
    geometry_wkt: str
    """MULTILINESTRING WKT from the reaches table."""

    score: DailyScore


def _parse_multilinestring_wkt(wkt: str) -> list[list[tuple[float, float]]]:
    """Return list-of-rings of (lon, lat). Skip any non-MULTILINESTRING."""
    if not wkt:
        return []
    s = wkt.strip()
    if s.startswith("MULTILINESTRING"):
        body = s[len("MULTILINESTRING"):].strip()
    elif s.startswith("LINESTRING"):
        body = "(" + s[len("LINESTRING"):].strip() + ")"
    else:
        return []
    if not body.startswith("(") or not body.endswith(")"):
        return []
    body = body[1:-1].strip()
    # Split into rings.
    rings_str: list[str] = []
    depth = 0
    cur = []
    for ch in body:
        if ch == "(":
            depth += 1
            if depth == 1:
                cur = []
                continue
        if ch == ")":
            depth -= 1
            if depth == 0:
                rings_str.append("".join(cur))
                continue
        if depth >= 1:
            cur.append(ch)
    rings: list[list[tuple[float, float]]] = []
    for r in rings_str:
        pts: list[tuple[float, float]] = []
        for raw in r.split(","):
            parts = raw.strip().split()
            if len(parts) < 2:
                continue
            try:
                lon = float(parts[0])
                lat = float(parts[1])
            except ValueError:
                continue
            pts.append((lon, lat))
        if len(pts) >= 2:
            rings.append(pts)
    return rings


def render_scored_reaches_png(
    *,
    scored: list[ScoredReach],
    output_path: str | Path,
    title: str,
    subtitle: str,
    caption: str,
    figsize: tuple[float, float] = (12.0, 9.0),
    dpi: int = 150,
) -> Path:
    """Render the scored reaches into a single PNG.

    Args:
        scored: list of ScoredReach. Each must have a parseable geometry and
            a score in [0, 1].
        output_path: file path for the PNG.
        title, subtitle, caption: text for the figure.
        figsize, dpi: matplotlib figure size / dpi.

    Returns:
        Path to the written PNG.
    """
    if not scored:
        raise ValueError("render_scored_reaches_png: empty score list")

    segments: list[list[tuple[float, float]]] = []
    values: list[float] = []
    for sr in scored:
        rings = _parse_multilinestring_wkt(sr.geometry_wkt)
        if not rings:
            continue
        for ring in rings:
            segments.append(ring)
            values.append(float(sr.score.suitability_index))

    if not segments:
        raise ValueError(
            "render_scored_reaches_png: no parseable geometries (likely empty WKT). "
            "Did you JOIN reaches in the scoring driver?"
        )

    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap("RdYlGn")
    colors = [cmap(norm(v)) for v in values]
    # Make stream-order-ish width: high-prob reaches a bit thicker so the
    # eye is drawn to good water.
    linewidths = [0.5 + 2.0 * v for v in values]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    lc = LineCollection(segments, colors=colors, linewidths=linewidths, alpha=0.95)
    ax.add_collection(lc)

    # Auto-fit the axes to the geometry extent with a small margin.
    all_x = [x for seg in segments for x, _ in seg]
    all_y = [y for seg in segments for _, y in seg]
    if all_x and all_y:
        margin = 0.02
        dx = (max(all_x) - min(all_x)) * margin or 0.01
        dy = (max(all_y) - min(all_y)) * margin or 0.01
        ax.set_xlim(min(all_x) - dx, max(all_x) + dx)
        ax.set_ylim(min(all_y) - dy, max(all_y) + dy)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.grid(True, alpha=0.25, linewidth=0.5)

    # Colorbar.
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Relative suitability index (0-1, NOT a probability)")

    # Title + subtitle + caption.
    fig.suptitle(title, fontsize=14, fontweight="bold")
    ax.set_title(subtitle, fontsize=10)
    fig.text(0.5, 0.02, caption, ha="center", va="bottom",
             fontsize=8, color="#444444", wrap=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Map rendered: %s (%d reaches)", output_path, len(segments))
    return output_path
