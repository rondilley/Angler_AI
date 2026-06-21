"""Render a per-reach scored prediction as a colored PNG map.

Color ramp: matplotlib RdYlGn applied to the [0, 1] daily_score.
Reaches are drawn as line collections from their NHDPlus MULTILINESTRING
geometry over a USGS National Map USTopo basemap (public-domain federal
tile service at basemap.nationalmap.gov). The output is a single PNG file
with title + colorbar + caption identifying the data sources used (USGS
BRT v2.0 + species niche citation + NWS forecast + NWIS water-temp source
if any + basemap attribution).

Honest behavior:
  - Reaches with no BRT prior are omitted (they would not appear at all
    in the input score list); the renderer does not invent them.
  - The caption surfaces how many reaches had real water-temp data vs
    'not_modeled', so a reader can see at a glance how much of the surface
    was actually conditioned on temperature.
  - When `basemap=True` (default) and the optional `contextily` extra is
    installed, the renderer fetches USGS topo tiles (cached locally after
    first hit) and draws the reach lines with a white halo for legibility
    against the varied terrain background. If the network is unavailable
    or contextily is missing, the renderer logs a warning and degrades to
    a plain white background WITHOUT silently substituting any value -
    the caption reflects whichever path was taken.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend; never opens a window
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from angler_ai.prediction.forecast_scoring import DailyScore

log = logging.getLogger(__name__)

# USGS National Map USTopo. Federal tile service; public-domain.
# https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer
_DEFAULT_BASEMAP_PROVIDER = "USGS.USTopo"
_BASEMAP_ATTRIBUTION = (
    "Basemap: USGS National Map USTopo (public-domain, basemap.nationalmap.gov)"
)


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


def _add_topo_basemap(ax, *, provider_path: str = _DEFAULT_BASEMAP_PROVIDER) -> str:
    """Fetch and draw a topo basemap under the current axes.

    Returns a status string for the caption: either the basemap attribution
    on success, or an explicit fallback note on failure (no silent
    degradation per CLAUDE.md). The caller appends this to the caption.

    The `crs="EPSG:4326"` arg tells contextily to reproject web-mercator
    tiles to match our WGS84 lon/lat axes - slower than reprojecting the
    geometries, but the geometries stay in their native CRS for the rest
    of the rendering pipeline.
    """
    try:
        import contextily as cx
    except ImportError:
        log.warning(
            "render: basemap requested but `contextily` is not installed. "
            "Install with: pip install angler-ai[maps]. Rendering with "
            "white background; caption updated to reflect."
        )
        return "Basemap: not_modeled (contextily not installed)"

    # Resolve dotted provider path against contextily.providers tree.
    provider = cx.providers
    for part in provider_path.split("."):
        provider = getattr(provider, part, None)
        if provider is None:
            log.warning(
                "render: basemap provider %s not found in contextily.providers; "
                "white background.", provider_path,
            )
            return f"Basemap: not_modeled (provider {provider_path} not found)"

    try:
        cx.add_basemap(
            ax,
            crs="EPSG:4326",
            source=provider,
            attribution_size=6,
            # `attribution` set to empty since we add our own line in the
            # caption (gives consistent formatting alongside data-source
            # citations).
            attribution="",
        )
    except Exception as exc:  # noqa: BLE001 - network/HTTP failures are expected offline
        log.warning(
            "render: basemap tile fetch failed (%s). Falling back to white "
            "background; caption updated to reflect.", exc,
        )
        return f"Basemap: not_modeled (tile fetch failed: {type(exc).__name__})"
    return _BASEMAP_ATTRIBUTION


def render_scored_reaches_png(
    *,
    scored: list[ScoredReach],
    output_path: str | Path,
    title: str,
    subtitle: str,
    caption: str,
    figsize: tuple[float, float] = (12.0, 9.0),
    dpi: int = 150,
    basemap: bool = True,
    basemap_provider: str = _DEFAULT_BASEMAP_PROVIDER,
) -> Path:
    """Render the scored reaches into a single PNG.

    Args:
        scored: list of ScoredReach. Each must have a parseable geometry and
            a score in [0, 1].
        output_path: file path for the PNG.
        title, subtitle, caption: text for the figure.
        figsize, dpi: matplotlib figure size / dpi.
        basemap: when True, draw a USGS topo basemap under the reaches.
            Requires `contextily` (install via `pip install angler-ai[maps]`).
            Gracefully degrades to white background with explicit caption
            note if contextily is missing or tile fetch fails.
        basemap_provider: dotted xyzservices provider path. Default
            'USGS.USTopo' (federal public-domain). Other useful options:
            'USGS.USImageryTopo' (satellite + roads), 'OpenTopoMap'
            (community CC-BY-SA contours).

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
    # eye is drawn to good water. Slightly bumped vs the white-bg renderer
    # so the lines stay legible against terrain.
    linewidths = [0.8 + 2.4 * v for v in values]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Set extent BEFORE basemap so contextily picks the right zoom level.
    all_x = [x for seg in segments for x, _ in seg]
    all_y = [y for seg in segments for _, y in seg]
    if all_x and all_y:
        margin = 0.04
        dx = (max(all_x) - min(all_x)) * margin or 0.01
        dy = (max(all_y) - min(all_y)) * margin or 0.01
        ax.set_xlim(min(all_x) - dx, max(all_x) + dx)
        ax.set_ylim(min(all_y) - dy, max(all_y) + dy)

    # Topo basemap (or fallback). Status string surfaced in caption so the
    # reader knows whether they're looking at a topo overlay or a plain bg.
    basemap_status: str
    if basemap:
        basemap_status = _add_topo_basemap(ax, provider_path=basemap_provider)
    else:
        basemap_status = "Basemap: disabled (basemap=False)"

    # Reach lines with a thin white halo so they stay legible over varied
    # terrain colors. The halo is path_effects-based and renders as part
    # of the LineCollection's stroke.
    lc = LineCollection(segments, colors=colors, linewidths=linewidths, alpha=0.95)
    lc.set_path_effects([
        path_effects.Stroke(linewidth=4.0, foreground="white", alpha=0.75),
        path_effects.Normal(),
    ])
    ax.add_collection(lc)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    # No grid over the topo basemap - the topo already carries graticule info.
    ax.grid(False)

    # Colorbar.
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Relative suitability index (0-1, NOT a probability)")

    # Title + subtitle + caption (basemap attribution appended).
    full_caption = caption + " | " + basemap_status
    fig.suptitle(title, fontsize=14, fontweight="bold")
    ax.set_title(subtitle, fontsize=10)
    fig.text(0.5, 0.02, full_caption, ha="center", va="bottom",
             fontsize=8, color="#444444", wrap=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info(
        "Map rendered: %s (%d reaches, basemap=%s)",
        output_path, len(segments),
        "ok" if basemap_status == _BASEMAP_ATTRIBUTION else "fallback",
    )
    return output_path


# --------------------------------------------------------------------------- #
# DELTA renderer + animation GIF stitcher (2026-06-20)                        #
# --------------------------------------------------------------------------- #
#
# Motivation: the per-day absolute-suitability maps make it hard to see WHICH
# DIRECTION fishing conditions are trending. The delta map shows day-over-day
# CHANGE per reach on a diverging "red shift = worse / blue shift = better"
# colormap. Reaches whose change is below a significance threshold render as
# neutral gray so the eye is drawn only to meaningful movement.

_DELTA_CMAP = "RdBu"
"""Matplotlib diverging colormap. RdBu maps negative (delta < 0) to red and
positive (delta > 0) to blue, with white at zero. Reads as 'redshift = getting
worse, blueshift = getting better' per the user's framing."""

_DELTA_VMAX_DEFAULT = 0.15
"""Symmetric +-vmax for the diverging colormap. Deltas larger in magnitude
than this clip to the colormap extremes. 0.15 is roughly 3x the typical
day-over-day median shift; bigger deltas read as 'major change'."""

_DELTA_SIGNIFICANCE_THRESHOLD = 0.02
"""|delta| values below this render as neutral gray (no signal). Tuned to
suppress numerical noise; bumps with real flow / thermal regime changes
will exceed it cleanly."""


def render_delta_reaches_png(
    *,
    scored_today: list[ScoredReach],
    scored_yesterday: list[ScoredReach],
    output_path: str | Path,
    title: str,
    subtitle: str,
    caption: str,
    significance_threshold: float = _DELTA_SIGNIFICANCE_THRESHOLD,
    delta_vmax: float = _DELTA_VMAX_DEFAULT,
    figsize: tuple[float, float] = (12.0, 9.0),
    dpi: int = 150,
    basemap: bool = True,
    basemap_provider: str = _DEFAULT_BASEMAP_PROVIDER,
) -> Path:
    """Render a day-over-day delta map.

    Each reach is colored by (today's suitability - yesterday's suitability):
      - red shades        : suitability DROPPED (fishing conditions worsening)
      - blue shades       : suitability ROSE   (fishing conditions improving)
      - neutral gray      : |delta| < significance_threshold (no signal)

    Reaches missing from yesterday (e.g., scoring failure on day 0) are
    silently omitted - the renderer does NOT invent a baseline. The caption
    surfaces how many reaches contributed to the delta plot.

    Args:
        scored_today: today's per-reach scores.
        scored_yesterday: yesterday's per-reach scores. Joined to today
            by COMID.
        output_path: file path for the PNG.
        title, subtitle, caption: text for the figure.
        significance_threshold: |delta| below this renders as gray.
        delta_vmax: symmetric color range (+- this value). Deltas larger
            in magnitude clip to the extremes.
        figsize, dpi, basemap, basemap_provider: same as
            `render_scored_reaches_png`.

    Returns:
        Path to the written PNG.
    """
    if not scored_today:
        raise ValueError("render_delta_reaches_png: empty today list")
    if not scored_yesterday:
        raise ValueError(
            "render_delta_reaches_png: empty yesterday list. Cannot delta "
            "against nothing - skip the first day in the window when calling."
        )

    yesterday_by_comid: dict[int, float] = {
        sr.comid: float(sr.score.suitability_index) for sr in scored_yesterday
    }

    segments: list[list[tuple[float, float]]] = []
    deltas: list[float] = []
    n_significant = 0
    n_total = 0
    n_dropped_no_match = 0

    for sr in scored_today:
        if sr.comid not in yesterday_by_comid:
            n_dropped_no_match += 1
            continue
        n_total += 1
        rings = _parse_multilinestring_wkt(sr.geometry_wkt)
        if not rings:
            continue
        delta = float(sr.score.suitability_index) - yesterday_by_comid[sr.comid]
        if abs(delta) >= significance_threshold:
            n_significant += 1
        for ring in rings:
            segments.append(ring)
            deltas.append(delta)

    if not segments:
        raise ValueError(
            "render_delta_reaches_png: no reaches matched between today "
            "and yesterday (and had parseable geometries). Check COMID sets."
        )

    norm = Normalize(vmin=-delta_vmax, vmax=delta_vmax)
    cmap = plt.get_cmap(_DELTA_CMAP)
    colors: list = []
    linewidths: list[float] = []
    for d in deltas:
        if abs(d) < significance_threshold:
            # Neutral gray for stable reaches.
            colors.append((0.55, 0.55, 0.55, 0.45))
            linewidths.append(0.6)
        else:
            colors.append(cmap(norm(d)))
            # Wider for larger absolute changes - draws eye to action.
            mag = min(abs(d), delta_vmax) / delta_vmax
            linewidths.append(1.0 + 2.6 * mag)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Set extent BEFORE basemap so contextily picks the right zoom level.
    all_x = [x for seg in segments for x, _ in seg]
    all_y = [y for seg in segments for _, y in seg]
    if all_x and all_y:
        margin = 0.04
        dx = (max(all_x) - min(all_x)) * margin or 0.01
        dy = (max(all_y) - min(all_y)) * margin or 0.01
        ax.set_xlim(min(all_x) - dx, max(all_x) + dx)
        ax.set_ylim(min(all_y) - dy, max(all_y) + dy)

    basemap_status: str
    if basemap:
        basemap_status = _add_topo_basemap(ax, provider_path=basemap_provider)
    else:
        basemap_status = "Basemap: disabled (basemap=False)"

    lc = LineCollection(segments, colors=colors, linewidths=linewidths, alpha=0.95)
    lc.set_path_effects([
        path_effects.Stroke(linewidth=4.5, foreground="white", alpha=0.75),
        path_effects.Normal(),
    ])
    ax.add_collection(lc)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.grid(False)

    # Diverging colorbar centered at 0.
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(
        f"Day-over-day change in suitability index\n"
        f"(red = getting worse, blue = getting better; "
        f"gray when |delta| < {significance_threshold:.02f})"
    )
    cbar.ax.axhline(y=0.0, color="black", linewidth=0.5)

    pct_significant = (100.0 * n_significant / n_total) if n_total else 0.0
    full_caption = (
        f"{caption} | "
        f"Delta plot: {n_total} reaches, {n_significant} significantly changed "
        f"({pct_significant:.0f}%, |delta| >= {significance_threshold:.02f}) | "
        f"{basemap_status}"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold")
    ax.set_title(subtitle, fontsize=10)
    fig.text(0.5, 0.02, full_caption, ha="center", va="bottom",
             fontsize=8, color="#444444", wrap=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info(
        "Delta map rendered: %s (%d reaches, %d significant, basemap=%s)",
        output_path, n_total, n_significant,
        "ok" if basemap_status == _BASEMAP_ATTRIBUTION else "fallback",
    )
    return output_path


def render_animation_gif(
    *,
    frame_paths: list[Path | str],
    output_path: str | Path,
    duration_ms: int = 600,
    loop: int = 0,
    first_last_hold_ms: int = 1800,
) -> Path:
    """Stitch a list of PNG paths into an animated GIF.

    Uses Pillow (already a matplotlib dep; no new package required). Per-frame
    durations are set so the first and last frame hold longer than the
    middle frames - the viewer needs time to register start and end state.

    Args:
        frame_paths: PNG file paths in display order. Must exist on disk.
        output_path: GIF file path to write.
        duration_ms: per-frame duration for middle frames, in milliseconds.
            Default 600ms (~1.7 fps).
        loop: 0 = infinite loop, 1 = play once, N = play N times.
        first_last_hold_ms: longer duration applied to frame[0] and frame[-1]
            so they read clearly as the start / end of the window.

    Returns:
        Path to the written GIF.
    """
    from PIL import Image

    if not frame_paths:
        raise ValueError("render_animation_gif: empty frame list")
    paths = [Path(p) for p in frame_paths]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"render_animation_gif: missing frame files: {missing[:3]}..."
        )

    # Pillow's GIF writer needs a palette-mode image. RGB PNGs at our size
    # (~12in x 9in @ 150dpi = 1800x1350) can be quite large. Downsample to
    # a reasonable display size to keep the GIF under ~10MB.
    target_width = 1200

    frames: list = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        if im.width > target_width:
            ratio = target_width / im.width
            new_size = (target_width, int(im.height * ratio))
            im = im.resize(new_size, Image.LANCZOS)
        # Quantize to a 256-color palette using median-cut (Pillow default).
        # adaptive palette per-frame would be larger; a global palette would
        # mute colors. median-cut per-frame is the right tradeoff for GIF.
        frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=256))

    durations: list[int] = []
    for i in range(len(frames)):
        if i == 0 or i == len(frames) - 1:
            durations.append(first_last_hold_ms)
        else:
            durations.append(duration_ms)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=loop,
        optimize=True,
        disposal=2,
    )
    log.info(
        "Animation GIF written: %s (%d frames, total %.1fs, size %d KB)",
        output_path, len(frames), sum(durations) / 1000.0,
        output_path.stat().st_size // 1024,
    )
    return output_path
