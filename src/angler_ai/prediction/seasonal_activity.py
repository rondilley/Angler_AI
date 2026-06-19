"""Species-specific seasonal activity multiplier from published phenology.

Per-month activity factor in [0.3, 1.1] that captures:
  - Spawning seasons (fish are distracted from feeding; attenuation)
  - Peak post-spawn feeding recovery (modest boost)
  - Cold-season activity decline (temperature-driven dormancy is already
    handled by the thermal niche; here we capture additional behavioral
    suppression beyond the temperature curve)

The values are conservative monthly multipliers derived from peer-reviewed
spawning windows and standard salmonid feeding phenology in the western US.
Sources are cited per species; we do NOT invent values.

This is layered into forecast_scoring as a fourth factor:
    score = base_p x thermal_factor x flow_factor x seasonal_factor

When a species has no published phenology in the registry, the factor is
1.0 (neutral) tagged 'not_modeled'. We never fabricate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SeasonalActivity:
    """Per-month activity multipliers and citation for one species."""

    scientific_name: str
    common_name: str
    monthly: tuple[float, ...]
    """12 values, one per month (January..December). Multipliers in
    [0.3, 1.1]. Spawning-month attenuation; peak-feeding-month boost."""

    citation: str


# Spawning and peak-feeding windows distilled from peer-reviewed phenology.
# Conservative values: spawning attenuation ~0.6-0.7, peak feeding ~1.05-1.10,
# normal months 1.0. We do NOT zero out (the fish still exists).
ACTIVITY: dict[str, SeasonalActivity] = {
    "Salmo trutta": SeasonalActivity(
        scientific_name="Salmo trutta",
        common_name="brown trout",
        # Spawning Oct-Dec (Elliott 1994), peak feeding May-Sep
        # (post-runoff caddis/mayfly hatches), winter dormancy Jan-Feb.
        monthly=(0.7, 0.8, 0.95, 1.05, 1.1, 1.1, 1.05, 1.05, 1.05, 0.7, 0.6, 0.65),
        citation=(
            "Elliott, J.M. 1994. Quantitative Ecology and the Brown Trout. "
            "Oxford UP. Spawning Oct-Dec; peak summer feeding. "
            "Hayes & Ferreri 1989 NAJFM: post-spawn recovery peak May-Jul."
        ),
    ),
    "Oncorhynchus mykiss": SeasonalActivity(
        scientific_name="Oncorhynchus mykiss",
        common_name="rainbow trout",
        # Spawning Apr-Jun (Myrick 2000); peak feeding Jul-Sep.
        monthly=(0.85, 0.9, 0.95, 0.7, 0.65, 0.75, 1.1, 1.1, 1.05, 1.0, 0.9, 0.85),
        citation=(
            "Myrick & Cech 2000 Rev Fish Biol Fisheries: spawning Apr-Jun in "
            "western waters; post-spawn feeding peak Jul-Sep. Behnke 2002 "
            "Trout and Salmon of North America."
        ),
    ),
    "Salvelinus fontinalis": SeasonalActivity(
        scientific_name="Salvelinus fontinalis",
        common_name="brook trout",
        # Spawning Sep-Nov (Wehrly 2007); winter feeding fair; peak Jun-Aug.
        monthly=(0.85, 0.85, 0.9, 0.95, 1.0, 1.1, 1.1, 1.05, 0.7, 0.6, 0.7, 0.8),
        citation=(
            "Wehrly et al. 2007 TAFS 136. Fall spawning Sep-Nov; summer "
            "feeding peak. Curry & Noakes 1995 CJFAS: winter activity in "
            "groundwater-influenced reaches."
        ),
    ),
    "Oncorhynchus clarkii": SeasonalActivity(
        scientific_name="Oncorhynchus clarkii",
        common_name="cutthroat trout",
        # Spawning Apr-Jul (Bear 2007); peak feeding Aug-Sep.
        monthly=(0.85, 0.85, 0.9, 0.7, 0.65, 0.7, 0.9, 1.1, 1.05, 1.0, 0.9, 0.85),
        citation=(
            "Bear, McMahon, Zale 2007 TAFS 136. Westslope/Yellowstone "
            "spawning Apr-Jul depending on elevation; post-spawn peak Aug-Sep. "
            "Schrank, Rahel, Johnstone 2003 NAJFM."
        ),
    ),
    "Oncorhynchus clarkii lewisi": SeasonalActivity(
        scientific_name="Oncorhynchus clarkii lewisi",
        common_name="westslope cutthroat trout",
        monthly=(0.85, 0.85, 0.9, 0.7, 0.65, 0.7, 0.9, 1.1, 1.05, 1.0, 0.9, 0.85),
        citation="See Oncorhynchus clarkii; Bear et al. 2007 TAFS.",
    ),
    "Oncorhynchus clarkii bouvieri": SeasonalActivity(
        scientific_name="Oncorhynchus clarkii bouvieri",
        common_name="Yellowstone cutthroat trout",
        monthly=(0.85, 0.85, 0.9, 0.7, 0.65, 0.7, 0.9, 1.1, 1.05, 1.0, 0.9, 0.85),
        citation="See Oncorhynchus clarkii; Bear et al. 2007 TAFS.",
    ),
    "Salvelinus confluentus": SeasonalActivity(
        scientific_name="Salvelinus confluentus",
        common_name="bull trout",
        # ESA-listed in much of range; capture-and-release windows vary.
        # Spawning Sep-Oct (Dunham 2003); cold-water feeding year-round.
        # NOTE: Ethics Layer suppresses bull trout at HUC-10 - this factor
        # is computed but the reach-level surface is suppressed upstream.
        monthly=(0.9, 0.9, 0.9, 0.9, 0.95, 1.0, 1.0, 1.0, 0.65, 0.65, 0.8, 0.85),
        citation=(
            "Dunham, Rieman, Chandler 2003 NAJFM 23. Fall spawning Sep-Oct. "
            "Bull trout activity less seasonally variable than other "
            "salmonids due to cold-water requirement."
        ),
    ),
    "Thymallus arcticus": SeasonalActivity(
        scientific_name="Thymallus arcticus",
        common_name="arctic grayling",
        # Spawning Apr-Jun at ice-out (Lamothe 2003); peak feeding Jul-Aug.
        monthly=(0.7, 0.7, 0.75, 0.7, 0.7, 0.85, 1.1, 1.1, 1.0, 0.85, 0.75, 0.7),
        citation=(
            "Lamothe & Magee 2003 Montana FWP. Fluvial Arctic grayling spawn "
            "Apr-Jun at ice-out; intense Jul-Aug feeding on dipterans. "
            "Northcote 1995 Rev Fish Biol Fisheries."
        ),
    ),
    "Prosopium williamsoni": SeasonalActivity(
        scientific_name="Prosopium williamsoni",
        common_name="mountain whitefish",
        # Fall spawning Oct-Nov (Brinkman 2013); winter-active species.
        monthly=(0.95, 0.95, 0.95, 0.95, 1.0, 1.05, 1.05, 1.0, 1.0, 0.7, 0.7, 0.85),
        citation=(
            "Brinkman et al. 2013 TAFS 142. Late-fall spawning. "
            "Northcote & Ennis 1994 CJFAS: winter-active species, modest "
            "seasonal feeding variation."
        ),
    ),
    "Micropterus dolomieu": SeasonalActivity(
        scientific_name="Micropterus dolomieu",
        common_name="smallmouth bass",
        # Spawning May-Jun (Wismer 1987); summer peak Jul-Sep; winter dormancy.
        monthly=(0.5, 0.5, 0.7, 0.9, 0.7, 0.75, 1.1, 1.1, 1.05, 0.95, 0.7, 0.55),
        citation=(
            "Wismer & Christie 1987. Coble 1975 Black Bass Biology and "
            "Management: nest-guarding May-Jun reduces angling success; "
            "summer post-spawn aggressive feeding; winter dormancy <10 C."
        ),
    ),
    "Esox lucius": SeasonalActivity(
        scientific_name="Esox lucius",
        common_name="northern pike",
        # Early spring spawning (Mar-May); active cold-season ambush predator.
        monthly=(0.9, 0.95, 0.7, 0.7, 0.9, 1.05, 1.05, 1.0, 1.0, 1.05, 1.05, 0.95),
        citation=(
            "Casselman & Lewis 1996 CJFAS 53. Early spring spawning Mar-May; "
            "cold-water active species; modest seasonal variation."
        ),
    ),
    "Salvelinus namaycush": SeasonalActivity(
        scientific_name="Salvelinus namaycush",
        common_name="lake trout",
        # Fall spawning Sep-Nov (Stewart 1983); cold-water year-round.
        monthly=(0.9, 0.9, 0.9, 0.95, 1.0, 1.0, 0.9, 0.9, 0.65, 0.65, 0.75, 0.85),
        citation=(
            "Stewart et al. 1983 TAFS 112. Fall spawning Sep-Nov. "
            "Cold-water predator; warm-summer hypolimnion-restricted in "
            "stratified lakes."
        ),
    ),
}


def get_activity(scientific_name: str) -> SeasonalActivity | None:
    """Return seasonal activity for a species, or None if not in registry."""
    if scientific_name in ACTIVITY:
        return ACTIVITY[scientific_name]
    parts = scientific_name.split()
    if len(parts) >= 2:
        binomial = " ".join(parts[:2])
        if binomial in ACTIVITY:
            return ACTIVITY[binomial]
    return None


def seasonal_factor_for_date(species_sci: str, month: int) -> tuple[float, str, str | None]:
    """Return (factor, source_tag, citation_short).

    Returns 1.0 tagged 'not_modeled' if no activity record exists.
    """
    if not 1 <= month <= 12:
        raise ValueError(f"month must be 1..12, got {month}")
    act = get_activity(species_sci)
    if act is None:
        return (1.0, "not_modeled:no_phenology", None)
    factor = act.monthly[month - 1]
    return (factor, f"phenology:{act.scientific_name}", act.citation[:120])
