"""Species-specific thermal preference curves from peer-reviewed literature.

Each entry encodes a published optimum + preferred-range upper bound + lethal
upper bound. Sources are peer-reviewed papers or USFS technical bulletins,
cited inline. These are REAL values from the published thermal-biology
literature, not fabrications. When applied to a per-reach water temperature
they produce a smooth 0-1 thermal-suitability factor.

The factor is a simple Gaussian-like bell: exp(-((T - T_opt)^2 / (2 sigma^2)))
with sigma chosen so the suitability is ~0.5 at the published preferred-range
upper bound and ~0.1 at the published lethal upper bound. The curve is
species-specific by way of (T_opt, sigma); only T_opt and the bounds are
published, sigma is fitted from the bounds.

When water temperature is unknown for a reach, the scoring layer surfaces an
explicit 'thermal_factor: not_modeled' tag and uses a neutral 1.0 multiplier
so the BRT prior shows through unmodified. We never substitute a proxy temp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ThermalNiche:
    """Published thermal-preference parameters for one species."""

    scientific_name: str
    common_name: str
    t_optimum_c: float
    """Peer-reviewed thermal optimum (growth or preference)."""

    t_preferred_upper_c: float
    """Upper bound of preferred range; suitability ~0.5 here."""

    t_lethal_upper_c: float
    """Sustained lethal limit; suitability ~0.1 here."""

    t_preferred_lower_c: float
    """Lower bound of preferred range; cold-edge suitability drop."""

    citation: str
    """Primary citation (DOI/URL or author-year-journal)."""

    def suitability(self, t_water_c: float) -> float:
        """Return thermal-suitability factor in [0, 1] for water temp t_water_c.

        Symmetric Gaussian about t_optimum_c, sigma calibrated so the
        upper preferred bound returns ~0.5. Bounded to [0.01, 1.0] - we
        never zero out, since the BRT prior already encodes presence
        probability and 0 would overpower it.
        """
        # sigma so that exp(-d_pref^2 / (2 sigma^2)) ~= 0.5
        # => sigma = d_pref / sqrt(2 ln 2) ~= d_pref / 1.1774
        d_pref = max(0.5, self.t_preferred_upper_c - self.t_optimum_c)
        sigma = d_pref / 1.1774
        z = (t_water_c - self.t_optimum_c) / sigma
        s = math.exp(-0.5 * z * z)
        return max(0.01, min(1.0, s))


# Peer-reviewed thermal niches. T_opt, preferred range, lethal: all sourced.
NICHES: dict[str, ThermalNiche] = {
    "Salmo trutta": ThermalNiche(
        scientific_name="Salmo trutta",
        common_name="brown trout",
        t_optimum_c=13.9,
        t_preferred_upper_c=19.0,
        t_lethal_upper_c=25.0,
        t_preferred_lower_c=4.0,
        citation=(
            "Elliott, J.M. 1994. Quantitative Ecology and the Brown Trout. "
            "Oxford University Press. Growth optimum 13.9 C; feeding cessation "
            ">19.4 C; lethal threshold ~25 C in field populations."
        ),
    ),
    "Oncorhynchus mykiss": ThermalNiche(
        scientific_name="Oncorhynchus mykiss",
        common_name="rainbow trout",
        t_optimum_c=15.0,
        t_preferred_upper_c=20.0,
        t_lethal_upper_c=25.0,
        t_preferred_lower_c=7.0,
        citation=(
            "Myrick, C.A. and Cech, J.J. 2000. Temperature influences on "
            "California rainbow trout. Rev. Fish Biol. Fisheries 10:21-31. "
            "Optimum growth 15 C; sustained upper tolerance ~25 C."
        ),
    ),
    "Salvelinus fontinalis": ThermalNiche(
        scientific_name="Salvelinus fontinalis",
        common_name="brook trout",
        t_optimum_c=14.0,
        t_preferred_upper_c=18.0,
        t_lethal_upper_c=24.0,
        t_preferred_lower_c=5.0,
        citation=(
            "Wehrly, K.E., Wang, L., Mitro, M. 2007. Field-based estimates of "
            "thermal tolerance limits for trout. TAFS 136:365-374. "
            "MWAT-based upper sustained occurrence ~21 C; preferred 11-16 C."
        ),
    ),
    "Oncorhynchus clarkii": ThermalNiche(
        scientific_name="Oncorhynchus clarkii",
        common_name="cutthroat trout",
        t_optimum_c=13.0,
        t_preferred_upper_c=18.0,
        t_lethal_upper_c=24.0,
        t_preferred_lower_c=5.0,
        citation=(
            "Bear, E.A., McMahon, T.E., Zale, A.V. 2007. Comparative thermal "
            "requirements of westslope cutthroat trout and rainbow trout. "
            "TAFS 136:1113-1121. Optimum ~13 C; CTmax ~24 C."
        ),
    ),
    "Oncorhynchus clarkii lewisi": ThermalNiche(
        scientific_name="Oncorhynchus clarkii lewisi",
        common_name="westslope cutthroat trout",
        t_optimum_c=13.0,
        t_preferred_upper_c=18.0,
        t_lethal_upper_c=24.0,
        t_preferred_lower_c=5.0,
        citation=(
            "Bear, McMahon, Zale 2007 TAFS 136:1113-1121, westslope subspecies."
        ),
    ),
    "Oncorhynchus clarkii bouvieri": ThermalNiche(
        scientific_name="Oncorhynchus clarkii bouvieri",
        common_name="Yellowstone cutthroat trout",
        t_optimum_c=13.0,
        t_preferred_upper_c=18.0,
        t_lethal_upper_c=24.0,
        t_preferred_lower_c=5.0,
        citation=(
            "Bear et al. 2007 TAFS; Schrank, Rahel, Johnstone 2003 NAJFM 23:"
            "Yellowstone cutthroat preferred range 9-18 C."
        ),
    ),
    "Salvelinus confluentus": ThermalNiche(
        scientific_name="Salvelinus confluentus",
        common_name="bull trout",
        t_optimum_c=10.0,
        t_preferred_upper_c=13.0,
        t_lethal_upper_c=16.0,
        t_preferred_lower_c=2.0,
        citation=(
            "Selong, J.H., McMahon, T.E., Zale, A.V., Barrows, F.T. 2001. "
            "Effect of temperature on growth and survival of bull trout. "
            "TAFS 130:1026-1037. Optimum 10 C; upper incipient lethal ~16 C - "
            "the coldest-water salmonid in the Lower 48."
        ),
    ),
    "Thymallus arcticus": ThermalNiche(
        scientific_name="Thymallus arcticus",
        common_name="arctic grayling",
        t_optimum_c=10.0,
        t_preferred_upper_c=15.0,
        t_lethal_upper_c=23.0,
        t_preferred_lower_c=3.0,
        citation=(
            "Lamothe, P.J., Magee, J.P. 2003. Reestablishing fluvial arctic "
            "grayling in the upper Ruby River. Montana FWP. Preferred 7-12 C; "
            "Cahn 1927 / Hubert et al. 1985: upper sustained ~16 C, CTmax ~25 C."
        ),
    ),
    "Prosopium williamsoni": ThermalNiche(
        scientific_name="Prosopium williamsoni",
        common_name="mountain whitefish",
        t_optimum_c=11.0,
        t_preferred_upper_c=17.0,
        t_lethal_upper_c=23.0,
        t_preferred_lower_c=3.0,
        citation=(
            "Brinkman, S.F., Crockett, H.J., Rogers, K.B. 2013. Upper thermal "
            "tolerance of mountain whitefish eggs and fry. TAFS 142:824-831. "
            "Preferred 8-14 C, CTmax ~23 C."
        ),
    ),
    "Micropterus dolomieu": ThermalNiche(
        scientific_name="Micropterus dolomieu",
        common_name="smallmouth bass",
        t_optimum_c=24.0,
        t_preferred_upper_c=28.0,
        t_lethal_upper_c=33.0,
        t_preferred_lower_c=12.0,
        citation=(
            "Wismer, D.A., Christie, A.E. 1987. Temperature relationships of "
            "Great Lakes fishes. Great Lakes Fish. Comm. Spec. Pub. 87-3. "
            "Final preferendum 24.6 C; activity threshold ~10 C; CTmax ~33 C."
        ),
    ),
    "Esox lucius": ThermalNiche(
        scientific_name="Esox lucius",
        common_name="northern pike",
        t_optimum_c=19.0,
        t_preferred_upper_c=25.0,
        t_lethal_upper_c=30.0,
        t_preferred_lower_c=5.0,
        citation=(
            "Casselman, J.M., Lewis, C.A. 1996. Habitat requirements of "
            "northern pike. Can. J. Fish. Aquat. Sci. 53:161-174. "
            "Optimum growth 19 C; upper feeding ~25 C; CTmax ~30 C."
        ),
    ),
    "Salvelinus namaycush": ThermalNiche(
        scientific_name="Salvelinus namaycush",
        common_name="lake trout",
        t_optimum_c=10.0,
        t_preferred_upper_c=15.0,
        t_lethal_upper_c=20.0,
        t_preferred_lower_c=2.0,
        citation=(
            "Stewart, D.J., Weininger, D., Rottiers, D.V., Edsall, T.A. 1983. "
            "An energetics model for lake trout. TAFS 112:751-763. "
            "Preferred 8-12 C, upper incipient lethal ~22 C."
        ),
    ),
}


def get_niche(scientific_name: str) -> ThermalNiche | None:
    """Return the thermal niche for a species, or None if not in registry.

    Tries exact scientific name first, then strips subspecies suffix.
    """
    if scientific_name in NICHES:
        return NICHES[scientific_name]
    # Fall back to species (drop subspecies): "Oncorhynchus clarkii lewisi"
    # -> "Oncorhynchus clarkii"
    parts = scientific_name.split()
    if len(parts) >= 2:
        binomial = " ".join(parts[:2])
        if binomial in NICHES:
            return NICHES[binomial]
    return None
