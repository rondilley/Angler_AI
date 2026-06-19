"""Ethics and Disclosure Layer (EL) - hard policy gates.

- Bull trout suppressed at HUC-10 (CR-1.1)
- ESA-listed native cutthroat (greenback, Rio Grande) at HUC-10 (CR-1.2)
- State species-of-concern native cutthroat (westslope, Yellowstone, Lahontan,
  Bonneville, Colorado River) at HUC-12 (CR-1.2)
- CRITFC + other tribal-managed waters: AGOL not scraped; redirect to tribe
  data resources (CR-2.1, CR-2.2)
- EPA ATTAINS impaired-water status surfaced as alerts by default (CR-4.3)

Default-on; user override per query is logged. No global disable path (CR-1.5).
"""

from angler_ai.ethics.policy import EthicsPolicy, PolicyDecision
from angler_ai.ethics.sensitive_species import SensitiveSpeciesEntry, SuppressLevel

__all__ = ["EthicsPolicy", "PolicyDecision", "SensitiveSpeciesEntry", "SuppressLevel"]
