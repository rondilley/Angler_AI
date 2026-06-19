"""Ethics policy gate. Called before every reach-level output surface.

`evaluate()` returns a PolicyDecision the caller MUST honor. Bypassing the
ethics layer in a code path is a violation of CR-1.1/1.2/2.x/4.x and will fail
the M3+ release gates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from angler_ai.ethics.sensitive_species import SensitiveSpeciesEntry, SuppressLevel

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Output of the policy gate. Callers honor `suppress_level` strictly."""

    allow_reach_level: bool
    """False if the species is sensitive at this geography."""

    suppress_level: SuppressLevel
    """Coarse-grained level the caller must aggregate to."""

    redirect_notice: str | None = None
    """If the query intersects tribal-managed waters, this is the user-facing
    redirect text. The caller must surface it instead of any reach data."""

    user_override_acknowledged: bool = False
    """Per-query user override (CR-1.5). When True, callers may emit reach-level
    output, but the override and acknowledgment are logged."""

    rationale: str = ""


class EthicsPolicy:
    """Hard policy gate.

    Implementations of `evaluate` consult the `sensitive_species` and
    `tribal_mask` tables and return a PolicyDecision. The default-on,
    no-global-disable invariant (CR-1.5, CR-4.2) is enforced here.
    """

    def __init__(self, species_table: tuple[SensitiveSpeciesEntry, ...]) -> None:
        self._by_species_id = {s.species_id: s for s in species_table}

    def evaluate(
        self,
        comid: int,
        species_id: str,
        state: str,
        user_override: bool = False,
    ) -> PolicyDecision:
        """Decide whether reach-level output is allowed for this species at this comid.

        Args:
            comid: NHDPlus HR identifier.
            species_id: species id from the registry.
            state: two-letter state code.
            user_override: per-query override (CR-1.5). Logged when used.

        Returns:
            PolicyDecision the caller must honor.
        """
        # Tribal-mask check (CR-2.1/2.2) is wired in at M3 once tribal_mask is loaded.
        # For v0 skeleton, fall back to the species-only check.
        entry = self._by_species_id.get(species_id)
        if entry is None:
            return PolicyDecision(
                allow_reach_level=True,
                suppress_level=SuppressLevel.REACH,
                rationale="No sensitive-species entry; reach-level allowed.",
            )
        if user_override:
            log.warning(
                "user override invoked for sensitive species %s at comid=%d state=%s; "
                "reach-level output emitted per CR-1.5 with acknowledgment.",
                species_id, comid, state,
            )
            return PolicyDecision(
                allow_reach_level=True,
                suppress_level=SuppressLevel.REACH,
                user_override_acknowledged=True,
                rationale=f"User override of {entry.suppress_level.value} suppression for {species_id}.",
            )
        return PolicyDecision(
            allow_reach_level=False,
            suppress_level=entry.suppress_level,
            rationale=(
                f"Sensitive species {entry.common_name} ({entry.status}); "
                f"reach-level suppressed per {entry.source_doc}. Coarse-grained at {entry.suppress_level.value}."
            ),
        )
