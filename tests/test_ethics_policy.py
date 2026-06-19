"""Tests for the Ethics policy gate - FR-8.x, CR-1.x.

These are hard policy invariants. A test failing here means a code path can
disclose sensitive-species data at reach level (CR-1.1/CR-1.2 violation) or
ignore tribal sovereignty (CR-2.x violation).
"""

from __future__ import annotations

from angler_ai.ethics import EthicsPolicy, SuppressLevel
from angler_ai.ethics.sensitive_species import load_seed


def test_seed_loads_with_expected_species() -> None:
    """Seed file includes bull trout + at least four cutthroat subspecies."""
    seed = load_seed()
    ids = {s.species_id for s in seed}
    assert "bull-trout" in ids
    assert "westslope-cutthroat" in ids
    assert "yellowstone-cutthroat" in ids
    assert "greenback-cutthroat" in ids
    assert "rio-grande-cutthroat" in ids


def test_bull_trout_suppressed_at_huc10() -> None:
    """CR-1.1: bull trout reach-level disclosure is denied."""
    policy = EthicsPolicy(species_table=load_seed())
    decision = policy.evaluate(comid=12345, species_id="bull-trout", state="ID")
    assert decision.allow_reach_level is False
    assert decision.suppress_level == SuppressLevel.HUC10


def test_westslope_cutthroat_suppressed_at_huc12() -> None:
    """CR-1.2: state-SOC cutthroats coarse-grained at HUC-12."""
    policy = EthicsPolicy(species_table=load_seed())
    decision = policy.evaluate(comid=22222, species_id="westslope-cutthroat", state="MT")
    assert decision.allow_reach_level is False
    assert decision.suppress_level == SuppressLevel.HUC12


def test_unlisted_species_allows_reach_level() -> None:
    """Common non-native species (e.g. smallmouth bass) are not suppressed."""
    policy = EthicsPolicy(species_table=load_seed())
    decision = policy.evaluate(comid=33333, species_id="smallmouth-bass", state="PA")
    assert decision.allow_reach_level is True
    assert decision.suppress_level == SuppressLevel.REACH


def test_user_override_acknowledged_and_logged() -> None:
    """CR-1.5: per-query override is permitted with acknowledgment."""
    policy = EthicsPolicy(species_table=load_seed())
    decision = policy.evaluate(
        comid=12345, species_id="bull-trout", state="ID", user_override=True,
    )
    assert decision.allow_reach_level is True
    assert decision.user_override_acknowledged is True


def test_greenback_and_rio_grande_are_huc10() -> None:
    """CR-1.2: ESA-listed cutthroat subspecies coarse-grained at HUC-10."""
    policy = EthicsPolicy(species_table=load_seed())
    for species_id in ("greenback-cutthroat", "rio-grande-cutthroat"):
        decision = policy.evaluate(comid=44444, species_id=species_id, state="CO")
        assert decision.suppress_level == SuppressLevel.HUC10, species_id
