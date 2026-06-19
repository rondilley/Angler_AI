"""Capability-based router. Picks a (ModelEntry, QuantVariant) for a problem set.

Implements FR-3.2, FR-3.6 (explicit error on no-fit; no silent degradation).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from angler_ai.hardware.models import HardwareProfile
from angler_ai.registry.models import (
    Capability,
    ModelEntry,
    ModelType,
    ProblemSet,
    QuantVariant,
)

log = logging.getLogger(__name__)


# Per-ProblemSet capability requirements. Adding a new ProblemSet only requires
# extending this table; the router needs no other changes (NFR-7.3).
#
# Long-context requirements are expressed numerically (third tuple element);
# the LONG_CONTEXT_* capability flags are advisory metadata on the model and
# are NOT part of the subset match.
PROBLEM_SET_REQUIREMENTS: dict[ProblemSet, tuple[ModelType, frozenset[Capability], int]] = {
    ProblemSet.REGULATION_QA: (
        ModelType.TEXT,
        frozenset({Capability.INSTRUCTION_FOLLOWING}),
        8192,
    ),
    ProblemSet.ANGLER_REPORT_EXTRACTION: (
        ModelType.TEXT,
        frozenset({Capability.JSON_SCHEMA, Capability.INSTRUCTION_FOLLOWING}),
        4096,
    ),
    ProblemSet.ANOMALY_EXPLANATION: (
        ModelType.TEXT,
        frozenset({Capability.INSTRUCTION_FOLLOWING}),
        4096,
    ),
    ProblemSet.EXPLAIN_REACH: (
        ModelType.TEXT,
        frozenset({Capability.TOOL_USE, Capability.INSTRUCTION_FOLLOWING}),
        16384,
    ),
    ProblemSet.SEMANTIC_SEARCH: (
        ModelType.EMBEDDING,
        frozenset({Capability.EMBEDDING}),
        512,
    ),
    ProblemSet.FISH_PHOTO_ID: (
        ModelType.VISION,
        frozenset({Capability.VISION_INPUT}),
        4096,
    ),
    ProblemSet.VOICE_NOTE_TRANSCRIPTION: (
        ModelType.ASR,
        frozenset(),
        0,
    ),
}


@dataclass(frozen=True, slots=True)
class RouterDecision:
    """Output of `select_model`. Logs the candidate set and chosen winner."""

    model: ModelEntry
    quant: QuantVariant
    candidates_considered: int
    hardware_budget_mb: int
    reason: str


class NoModelFitsError(RuntimeError):
    """Raised when the registry has no model that satisfies a problem set on the
    given hardware. FR-3.6 requires this to be an explicit error, never a silent
    fallback to a worse model.
    """


def select_model(
    problem_set: ProblemSet,
    hardware: HardwareProfile,
    catalog: Iterable[ModelEntry],
) -> RouterDecision:
    """Pick a (model, quant) pair for the problem set on the given hardware.

    Args:
        problem_set: The internal capability declaration.
        hardware: HardwareProfile from `angler_ai.hardware.probe()`.
        catalog: Registry entries (load via `load_catalog()`).

    Returns:
        RouterDecision wrapping the winner plus the basis for the decision.

    Raises:
        NoModelFitsError: when no quant from any candidate model fits the
            VRAM/context budget.
    """
    if problem_set not in PROBLEM_SET_REQUIREMENTS:
        raise ValueError(f"Unknown problem set: {problem_set}")
    required_type, required_caps, required_ctx = PROBLEM_SET_REQUIREMENTS[problem_set]
    budget_mb = hardware.available_vram_mb(margin=0.10)
    candidates: list[tuple[ModelEntry, QuantVariant]] = []
    considered = 0
    for model in catalog:
        considered += 1
        if model.type != required_type:
            continue
        if not required_caps.issubset(model.capabilities):
            continue
        if model.context_length < required_ctx:
            continue
        for quant in model.quants:
            if quant.vram_required_mb <= budget_mb:
                candidates.append((model, quant))
    if not candidates:
        raise NoModelFitsError(
            f"No model in the registry fits problem set {problem_set.value!r} on "
            f"this hardware (budget {budget_mb} MB). Considered {considered} entries. "
            "Increase available VRAM, install more models, or run the task on a "
            "more capable machine."
        )
    # Default ranking: prefer higher quality_score for the problem set, then
    # higher total_params_b (richer model), then larger active_params (more
    # compute per token for non-MoE; MoE active is small by design and is a
    # latency lever, not quality).
    def rank(item: tuple[ModelEntry, QuantVariant]) -> tuple[float, float, float]:
        model, quant = item
        q = quant.quality_score.get(problem_set.value, 0.0)
        return (q, model.total_params_b, quant.vram_required_mb)

    candidates.sort(key=rank, reverse=True)
    winner_model, winner_quant = candidates[0]
    decision = RouterDecision(
        model=winner_model,
        quant=winner_quant,
        candidates_considered=considered,
        hardware_budget_mb=budget_mb,
        reason=(
            f"Best fit for {problem_set.value} at budget {budget_mb} MB; "
            f"{len(candidates)} candidate quant(s) fit."
        ),
    )
    log.info(
        "router decision: problem=%s winner=%s/%s candidates=%d budget_mb=%d",
        problem_set.value, winner_model.id, winner_quant.name,
        len(candidates), budget_mb,
    )
    return decision
