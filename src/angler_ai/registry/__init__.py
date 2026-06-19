"""Model Registry and Router (MR). Capability-based model selection.

See FR-3, design 5.3, research/06_hardware_adaptive_llama_cpp_research_2026-06-17.md.
"""

from angler_ai.registry.downloader import (
    DownloadResult,
    GGUFNotFoundError,
    download,
    read_manifest,
    update_manifest,
)
from angler_ai.registry.models import (
    Capability,
    ModelEntry,
    ModelType,
    ProblemSet,
    QuantVariant,
    load_catalog,
)
from angler_ai.registry.router import NoModelFitsError, RouterDecision, select_model

__all__ = [
    "Capability",
    "DownloadResult",
    "GGUFNotFoundError",
    "ModelEntry",
    "ModelType",
    "NoModelFitsError",
    "ProblemSet",
    "QuantVariant",
    "RouterDecision",
    "download",
    "load_catalog",
    "read_manifest",
    "select_model",
    "update_manifest",
]
