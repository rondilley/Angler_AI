"""Model registry types. Capability metadata, quant variants, problem-set bindings."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class ModelType(str, Enum):
    """Model modality / kind."""

    TEXT = "text"
    """Causal LM for reasoning + tool use."""

    EMBEDDING = "embedding"
    """Sentence-embedding model."""

    RERANKER = "reranker"

    VISION = "vision"
    """Vision-language model (text + image)."""

    ASR = "asr"
    """Automatic speech recognition (Whisper.cpp etc.)."""


class Capability(str, Enum):
    """Capabilities a model declares; used by the router."""

    TOOL_USE = "tool_use"
    """OpenAI-style function/tool calling."""

    JSON_SCHEMA = "json_schema"
    """Structurally-constrained output via response_format."""

    LONG_CONTEXT_8K = "long_context_8k"
    LONG_CONTEXT_16K = "long_context_16k"
    LONG_CONTEXT_32K = "long_context_32k"
    LONG_CONTEXT_64K = "long_context_64k"
    LONG_CONTEXT_128K = "long_context_128k"

    VISION_INPUT = "vision_input"
    EMBEDDING = "embedding"
    RERANKING = "reranking"
    MULTILINGUAL = "multilingual"
    INSTRUCTION_FOLLOWING = "instruction_following"


class ProblemSet(str, Enum):
    """Internal problem-set declaration the Router maps to a model.

    Each Reasoning-Layer query is tagged with one or more ProblemSet values;
    the Router picks a model whose capabilities and quants fit the user's
    hardware. Adding a new ProblemSet must not require Router changes (NFR-7.3).
    """

    REGULATION_QA = "regulation_qa"
    """Long-context Q&A over state regulations."""

    ANGLER_REPORT_EXTRACTION = "angler_report_extraction"
    """Structured extraction from free-text angler reports."""

    ANOMALY_EXPLANATION = "anomaly_explanation"
    """Plain-English explanation of HydroGEM output."""

    EXPLAIN_REACH = "explain_reach"
    """'Why is this reach good right now' multi-source narrative."""

    SEMANTIC_SEARCH = "semantic_search"
    """Embedding + reranking over papers, regs, angler reports."""

    FISH_PHOTO_ID = "fish_photo_id"
    """Vision-language fish species identification."""

    VOICE_NOTE_TRANSCRIPTION = "voice_note_transcription"


@dataclass(frozen=True, slots=True)
class QuantVariant:
    """A single quantization of a model."""

    name: str
    """e.g. 'Q4_K_M', 'IQ2_M', 'Q4_K_XL'."""

    vram_required_mb: int
    """Measured VRAM at load + a typical 4K context. Acceptance gate updates this."""

    file_size_mb: int

    gguf_file_pattern: str | None = None
    """Glob pattern that matches this quant's GGUF in the HF repo."""

    quality_score: dict[str, float] = field(default_factory=dict)
    """Per-ProblemSet eval score from our acceptance gates. NOT marketing MMLU
    (Unsloth Dynamic 2.0 numbers were refuted in pass 6; CR-4.5)."""


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """A model in the registry."""

    id: str
    """Stable id, e.g. 'qwen3.5-35b-a3b'."""

    family: str
    type: ModelType
    total_params_b: float
    active_params_b: float | None
    """MoE only; None for dense. Drives latency budget."""

    context_length: int
    license: str
    """SPDX-like id, e.g. 'apache-2.0', 'qwen', 'llama-community'."""

    hf_repo: str
    """HuggingFace repo id, e.g. 'unsloth/Qwen3.5-35B-A3B-GGUF'."""

    quants: tuple[QuantVariant, ...] = ()
    capabilities: frozenset[Capability] = frozenset()
    recommended_for: frozenset[ProblemSet] = frozenset()


def load_catalog(path: Path | None = None) -> tuple[ModelEntry, ...]:
    """Load the seed catalog YAML and return a tuple of ModelEntry.

    Defaults to the shipped `catalog.yaml` in this package.
    """
    if path is None:
        path = Path(__file__).parent / "catalog.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return tuple(_parse_entry(item) for item in raw.get("models", ()))


def _parse_entry(item: dict[str, Any]) -> ModelEntry:
    quants = tuple(
        QuantVariant(
            name=q["name"],
            vram_required_mb=int(q["vram_required_mb"]),
            file_size_mb=int(q.get("file_size_mb", 0)),
            gguf_file_pattern=q.get("gguf_file_pattern"),
            quality_score=dict(q.get("quality_score", {})),
        )
        for q in item.get("quants", ())
    )
    capabilities = frozenset(Capability(c) for c in item.get("capabilities", ()))
    recommended_for = frozenset(ProblemSet(p) for p in item.get("recommended_for", ()))
    return ModelEntry(
        id=item["id"],
        family=item["family"],
        type=ModelType(item["type"]),
        total_params_b=float(item["total_params_b"]),
        active_params_b=(
            float(item["active_params_b"]) if item.get("active_params_b") is not None else None
        ),
        context_length=int(item["context_length"]),
        license=item["license"],
        hf_repo=item["hf_repo"],
        quants=quants,
        capabilities=capabilities,
        recommended_for=recommended_for,
    )
