"""Hardware profile dataclasses. Frozen, slot-based, JSON-serializable."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class GPUVendor(str, Enum):
    """GPU vendor families llama.cpp may target."""

    NVIDIA = "nvidia"
    AMD = "amd"
    APPLE = "apple"
    INTEL = "intel"
    VULKAN_GENERIC = "vulkan-generic"


@dataclass(frozen=True, slots=True)
class GPUDevice:
    """A single GPU device discovered by the probe."""

    vendor: GPUVendor
    name: str
    vram_total_mb: int
    vram_available_mb: int
    compute_capability: str | None = None
    """NVIDIA-only, e.g. '8.6'."""
    driver_version: str | None = None
    cuda_version: str | None = None
    """NVIDIA-only."""
    unified_memory: bool = False
    """True for Apple Silicon (vram_* mirrors system RAM budget)."""


@dataclass(frozen=True, slots=True)
class NPUDevice:
    """An NPU device. Best-effort detection; often empty in v0 (FR-1.5..7 are SHOULD/COULD)."""

    vendor: str
    name: str
    note: str = ""


@dataclass(frozen=True, slots=True)
class CPUInfo:
    """CPU detail from archspec + psutil."""

    arch: str
    """One of 'x86_64', 'arm64', 'aarch64', etc."""

    microarch: str
    """archspec microarchitecture name, e.g. 'icelake', 'm1', 'zen3'."""

    features: frozenset[str]
    """e.g. frozenset({'avx2', 'avx512f'})."""

    cores_physical: int
    cores_logical: int


@dataclass(frozen=True, slots=True)
class RAMInfo:
    """System memory snapshot."""

    total_gb: float
    available_gb: float


@dataclass(frozen=True, slots=True)
class OSInfo:
    """OS identification used to choose download wheels and backend builds."""

    system: str
    """'Windows', 'Darwin', 'Linux'."""

    release: str
    machine: str
    """Architecture string from platform.machine()."""


@dataclass(frozen=True, slots=True)
class PythonInfo:
    """Python version, gated at 3.10-3.12 for llama-cpp-python wheels (FR-1.9)."""

    major: int
    minor: int
    micro: int

    @property
    def supported(self) -> bool:
        return self.major == 3 and 10 <= self.minor <= 12


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Full hardware profile snapshot. Output of `probe()`."""

    cpu: CPUInfo
    ram: RAMInfo
    gpus: tuple[GPUDevice, ...] = ()
    npus: tuple[NPUDevice, ...] = ()
    os: OSInfo = field(default_factory=lambda: OSInfo("", "", ""))
    python: PythonInfo = field(default_factory=lambda: PythonInfo(0, 0, 0))

    def available_vram_mb(self, margin: float = 0.10) -> int:
        """Return the available VRAM minus a safety margin, across all GPUs.

        For unified-memory devices (Apple Silicon), this mirrors the RAM budget.
        """
        total_available = sum(g.vram_available_mb for g in self.gpus)
        if total_available == 0:
            # CPU-only path uses a fraction of RAM as the working set.
            total_available = int(self.ram.available_gb * 1024)
        return int(total_available * (1.0 - margin))
