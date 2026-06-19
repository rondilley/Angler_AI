"""Tests for HP - FR-1.x. Runs against the actual host (cannot fake psutil/archspec
without losing the value of the test). Reference-platform tests are marked.
"""

from __future__ import annotations

import pytest

from angler_ai.hardware import probe
from angler_ai.hardware.backend import Backend, select_backend


@pytest.mark.reference_platform
def test_probe_completes_and_populates_cpu_and_ram() -> None:
    """FR-1.1, FR-1.2: probe returns CPU + RAM detail on any host."""
    hp = probe()
    assert hp.cpu.cores_logical >= 1
    assert hp.cpu.cores_physical >= 1
    assert hp.cpu.arch
    assert hp.ram.total_gb > 0
    assert hp.ram.available_gb >= 0
    assert hp.ram.available_gb <= hp.ram.total_gb


@pytest.mark.reference_platform
def test_probe_reports_os_and_python() -> None:
    """FR-1.8, FR-1.9: probe reports OS and Python version."""
    hp = probe()
    assert hp.os.system
    assert hp.python.major == 3
    # FR-1.9: Python must be in 3.10-3.12 for v0.
    assert hp.python.supported, f"Python {hp.python.major}.{hp.python.minor} is outside the v0 wheel matrix"


def test_select_backend_cpu_only(cpu_only_hardware) -> None:
    """No GPU -> CPU backend (no extra index URL)."""
    spec = select_backend(cpu_only_hardware)
    assert spec.backend == Backend.CPU
    assert spec.extra_index_url is None


def test_select_backend_cuda(consumer_gpu_hardware) -> None:
    """NVIDIA GPU with supported CUDA + installed CUDA 12.5 runtime -> cu125 wheel."""
    spec = select_backend(consumer_gpu_hardware, installed_cuda_runtimes=["12.5"])
    assert spec.backend == Backend.CUDA
    assert spec.extra_index_url is not None
    assert "cu125" in spec.extra_index_url
    assert spec.runtime_cuda_version is not None
    assert spec.runtime_cuda_version.startswith("12.")


def test_select_backend_cuda_no_runtime_falls_back_to_vulkan(consumer_gpu_hardware) -> None:
    """NVIDIA GPU but no installed CUDA runtime -> Vulkan fallback (not silent CPU)."""
    spec = select_backend(consumer_gpu_hardware, installed_cuda_runtimes=[])
    assert spec.backend == Backend.VULKAN
    assert "vulkan" in (spec.extra_index_url or "")
    # The note must explain why we did not pick CUDA.
    assert "CUDA" in spec.note


def test_select_backend_prefers_highest_installed_cuda(consumer_gpu_hardware) -> None:
    """Multiple CUDA runtimes installed -> highest one within driver-advertised max."""
    spec = select_backend(
        consumer_gpu_hardware,
        installed_cuda_runtimes=["11.8", "12.1", "12.5"],
    )
    assert spec.backend == Backend.CUDA
    assert "cu125" in (spec.extra_index_url or "")


def test_select_backend_apple_silicon(apple_silicon_hardware) -> None:
    """Apple Silicon -> Metal."""
    spec = select_backend(apple_silicon_hardware)
    assert spec.backend == Backend.METAL


def test_available_vram_unified_memory_uses_gpu_total(apple_silicon_hardware) -> None:
    """Unified memory devices report VRAM mirroring the budget RAM fraction."""
    vram = apple_silicon_hardware.available_vram_mb(margin=0.0)
    # Apple Silicon fixture allocates 60% of 30 GB as budget.
    assert vram > 15_000


def test_available_vram_cpu_only_uses_ram(cpu_only_hardware) -> None:
    """No GPUs: budget falls back to a fraction of RAM."""
    vram = cpu_only_hardware.available_vram_mb(margin=0.0)
    assert vram > 0
