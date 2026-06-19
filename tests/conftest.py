"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from angler_ai.hardware.models import (
    CPUInfo,
    GPUDevice,
    GPUVendor,
    HardwareProfile,
    OSInfo,
    PythonInfo,
    RAMInfo,
)


@pytest.fixture
def cpu_only_hardware() -> HardwareProfile:
    """Synthetic CPU-only HardwareProfile."""
    return HardwareProfile(
        cpu=CPUInfo(
            arch="x86_64", microarch="haswell",
            features=frozenset({"avx", "avx2"}),
            cores_physical=4, cores_logical=8,
        ),
        ram=RAMInfo(total_gb=16.0, available_gb=12.0),
        gpus=(),
        os=OSInfo(system="Linux", release="6.1", machine="x86_64"),
        python=PythonInfo(major=3, minor=11, micro=5),
    )


@pytest.fixture
def consumer_gpu_hardware() -> HardwareProfile:
    """Synthetic 16 GB NVIDIA RTX-class profile (mid tier)."""
    return HardwareProfile(
        cpu=CPUInfo(
            arch="x86_64", microarch="zen4",
            features=frozenset({"avx", "avx2", "avx512f"}),
            cores_physical=16, cores_logical=32,
        ),
        ram=RAMInfo(total_gb=64.0, available_gb=48.0),
        gpus=(
            GPUDevice(
                vendor=GPUVendor.NVIDIA,
                name="NVIDIA GeForce RTX 4080",
                vram_total_mb=16384,
                vram_available_mb=15000,
                compute_capability="8.9",
                driver_version="555.85",
                cuda_version="12.5",
            ),
        ),
        os=OSInfo(system="Windows", release="11", machine="AMD64"),
        python=PythonInfo(major=3, minor=11, micro=8),
    )


@pytest.fixture
def apple_silicon_hardware() -> HardwareProfile:
    """Synthetic Apple Silicon profile (unified memory)."""
    return HardwareProfile(
        cpu=CPUInfo(
            arch="arm64", microarch="m3",
            features=frozenset({"neon"}),
            cores_physical=12, cores_logical=12,
        ),
        ram=RAMInfo(total_gb=36.0, available_gb=30.0),
        gpus=(
            GPUDevice(
                vendor=GPUVendor.APPLE,
                name="Apple Silicon GPU (Metal)",
                vram_total_mb=int(36 * 1024 * 0.60),
                vram_available_mb=int(30 * 1024 * 0.60),
                unified_memory=True,
            ),
        ),
        os=OSInfo(system="Darwin", release="23.0", machine="arm64"),
        python=PythonInfo(major=3, minor=12, micro=2),
    )
