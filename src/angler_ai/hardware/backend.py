"""llama.cpp backend selection from a HardwareProfile.

Implements FR-1.8: report the chosen llama.cpp backend variant and wheel index.

The backend selector prefers the highest CUDA wheel whose runtime DLLs are
actually loadable on this host, not just NVML's driver-advertised maximum.
NVML's `nvmlSystemGetCudaDriverVersion` returns the *highest* CUDA version
the driver could support; the wheel needs the matching CUDA *runtime*
libraries (cudart64_*.dll, cublas64_*.dll) to be on PATH. Lesson learned
the hard way during the cu132 install on a host that only had CUDA 12.x
runtime DLLs reachable.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from angler_ai.hardware.models import GPUVendor, HardwareProfile

log = logging.getLogger(__name__)


class Backend(str, Enum):
    """llama.cpp backend variant choices."""

    CPU = "cpu"
    CUDA = "cuda"
    METAL = "metal"
    VULKAN = "vulkan"
    ROCM = "rocm"
    SYCL = "sycl"


@dataclass(frozen=True, slots=True)
class WheelSpec:
    """Pre-built wheel pointer for llama-cpp-python.

    `extra_index_url` is the URL to pass as `--extra-index-url` when installing,
    or None for the default wheel (CPU or Metal on macOS).
    """

    backend: Backend
    extra_index_url: str | None
    note: str = ""
    runtime_cuda_version: str | None = None
    """For Backend.CUDA: which installed CUDA runtime version we matched, vs
    the driver-advertised maximum. Surfaced for transparency."""


# Pre-built wheel indices confirmed in research pass 6. Ordered by preference
# (descending CUDA version) so the iteration picks the highest installed runtime.
CUDA_WHEEL_MATRIX: dict[str, str] = {
    "13.2": "https://abetlen.github.io/llama-cpp-python/whl/cu132",
    "13.0": "https://abetlen.github.io/llama-cpp-python/whl/cu130",
    "12.5": "https://abetlen.github.io/llama-cpp-python/whl/cu125",
    "12.4": "https://abetlen.github.io/llama-cpp-python/whl/cu124",
    "12.3": "https://abetlen.github.io/llama-cpp-python/whl/cu123",
    "12.2": "https://abetlen.github.io/llama-cpp-python/whl/cu122",
    "12.1": "https://abetlen.github.io/llama-cpp-python/whl/cu121",
    "11.8": "https://abetlen.github.io/llama-cpp-python/whl/cu118",
}


# Common Windows CUDA Toolkit install roots. Probed when NVML reports CUDA but
# no cudart DLL is on PATH (the v13.x layout moved bin -> bin\x64).
_WIN_CUDA_TOOLKIT_ROOTS = (
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
)


def detect_installed_cuda_runtimes() -> list[str]:
    """Return CUDA runtime versions whose cudart DLL is reachable on this host.

    Returns versions as strings like '12.5', '13.2'. Empty list if no CUDA
    runtime is installed (or if not on a CUDA-capable OS).
    """
    versions: list[str] = []
    if sys.platform == "win32":
        versions.extend(_detect_win_cuda_runtimes())
    elif sys.platform.startswith("linux"):
        versions.extend(_detect_linux_cuda_runtimes())
    # macOS does not have CUDA; nothing to do.
    return sorted(set(versions), reverse=True)


def _detect_win_cuda_runtimes() -> list[str]:
    """Windows: try LoadLibrary on cudart64_N.dll for N from 11..14; also probe
    the conventional Toolkit install path for v13.x's bin\\x64 layout."""
    versions: list[str] = []
    # Direct LoadLibrary by basename - resolves via PATH.
    for major in (11, 12, 13, 14):
        try:
            ctypes.WinDLL(f"cudart64_{major}.dll")
            versions.append(_runtime_version_from_major(major))
        except OSError:
            continue
    # Probe Toolkit install paths to catch the 13.x bin\x64 layout where the
    # installer did not add bin\x64 to PATH.
    for root in _WIN_CUDA_TOOLKIT_ROOTS:
        root_p = Path(root)
        if not root_p.is_dir():
            continue
        for vdir in root_p.glob("v*"):
            if not vdir.is_dir():
                continue
            vstr = vdir.name.lstrip("v")
            # Check both bin\ and bin\x64\ for cudart.
            for sub in ("bin", os.path.join("bin", "x64")):
                bindir = vdir / sub
                if bindir.is_dir() and any(bindir.glob("cudart64_*.dll")):
                    versions.append(vstr)
                    break
    return versions


def _detect_linux_cuda_runtimes() -> list[str]:
    """Linux: try dlopen on libcudart.so.MAJOR for MAJOR in 11..14, then probe
    /usr/local/cuda* directories."""
    versions: list[str] = []
    for major in (11, 12, 13, 14):
        try:
            ctypes.CDLL(f"libcudart.so.{major}")
            versions.append(_runtime_version_from_major(major))
        except OSError:
            continue
    # Probe /usr/local for cuda-<version> directories.
    for candidate in Path("/usr/local").glob("cuda-*"):
        if not candidate.is_dir():
            continue
        vstr = candidate.name.removeprefix("cuda-")
        if (candidate / "lib64").is_dir():
            versions.append(vstr)
    return versions


def _runtime_version_from_major(major: int) -> str:
    """Loose default minor version per CUDA major. NVML may report a tighter
    minor; we use a representative '.0' here since the wheel matrix's matching
    is governed by `select_wheel_for_cuda` below."""
    return f"{major}.0"


def select_wheel_for_cuda(driver_max_cuda: str | None, installed_runtimes: list[str]) -> tuple[str, str] | None:
    """Pick the CUDA wheel index. Returns (matched_version, wheel_url) or None.

    Logic: among CUDA_WHEEL_MATRIX entries whose major version is also present
    in `installed_runtimes`, pick the highest. Driver-advertised max is a
    secondary upper bound (we never pick a wheel newer than what the driver
    claims to support).
    """
    def major(v: str) -> int:
        return int(v.split(".")[0]) if v else 0

    driver_major = major(driver_max_cuda) if driver_max_cuda else 99
    installed_majors = {major(v) for v in installed_runtimes}
    for ver, url in CUDA_WHEEL_MATRIX.items():
        wheel_major = major(ver)
        if wheel_major > driver_major:
            continue
        if wheel_major in installed_majors:
            return (ver, url)
    return None


def select_backend(
    hp: HardwareProfile,
    *,
    installed_cuda_runtimes: list[str] | None = None,
) -> WheelSpec:
    """Choose a llama.cpp backend for the detected hardware.

    Priority: NVIDIA CUDA whose runtime is installed -> Apple Metal -> AMD ROCm
    -> Intel SYCL -> generic Vulkan -> CPU. Logs candidates so the decision is
    auditable (NFR-6.2).

    Args:
        hp: HardwareProfile from the probe.
        installed_cuda_runtimes: when given, used as the CUDA runtime version
            list instead of probing the live host. Lets callers (tests, dry-runs,
            simulated profiles) drive the selector deterministically.
    """
    for gpu in hp.gpus:
        if gpu.vendor == GPUVendor.NVIDIA:
            installed = (
                installed_cuda_runtimes
                if installed_cuda_runtimes is not None
                else detect_installed_cuda_runtimes()
            )
            match = select_wheel_for_cuda(gpu.cuda_version, installed)
            if match is not None:
                matched_ver, url = match
                note = (
                    f"CUDA {matched_ver} (installed runtime; driver advertises "
                    f"max {gpu.cuda_version or 'unknown'}) on {gpu.name}"
                )
                return WheelSpec(
                    backend=Backend.CUDA,
                    extra_index_url=url,
                    note=note,
                    runtime_cuda_version=matched_ver,
                )
            note = (
                f"NVIDIA {gpu.name} detected; driver supports CUDA "
                f"{gpu.cuda_version or 'unknown'} but no matching CUDA runtime "
                f"DLLs found on PATH or in standard Toolkit paths. "
                f"Install the CUDA Toolkit or use the Vulkan wheel."
            )
            return WheelSpec(
                backend=Backend.VULKAN,
                extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/vulkan",
                note=note,
            )
        if gpu.vendor == GPUVendor.APPLE:
            return WheelSpec(
                backend=Backend.METAL,
                extra_index_url=None,
                note="Apple Silicon Metal (default macOS wheel)",
            )
        if gpu.vendor == GPUVendor.AMD:
            return WheelSpec(
                backend=Backend.ROCM,
                extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/rocm",
                note=f"AMD ROCm on {gpu.name}",
            )
        if gpu.vendor == GPUVendor.INTEL:
            return WheelSpec(
                backend=Backend.SYCL,
                extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/sycl",
                note=f"Intel oneAPI SYCL on {gpu.name}",
            )
    if hp.gpus:
        return WheelSpec(
            backend=Backend.VULKAN,
            extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/vulkan",
            note="Generic GPU detected; Vulkan fallback",
        )
    return WheelSpec(
        backend=Backend.CPU,
        extra_index_url=None,
        note="CPU-only inference path",
    )
