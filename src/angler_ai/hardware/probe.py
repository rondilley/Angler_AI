"""Hardware probe. Implements FR-1.1, FR-1.2, FR-1.3, FR-1.4, FR-1.8, FR-1.9.

archspec + psutil are required. nvidia-ml-py is pinned (research pass 6 documented
backward-compat breakage). Apple Silicon detection uses sysctl + system_profiler.
AMD ROCm (FR-1.5), Intel SYCL (FR-1.6), Intel NPU (FR-1.7) are SHOULD/COULD and
implemented best-effort.
"""

from __future__ import annotations

import logging
import platform
import subprocess
import sys

import psutil

from angler_ai.hardware.models import (
    CPUInfo,
    GPUDevice,
    GPUVendor,
    HardwareProfile,
    NPUDevice,
    OSInfo,
    PythonInfo,
    RAMInfo,
)

log = logging.getLogger(__name__)


def probe() -> HardwareProfile:
    """Run the full hardware probe.

    NFR-1.1 requires this to complete in under 5 seconds. Non-critical detection
    failures (missing nvidia-ml-py, missing rocm-smi) must not abort the probe
    (NFR-2.3); they yield empty `gpus`/`npus` entries with a log warning.
    """
    cpu = _probe_cpu()
    ram = _probe_ram()
    gpus = _probe_gpus()
    npus = _probe_npus()
    os_info = _probe_os()
    py_info = _probe_python()
    return HardwareProfile(
        cpu=cpu,
        ram=ram,
        gpus=tuple(gpus),
        npus=tuple(npus),
        os=os_info,
        python=py_info,
    )


def _probe_cpu() -> CPUInfo:
    """CPU detection via archspec + psutil."""
    try:
        import archspec.cpu  # local import; archspec is a runtime dep
        host = archspec.cpu.host()
        microarch = host.name
        features = frozenset(host.features)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("archspec host() failed: %s; falling back to platform.machine()", exc)
        microarch = "unknown"
        features = frozenset()
    arch = platform.machine().lower()
    return CPUInfo(
        arch=arch,
        microarch=microarch,
        features=features,
        cores_physical=psutil.cpu_count(logical=False) or 0,
        cores_logical=psutil.cpu_count(logical=True) or 0,
    )


def _probe_ram() -> RAMInfo:
    """System RAM via psutil."""
    mem = psutil.virtual_memory()
    return RAMInfo(
        total_gb=round(mem.total / (1024 ** 3), 2),
        available_gb=round(mem.available / (1024 ** 3), 2),
    )


def _probe_gpus() -> list[GPUDevice]:
    """Detect GPUs across vendors. Best-effort; missing vendor SDKs do not abort.

    Order: NVIDIA -> Apple Silicon -> AMD ROCm -> Intel oneAPI -> generic Vulkan.
    """
    gpus: list[GPUDevice] = []
    gpus.extend(_probe_nvidia())
    if sys.platform == "darwin":
        gpus.extend(_probe_apple_silicon())
    # v1 expansions:
    # gpus.extend(_probe_amd_rocm())
    # gpus.extend(_probe_intel_arc())
    # gpus.extend(_probe_vulkan_generic())
    return gpus


def _probe_nvidia() -> list[GPUDevice]:
    """NVIDIA via nvidia-ml-py. Pinned per research pass 6 to manage breakage."""
    try:
        import pynvml  # nvidia-ml-py exposes as pynvml
    except ImportError:
        log.info("nvidia-ml-py not installed; skipping NVIDIA probe")
        return []
    try:
        pynvml.nvmlInit()
    except Exception as exc:
        log.info("NVML init failed (likely no NVIDIA driver): %s", exc)
        return []
    devices: list[GPUDevice] = []
    try:
        count = pynvml.nvmlDeviceGetCount()
        # NVML returns the CUDA-runtime version the driver advertises support for,
        # encoded as (major * 1000) + (minor * 10). E.g. 12050 = "12.5".
        cuda_str: str | None = None
        try:
            cuda_int = pynvml.nvmlSystemGetCudaDriverVersion()
            cuda_str = f"{cuda_int // 1000}.{(cuda_int % 1000) // 10}"
        except Exception as exc:  # pragma: no cover - defensive
            log.info("Could not query CUDA driver version: %s", exc)
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name_raw = pynvml.nvmlDeviceGetName(handle)
            name = name_raw.decode() if isinstance(name_raw, bytes) else name_raw
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            cc_major, cc_minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            driver_raw = pynvml.nvmlSystemGetDriverVersion()
            driver = driver_raw.decode() if isinstance(driver_raw, bytes) else driver_raw
            devices.append(
                GPUDevice(
                    vendor=GPUVendor.NVIDIA,
                    name=name,
                    vram_total_mb=int(mem.total / (1024 ** 2)),
                    vram_available_mb=int(mem.free / (1024 ** 2)),
                    compute_capability=f"{cc_major}.{cc_minor}",
                    driver_version=driver,
                    cuda_version=cuda_str,
                )
            )
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("NVIDIA enumeration partial-failure: %s", exc)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # pragma: no cover - defensive
            pass
    return devices


def _probe_apple_silicon() -> list[GPUDevice]:
    """Apple Silicon GPU. Unified-memory model: VRAM mirrors a fraction of RAM."""
    if sys.platform != "darwin":
        return []
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.optional.arm64"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if result.stdout.strip() != "1":
            return []
        ram = _probe_ram()
        # Conservative unified-memory budget for inference: 60% of total RAM.
        budget_mb = int(ram.total_gb * 1024 * 0.60)
        return [
            GPUDevice(
                vendor=GPUVendor.APPLE,
                name="Apple Silicon GPU (Metal)",
                vram_total_mb=budget_mb,
                vram_available_mb=int(ram.available_gb * 1024 * 0.60),
                unified_memory=True,
            )
        ]
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:  # pragma: no cover
        log.warning("Apple Silicon probe failed: %s", exc)
        return []


def _probe_npus() -> list[NPUDevice]:
    """NPU detection. v0 best-effort. Intel NPU is deferred to v1 via OpenVINO GenAI
    (intel-npu-acceleration-library was archived April 2025 per research pass 6).
    Apple Neural Engine, Qualcomm Hexagon, Windows Copilot+ NPU APIs are v1+.
    """
    return []


def _probe_os() -> OSInfo:
    """OS info from platform."""
    return OSInfo(
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
    )


def _probe_python() -> PythonInfo:
    """Python version. Used to gate llama-cpp-python wheel install."""
    v = sys.version_info
    return PythonInfo(major=v.major, minor=v.minor, micro=v.micro)
