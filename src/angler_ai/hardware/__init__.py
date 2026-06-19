"""Hardware Probe (HP) - detect compute, RAM, GPU/VRAM, NPU; choose llama.cpp backend.

See FR-1, design 5.1, research/06_hardware_adaptive_llama_cpp_research_2026-06-17.md.
"""

from angler_ai.hardware.models import GPUDevice, GPUVendor, HardwareProfile, NPUDevice
from angler_ai.hardware.probe import probe

__all__ = ["GPUDevice", "GPUVendor", "HardwareProfile", "NPUDevice", "probe"]
