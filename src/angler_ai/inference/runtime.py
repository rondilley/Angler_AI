"""llama-cpp-python embedded runtime. One process, one loaded model in v0;
llama-swap proxy is added at v1 (FR-3.7).
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from angler_ai.registry.models import ModelEntry, QuantVariant

log = logging.getLogger(__name__)


# Common CUDA Toolkit roots on Windows. CUDA 13.x reorganized bin -> bin\x64;
# we add both layouts to the DLL search path so llama-cpp-python's cuda build
# finds cudart64_*.dll without depending on the user's PATH being correct.
_WIN_CUDA_TOOLKIT_ROOTS = (
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
)


def _add_cuda_dll_directories() -> list[str]:
    """Make CUDA Toolkit bin directories reachable by the Windows loader.

    Returns the list of paths added. Idempotent. No-op on non-Windows.

    Belt-and-braces approach:
      1. `os.add_dll_directory(path)` for the direct loader hook
      2. Prepend the directory to os.environ['PATH'] so transitive CUDA deps
         (cublas pulls cublasLt, cudnn pulls cudart, etc.) also resolve via
         the default Win32 search order

    Necessary because:
      - llama-cpp-python's CUDA build dynamically links cudart64_*.dll,
        cublas64_*.dll, and (transitively) others
      - CUDA 13.x installs runtime DLLs to `bin\\x64\\` (not the old `bin\\`)
      - The installer does not always add bin\\x64 to PATH automatically
      - Inheriting PATH from a parent process that pre-dates the install
        misses the new location even when the system PATH is correct
    """
    if sys.platform != "win32":
        return []
    added: list[str] = []
    for root in _WIN_CUDA_TOOLKIT_ROOTS:
        root_p = Path(root)
        if not root_p.is_dir():
            continue
        for vdir in sorted(root_p.glob("v*"), reverse=True):
            for sub in ("bin", os.path.join("bin", "x64")):
                bindir = vdir / sub
                if bindir.is_dir() and any(bindir.glob("cudart64_*.dll")):
                    bindir_str = str(bindir)
                    try:
                        os.add_dll_directory(bindir_str)
                    except OSError as exc:  # pragma: no cover - defensive
                        log.debug("add_dll_directory failed for %s: %s", bindir, exc)
                    # Prepend to PATH so transitive deps resolve too.
                    current_path = os.environ.get("PATH", "")
                    if bindir_str not in current_path:
                        os.environ["PATH"] = f"{bindir_str}{os.pathsep}{current_path}"
                    added.append(bindir_str)
    if added:
        log.info("Added CUDA DLL search paths: %s", added)
    return added


@dataclass(slots=True)
class LoadedModel:
    """A loaded llama.cpp model handle.

    Wraps `llama_cpp.Llama` so the rest of the codebase doesn't import the
    third-party type directly.
    """

    model_id: str
    quant: str
    local_path: Path
    n_ctx: int
    n_gpu_layers: int
    handle: object  # llama_cpp.Llama instance; opaque to callers


class InferenceRuntime:
    """v0 single-model runtime. Loads on demand, unloads on swap.

    v1 will front this with llama-swap to support concurrent multi-model use
    cases (e.g. reasoning + embedding in the same query). For v0, the Router
    selects one model per request; if a follow-up requires a different model,
    we unload and reload.
    """

    def __init__(self) -> None:
        self._loaded: LoadedModel | None = None

    @property
    def loaded(self) -> LoadedModel | None:
        return self._loaded

    def ensure_loaded(
        self,
        model: ModelEntry,
        quant: QuantVariant,
        file_path: Path,
        *,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
    ) -> LoadedModel:
        """Load the model if not already loaded; swap if needed.

        Args:
            model: registry entry.
            quant: variant within the entry.
            file_path: local GGUF path produced by the downloader.
            n_ctx: context window allocated for the KV cache. Default 8192;
                callers may bump for long-context problem sets.
            n_gpu_layers: layers to offload to GPU. -1 = all (CUDA / Metal /
                Vulkan / ROCm builds). 0 = CPU only.
        """
        if (
            self._loaded is not None
            and self._loaded.model_id == model.id
            and self._loaded.quant == quant.name
            and self._loaded.n_ctx == n_ctx
        ):
            return self._loaded
        if self._loaded is not None:
            log.info(
                "Unloading %s/%s to load %s/%s",
                self._loaded.model_id, self._loaded.quant, model.id, quant.name,
            )
            self.unload()
        log.info(
            "Loading model %s/%s from %s (n_ctx=%d, n_gpu_layers=%d)",
            model.id, quant.name, file_path, n_ctx, n_gpu_layers,
        )
        # Make CUDA runtime DLLs reachable regardless of PATH state on Windows.
        _add_cuda_dll_directories()
        # Deferred import: llama-cpp-python is heavy; not needed when
        # the caller only wanted the registry router.
        from llama_cpp import Llama
        handle = Llama(
            model_path=str(file_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self._loaded = LoadedModel(
            model_id=model.id,
            quant=quant.name,
            local_path=file_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            handle=handle,
        )
        log.info("Model %s/%s loaded", model.id, quant.name)
        return self._loaded

    def unload(self) -> None:
        """Free the model handle."""
        if self._loaded is not None:
            log.debug("Unloading %s/%s", self._loaded.model_id, self._loaded.quant)
            # Drop the reference; llama_cpp's __del__ frees the native resources.
            self._loaded = None
