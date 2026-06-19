"""HydroGEM-style USGS NWIS anomaly detection (FR-5.4).

Loads the published Ejokhan/HydroGEM checkpoint and runs real inference on
USGS streamflow sequences. Returns concrete `AnomalyStatus` values with the
model's per-timestep anomaly probability and calibrated mask.

License of the model weights: CC-BY-NC-4.0 (non-commercial). Surfaced in the
model registry / manifest at download time.

NO FABRICATED VALUES. If the checkpoint is not downloaded, `detect_anomaly`
raises FileNotFoundError; if the input is malformed, raises ValueError.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from angler_ai.prediction.hydrogem_arch import (
    HydrogemConfig,
    HydrogemFoundationModel,
    MultiScaleAnomalyHead,
    deploy_safe_features,
    morphological_closing,
)

log = logging.getLogger(__name__)

HYDROGEM_HF_REPO = "Ejokhan/HydroGEM"
HYDROGEM_CHECKPOINT_FILENAME = "hydrogem_inference .pt"  # space in filename per upstream

SEQUENCE_LENGTH = 576  # hours; the architecture's fixed window
INPUT_DIM = 12


@dataclass(frozen=True, slots=True)
class AnomalyStatus:
    """Anomaly detection output for a USGS NWIS gauge over a window."""

    gauge_id: str
    is_anomalous: bool
    """True if any timestep crossed the calibrated threshold + morphology."""

    n_anomalous_steps: int
    """Count of timesteps flagged anomalous after smoothing + closing."""

    total_steps: int
    """Length of the window analyzed."""

    max_probability: float
    """Highest per-timestep anomaly probability seen in the window."""

    mean_probability: float
    """Mean per-timestep anomaly probability across the window."""

    threshold: float
    """Calibrated threshold from the checkpoint."""

    model_version: str
    """HydroGEM checkpoint version string."""


_LOADED: dict[str, Any] = {}


def _load_model(checkpoint_path: Path, device: torch.device) -> dict:
    """Load (model, anomaly_head, calibration, global_stats) once and cache."""
    key = str(checkpoint_path) + ":" + str(device)
    if key in _LOADED:
        return _LOADED[key]
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"HydroGEM checkpoint not found at {checkpoint_path}. "
            f"Download with `from huggingface_hub import hf_hub_download; "
            f"hf_hub_download({HYDROGEM_HF_REPO!r}, {HYDROGEM_CHECKPOINT_FILENAME!r})`."
        )
    log.info("Loading HydroGEM checkpoint: %s on %s", checkpoint_path, device)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    config = HydrogemConfig.from_dict(ckpt.get("config", {}))
    model = HydrogemFoundationModel(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    anomaly_head = MultiScaleAnomalyHead(in_dim=11, hidden_dim=128, scales=(1, 4)).to(device)
    anomaly_head.load_state_dict(ckpt["anomaly_head_state_dict"])
    anomaly_head.eval()
    payload = {
        "model": model,
        "anomaly_head": anomaly_head,
        "global_stats": ckpt["global_stats"],
        "calibration": ckpt["calibration"],
        "version": ckpt.get("hydrogem_version", "unknown"),
    }
    _LOADED[key] = payload
    n_params = sum(p.numel() for p in model.parameters()) + sum(
        p.numel() for p in anomaly_head.parameters()
    )
    log.info("HydroGEM loaded: %s params, T_det=%.4f, threshold=%.4f, version=%s",
             f"{n_params:,}", payload["calibration"]["T_det"],
             payload["calibration"]["threshold"], payload["version"])
    return payload


def _infer_on_window(
    x: np.ndarray,
    checkpoint_path: Path,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run HydroGEM on a (T, INPUT_DIM) numpy window. Returns (probs, mask)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if x.ndim != 2 or x.shape[0] != SEQUENCE_LENGTH or x.shape[1] != INPUT_DIM:
        raise ValueError(
            f"HydroGEM expects window shape ({SEQUENCE_LENGTH}, {INPUT_DIM}); "
            f"got {x.shape}. The published checkpoint is fixed-window."
        )
    payload = _load_model(checkpoint_path, device)
    model = payload["model"]
    head = payload["anomaly_head"]
    cal = payload["calibration"]
    t_det = float(cal["T_det"])
    threshold = float(cal["threshold"])

    x_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x_t)
        feat = deploy_safe_features(out, x_t, payload["global_stats"])
        logits = head(feat)
        probs = torch.sigmoid(logits.squeeze(-1) / t_det)
        probs_smooth = torch.nn.functional.avg_pool1d(
            probs.unsqueeze(1), kernel_size=5, stride=1, padding=2,
        ).squeeze(1)
        mask = morphological_closing(probs_smooth >= threshold, kernel_size=7)
    return probs_smooth.squeeze(0).cpu().numpy(), mask.squeeze(0).cpu().numpy()


def detect_anomaly(
    x: np.ndarray,
    *,
    gauge_id: str,
    checkpoint_path: Path | None = None,
    device: torch.device | None = None,
) -> AnomalyStatus:
    """Run HydroGEM on one USGS NWIS window.

    Args:
        x: Pre-normalized input window with shape (SEQUENCE_LENGTH, INPUT_DIM)
           matching the published HydroGEM USGS feature stack.
        gauge_id: USGS gauge identifier for attribution.
        checkpoint_path: Path to `hydrogem_inference .pt`. Defaults to the
           tool's canonical model cache.
        device: torch device. Defaults to cuda if available, else cpu.

    Returns:
        AnomalyStatus with REAL model outputs. NO FABRICATED VALUES.
    """
    if checkpoint_path is None:
        from angler_ai.config import default_paths
        checkpoint_path = default_paths().cache_dir / "models" / "hydrogem" / HYDROGEM_CHECKPOINT_FILENAME
    probs, mask = _infer_on_window(x, checkpoint_path, device)
    payload = _LOADED[str(checkpoint_path) + ":" + str(device or (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ))]
    return AnomalyStatus(
        gauge_id=gauge_id,
        is_anomalous=bool(mask.any()),
        n_anomalous_steps=int(mask.sum()),
        total_steps=int(mask.shape[0]),
        max_probability=float(probs.max()),
        mean_probability=float(probs.mean()),
        threshold=float(payload["calibration"]["threshold"]),
        model_version=str(payload["version"]),
    )
