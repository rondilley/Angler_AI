"""Tests for the HydroGEM anomaly-detection wrapper.

The model checkpoint is optional in CI — tests that require it are marked
`gpu` (heavy ~16 MB load + inference). The interface invariants are tested
without the checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from angler_ai.prediction.hydrogem import (
    HYDROGEM_CHECKPOINT_FILENAME,
    INPUT_DIM,
    SEQUENCE_LENGTH,
    AnomalyStatus,
    _infer_on_window,
    detect_anomaly,
)

_CKPT = (
    Path(r"C:\Users\rondi\AppData\Local\angler_ai\models\hydrogem")
    / HYDROGEM_CHECKPOINT_FILENAME
)


def test_anomaly_status_required_fields() -> None:
    """Dataclass type + structural invariants."""
    s = AnomalyStatus(
        gauge_id="12345678",
        is_anomalous=True,
        n_anomalous_steps=10,
        total_steps=576,
        max_probability=0.9,
        mean_probability=0.4,
        threshold=0.5,
        model_version="1.0.0",
    )
    assert s.gauge_id == "12345678"
    assert s.is_anomalous is True
    assert 0 <= s.max_probability <= 1
    assert 0 <= s.mean_probability <= 1
    assert s.n_anomalous_steps <= s.total_steps


def test_detect_anomaly_rejects_wrong_shape(tmp_path: Path) -> None:
    """Shape validation is structural - no fabricated answer for bad input."""
    bad = np.zeros((100, 12), dtype=np.float32)  # wrong sequence length
    with pytest.raises(ValueError):
        _infer_on_window(bad, _CKPT)


def test_detect_anomaly_rejects_wrong_dim() -> None:
    bad = np.zeros((SEQUENCE_LENGTH, 8), dtype=np.float32)  # wrong feature count
    with pytest.raises(ValueError):
        _infer_on_window(bad, _CKPT)


def test_detect_anomaly_raises_when_checkpoint_missing(tmp_path: Path) -> None:
    """No checkpoint -> FileNotFoundError, not a silent fabrication."""
    x = np.zeros((SEQUENCE_LENGTH, INPUT_DIM), dtype=np.float32)
    missing = tmp_path / "no_such_checkpoint.pt"
    with pytest.raises(FileNotFoundError):
        _infer_on_window(x, missing)


@pytest.mark.gpu
@pytest.mark.skipif(not _CKPT.exists(), reason="HydroGEM checkpoint not available")
def test_hydrogem_real_inference_on_synthetic_anomaly() -> None:
    """Smoke: real model on a single sample from the published test set.

    Verifies the returned AnomalyStatus is structurally sound; the recall /
    precision evaluation lives in the benchmark notebook.
    """
    import pickle

    test_path = _CKPT.parent / "test_synthetic_mini.pkl"
    if not test_path.exists():
        pytest.skip("test_synthetic_mini.pkl not available")
    with test_path.open("rb") as fh:
        data = pickle.load(fh)
    x, _, meta = data[0]
    if hasattr(x, "numpy"):
        x = x.numpy()
    x = np.asarray(x, dtype=np.float32)
    status = detect_anomaly(x, gauge_id=str(meta.get("site_id", "?")))
    assert 0.0 <= status.max_probability <= 1.0
    assert 0.0 <= status.mean_probability <= 1.0
    assert status.total_steps == SEQUENCE_LENGTH
    assert status.threshold > 0
    assert status.model_version == "1.0.0"
