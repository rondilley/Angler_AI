"""Tests for the model downloader. Offline-friendly: HF Hub calls use a small
in-process patch on HfApi so the suite does not hit the network.

Network-tagged tests run against the live HF Hub when explicitly enabled via
`pytest -m network`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from angler_ai.registry import (
    DownloadResult,
    GGUFNotFoundError,
    QuantVariant,
    read_manifest,
    update_manifest,
)
from angler_ai.registry.downloader import (
    compute_sha256,
    resolve_filename,
    verify_sha256,
)


# ----- pure-logic tests ---------------------------------------------------


def test_compute_sha256_matches_hashlib(tmp_path: Path) -> None:
    """compute_sha256 returns the canonical hex digest of file contents."""
    p = tmp_path / "x.bin"
    payload = b"Angler_AI sha test\n" * 100
    p.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert compute_sha256(p) == expected


def test_verify_sha256_round_trips(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    sha = compute_sha256(p)
    assert verify_sha256(p, sha) is True
    assert verify_sha256(p, "00" * 32) is False


# ----- HF interaction (patched) -------------------------------------------


class _FakeApi:
    def __init__(self, files: list[str]) -> None:
        self._files = files

    def list_repo_files(self, repo: str) -> list[str]:
        return list(self._files)


def test_resolve_filename_picks_matching_glob(monkeypatch) -> None:
    files = [
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "Llama-3.2-3B-Instruct-Q5_K_M.gguf",
        "Llama-3.2-3B-Instruct-Q8_0.gguf",
        "README.md",
    ]
    monkeypatch.setattr(
        "angler_ai.registry.downloader.HfApi",
        lambda *a, **k: _FakeApi(files),
    )
    assert (
        resolve_filename("bartowski/Llama-3.2-3B-Instruct-GGUF", "*Q4_K_M.gguf")
        == "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    )


def test_resolve_filename_prefers_shortest_match(monkeypatch) -> None:
    """When a quant pattern matches multiple files, the shortest filename
    wins (avoids accidentally picking multi-part split files)."""
    files = [
        "Model-Q4_K_M-of-2.gguf",
        "Model-Q4_K_M.gguf",
        "Model-Q4_K_M-extra.gguf",
    ]
    monkeypatch.setattr(
        "angler_ai.registry.downloader.HfApi",
        lambda *a, **k: _FakeApi(files),
    )
    assert resolve_filename("any/repo", "*Q4_K_M*.gguf") == "Model-Q4_K_M.gguf"


def test_resolve_filename_raises_on_no_match(monkeypatch) -> None:
    monkeypatch.setattr(
        "angler_ai.registry.downloader.HfApi",
        lambda *a, **k: _FakeApi(["README.md"]),
    )
    with pytest.raises(GGUFNotFoundError):
        resolve_filename("any/repo", "*Q4_K_M.gguf")


def test_resolve_filename_raises_on_missing_pattern() -> None:
    with pytest.raises(GGUFNotFoundError):
        resolve_filename("any/repo", None)


# ----- manifest -----------------------------------------------------------


def _result(tmp_path: Path, model_id: str = "qwen3.5-4b", quant: str = "Q4_K_M") -> DownloadResult:
    p = tmp_path / f"{model_id}-{quant}.gguf"
    p.write_bytes(b"fake gguf")
    return DownloadResult(
        model_id=model_id,
        quant=quant,
        hf_repo="unsloth/Qwen3.5-4B-GGUF",
        hf_filename="Qwen3.5-4B-Q4_K_M.gguf",
        local_path=str(p),
        sha256=compute_sha256(p),
        license="qwen",
        bytes_downloaded=p.stat().st_size,
        installed_at="2026-06-17T00:00:00+00:00",
    )


def test_manifest_round_trips(tmp_path: Path) -> None:
    mpath = tmp_path / "data_manifest.json"
    r = _result(tmp_path)
    update_manifest(mpath, r)
    data = read_manifest(mpath)
    key = f"{r.model_id}::{r.quant}"
    assert key in data["models"]  # type: ignore[index]
    entry = data["models"][key]  # type: ignore[index]
    assert entry["license"] == "qwen"
    assert entry["sha256"] == r.sha256


def test_manifest_idempotent(tmp_path: Path) -> None:
    """Writing the same DownloadResult twice produces one entry."""
    mpath = tmp_path / "data_manifest.json"
    r = _result(tmp_path)
    update_manifest(mpath, r)
    update_manifest(mpath, r)
    data = json.loads(mpath.read_text(encoding="utf-8"))
    assert len(data["models"]) == 1


def test_read_manifest_returns_empty_skeleton_when_missing(tmp_path: Path) -> None:
    mpath = tmp_path / "missing.json"
    data = read_manifest(mpath)
    assert data == {"models": {}, "datasets": {}}


# ----- license gate -------------------------------------------------------


def test_download_refuses_without_license_acknowledgment(tmp_path: Path) -> None:
    """FR-3.5: never download without explicit license acceptance."""
    from angler_ai.registry.downloader import download
    from angler_ai.registry.models import ModelEntry, ModelType

    entry = ModelEntry(
        id="x", family="x", type=ModelType.TEXT, total_params_b=1.0, active_params_b=1.0,
        context_length=4096, license="qwen", hf_repo="any/repo",
        quants=(QuantVariant(name="Q4_K_M", vram_required_mb=1000, file_size_mb=500,
                             gguf_file_pattern="*Q4_K_M.gguf"),),
    )
    with pytest.raises(PermissionError):
        download(entry, entry.quants[0], tmp_path, license_acknowledged=False)
