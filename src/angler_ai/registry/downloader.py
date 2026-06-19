"""GGUF acquisition. Implements FR-3.3 (HF Hub download), FR-3.4 (SHA verify),
FR-3.5 (license surfacing on first download).

hf_transfer is enabled best-effort (research pass 6 noted the activation env-var
name was refuted; we re-verify against the live huggingface_hub at integration).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

from angler_ai.registry.models import ModelEntry, QuantVariant

log = logging.getLogger(__name__)


@dataclass(slots=True)
class DownloadResult:
    """Outcome of a model download. Persisted in the data manifest (DR-2.4)."""

    model_id: str
    quant: str
    hf_repo: str
    hf_filename: str
    local_path: str
    sha256: str
    license: str
    bytes_downloaded: int
    installed_at: str


class GGUFNotFoundError(RuntimeError):
    """No file in the HF repo matched the quant's glob pattern."""


def enable_hf_transfer_if_available() -> bool:
    """Best-effort enable hf_transfer for faster downloads.

    The env-var name was refuted in research pass 6; we set the documented
    name optimistically. If hf_transfer is unavailable or the env var is
    inert, the standard download path still works.
    Returns True if hf_transfer is importable.
    """
    try:
        import hf_transfer  # noqa: F401
    except ImportError:
        return False
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    log.info("hf_transfer available; enabled via HF_HUB_ENABLE_HF_TRANSFER")
    return True


def resolve_filename(repo: str, glob_pattern: str | None) -> str:
    """List files in an HF repo and return the first one matching `glob_pattern`.

    Raises GGUFNotFoundError when no file matches. RepositoryNotFoundError
    propagates if the repo itself does not exist.
    """
    if glob_pattern is None:
        raise GGUFNotFoundError(f"Quant in repo {repo!r} has no gguf_file_pattern")
    api = HfApi()
    try:
        files = api.list_repo_files(repo)
    except RepositoryNotFoundError as exc:
        raise GGUFNotFoundError(
            f"HuggingFace repo {repo!r} not found (gated / typo / removed): {exc}"
        ) from exc
    matches = [f for f in files if fnmatch.fnmatch(f, glob_pattern)]
    if not matches:
        raise GGUFNotFoundError(
            f"No file in {repo!r} matched pattern {glob_pattern!r}. "
            f"Repo has {len(files)} files; first 5: {files[:5]}"
        )
    # If multiple match, prefer the shortest filename (avoids picking
    # multi-part splits like '-of-2' when a single-file variant exists).
    matches.sort(key=len)
    return matches[0]


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 of a file. Used to populate the manifest on first download (FR-3.4).
    Subsequent loads verify against the manifest record.
    """
    h = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256(file_path: Path, expected_hex: str) -> bool:
    """Compare SHA256 against the manifest record. FR-3.4."""
    actual = compute_sha256(file_path)
    ok = actual.lower() == expected_hex.lower()
    if not ok:
        log.error("SHA mismatch for %s: expected %s, got %s", file_path, expected_hex, actual)
    return ok


def download(
    model: ModelEntry,
    quant: QuantVariant,
    cache_dir: Path,
    *,
    license_acknowledged: bool = False,
) -> DownloadResult:
    """Download a quant from HuggingFace Hub into cache_dir.

    Args:
        model: registry entry to acquire.
        quant: variant within the entry (defines the glob pattern).
        cache_dir: destination root (per config.Paths.models).
        license_acknowledged: caller has surfaced the license to the user and
            recorded acknowledgment (FR-3.5). Pass True from `pull-models`
            after the user accepts, or pass True from non-interactive paths
            that handle license elsewhere.

    Returns:
        DownloadResult.

    Raises:
        GGUFNotFoundError: when no file in the repo matches the quant pattern.
        PermissionError: when license_acknowledged is False.
    """
    if not license_acknowledged:
        raise PermissionError(
            f"License {model.license!r} for {model.id} must be acknowledged before download (FR-3.5)."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    enable_hf_transfer_if_available()
    filename = resolve_filename(model.hf_repo, quant.gguf_file_pattern)
    log.info(
        "Downloading %s/%s (license=%s) from %s -> %s",
        model.id, quant.name, model.license, model.hf_repo, filename,
    )
    target_dir = cache_dir / model.id
    target_dir.mkdir(parents=True, exist_ok=True)
    local_path = Path(
        hf_hub_download(
            repo_id=model.hf_repo,
            filename=filename,
            local_dir=str(target_dir),
        )
    )
    size = local_path.stat().st_size
    sha = compute_sha256(local_path)
    result = DownloadResult(
        model_id=model.id,
        quant=quant.name,
        hf_repo=model.hf_repo,
        hf_filename=filename,
        local_path=str(local_path),
        sha256=sha,
        license=model.license,
        bytes_downloaded=size,
        installed_at=datetime.now(timezone.utc).isoformat(),
    )
    log.info(
        "Downloaded %s/%s: %d bytes, sha256=%s",
        model.id, quant.name, size, sha,
    )
    return result


def update_manifest(manifest_path: Path, result: DownloadResult) -> None:
    """Append/update a model entry in the data manifest. Idempotent."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        data = {"models": {}, "datasets": {}}
    key = f"{result.model_id}::{result.quant}"
    data.setdefault("models", {})[key] = asdict(result)
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    log.info("Manifest updated: %s = %s", manifest_path, key)


def read_manifest(manifest_path: Path) -> dict[str, object]:
    """Return the current manifest contents, or empty skeleton if missing."""
    if not manifest_path.exists():
        return {"models": {}, "datasets": {}}
    return json.loads(manifest_path.read_text(encoding="utf-8"))
