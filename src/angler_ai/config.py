"""Project-wide configuration: paths, defaults, and environment-derived settings.

Honors XDG Base Directory specification on Linux/macOS and uses the conventional
LocalAppData on Windows.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _xdg_or_default(env_var: str, default_subpath: str) -> Path:
    """Return XDG-style path or fall back to OS-appropriate default."""
    if (value := os.environ.get(env_var)):
        return Path(value) / "angler_ai"
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "angler_ai"
    return Path.home() / default_subpath / "angler_ai"


@dataclass(frozen=True, slots=True)
class Paths:
    """Filesystem layout used by the tool."""

    data_dir: Path
    """Persistent data: features.duckdb, raw extracts."""

    cache_dir: Path
    """Downloaded model files."""

    state_dir: Path
    """Logs, transient state."""

    config_dir: Path
    """User configuration overrides."""

    @property
    def feature_store(self) -> Path:
        return self.data_dir / "features.duckdb"

    @property
    def raw_data(self) -> Path:
        return self.data_dir / "raw"

    @property
    def models(self) -> Path:
        return self.cache_dir / "models"

    @property
    def logs(self) -> Path:
        return self.state_dir / "logs"

    @property
    def manifest(self) -> Path:
        return self.data_dir / "data_manifest.json"


def default_paths() -> Paths:
    """Return the default Paths for the host OS."""
    return Paths(
        data_dir=_xdg_or_default("XDG_DATA_HOME", ".local/share"),
        cache_dir=_xdg_or_default("XDG_CACHE_HOME", ".cache"),
        state_dir=_xdg_or_default("XDG_STATE_HOME", ".local/state"),
        config_dir=_xdg_or_default("XDG_CONFIG_HOME", ".config"),
    )


def ensure_paths(paths: Paths) -> None:
    """Create the directory tree if any element is missing. Idempotent."""
    for d in (paths.data_dir, paths.cache_dir, paths.state_dir, paths.config_dir,
              paths.raw_data, paths.models, paths.logs):
        d.mkdir(parents=True, exist_ok=True)


# v0 launch states (resolved 2026-06-17; see docs/v0_design.md section 9).
LAUNCH_STATES: tuple[str, ...] = ("PA", "VA", "ID")
