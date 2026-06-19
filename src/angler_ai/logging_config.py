"""Logging configuration. Logs default to a rotating file under the state directory.

Satisfies NFR-6.5 (rotation at 50 MB) and provides structured log records for
NFR-6.1 (external API calls), NFR-6.2 (model selection), NFR-6.3 (calibration).
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-5s [%(name)s] %(message)s"
MAX_BYTES = 50 * 1024 * 1024
BACKUP_COUNT = 5


def configure(log_dir: Path, level: int = logging.INFO) -> None:
    """Initialize the root logger with a console handler and a rotating file handler.

    Idempotent: calling twice does not duplicate handlers.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)

    if any(getattr(h, "_angler_ai", False) for h in root.handlers):
        return

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(LOG_FORMAT))
    console._angler_ai = True  # type: ignore[attr-defined]
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        log_dir / "angler-ai.log",
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    file_handler._angler_ai = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)
