"""DuckDB feature store wrapper."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import duckdb

log = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class FeatureStore:
    """Thin wrapper around a DuckDB connection.

    Methods are intentionally narrow; complex queries live in feature-specific
    modules (e.g. `prediction/brt_priors.py`) so the wrapper stays a transport
    layer.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: duckdb.DuckDBPyConnection | None = None

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Open the connection. Idempotent."""
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            log.debug("Opening DuckDB at %s", self.db_path)
            self._conn = duckdb.connect(str(self.db_path))
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def initialize_schema(self) -> None:
        """Run the schema SQL. Idempotent (uses CREATE TABLE IF NOT EXISTS)."""
        conn = self.connect()
        sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        log.info("Initializing feature-store schema (%s)", _SCHEMA_PATH)
        conn.execute(sql)

    @contextmanager
    def transaction(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """Context-managed transaction."""
        conn = self.connect()
        conn.execute("BEGIN")
        try:
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


@contextmanager
def open_store(db_path: Path) -> Iterator[FeatureStore]:
    """Context-managed FeatureStore."""
    store = FeatureStore(db_path)
    try:
        store.connect()
        yield store
    finally:
        store.close()
