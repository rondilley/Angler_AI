"""DuckDB bulk-insert helper. Uses pyarrow registration which is ~800x faster
than `executemany` for moderate row counts (20k rows: ~0.15s vs ~124s).
"""

from __future__ import annotations

from typing import Any, Sequence

import pyarrow as pa


def bulk_insert(
    conn: Any,
    table: str,
    rows: Sequence[Sequence[Any]],
    columns: Sequence[str],
    *,
    extra_literals: dict[str, Any] | None = None,
    geometry_columns: Sequence[str] = (),
) -> int:
    """Insert `rows` into `table` via pyarrow.Table registration.

    Args:
        conn: DuckDB connection.
        table: target table name.
        rows: per-row tuples; position must match `columns`.
        columns: tuple column names corresponding to `rows[i]` positions.
        extra_literals: extra columns whose values are constants (e.g.
            `{'source': 'EPA_ATTAINS', 'ingested_at': '2026-...'}`). Inserted
            as parameter literals, not part of `rows`.
        geometry_columns: subset of `columns` whose values are WKT strings to
            be wrapped in TRY(ST_GeomFromText(...)) so malformed WKTs become
            NULL geometries instead of crashing the batch.

    Returns:
        Number of rows inserted.
    """
    if not rows:
        return 0
    arr = pa.table({col: [r[i] for r in rows] for i, col in enumerate(columns)})
    conn.register("_bulk_tmp", arr)
    try:
        select_parts: list[str] = []
        for col in columns:
            if col in geometry_columns:
                select_parts.append(f"TRY(ST_GeomFromText({col})) AS {col}")
            else:
                select_parts.append(col)
        target_cols = list(columns)
        params: list[Any] = []
        if extra_literals:
            for col, val in extra_literals.items():
                target_cols.append(col)
                select_parts.append("?")
                params.append(val)
        sql = (
            f"INSERT INTO {table} ({', '.join(target_cols)}) "
            f"SELECT {', '.join(select_parts)} FROM _bulk_tmp"
        )
        conn.execute(sql, params)
    finally:
        conn.unregister("_bulk_tmp")
    return len(rows)
