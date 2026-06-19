"""Tests for the ingest dispatcher (offline). Drives the IngestionModule
contract with stand-in modules so we don't hit the live federal/state APIs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from angler_ai.features import open_store
from angler_ai.ingest.base import IngestionModule, SourceMetadata
from angler_ai.ingest.dispatch import _update_manifest, IngestSummary, run


class _FakeIngest:
    """Conforms to the IngestionModule protocol."""

    metadata = SourceMetadata(
        source_id="FAKE_SOURCE",
        display_name="Fake source",
        source_url="https://example.invalid",
        license="public-domain",
        refresh_cadence="manual",
    )

    rows_to_write = 7
    raise_exc: Exception | None = None

    def ingest(self, store, **_) -> int:  # noqa: ANN001
        if self.raise_exc:
            raise self.raise_exc
        return self.rows_to_write

    def last_ingested_at(self, store) -> str | None:  # noqa: ANN001
        return None


def test_update_manifest_writes_dataset_entries(tmp_path: Path) -> None:
    mpath = tmp_path / "data_manifest.json"
    summary = IngestSummary(
        source_id="EPA_ATTAINS",
        display_name="EPA ATTAINS",
        license="public-domain",
        refresh_cadence="twice per cycle",
        state="PA",
        rows_written=1234,
        last_refresh=datetime.now(timezone.utc).isoformat(),
        ok=True,
    )
    _update_manifest(mpath, [summary])
    data = json.loads(mpath.read_text(encoding="utf-8"))
    assert "datasets" in data
    assert "EPA_ATTAINS::PA" in data["datasets"]
    assert data["datasets"]["EPA_ATTAINS::PA"]["rows_written"] == 1234


def test_update_manifest_is_idempotent(tmp_path: Path) -> None:
    """Re-running updates the same key, not appending duplicates."""
    mpath = tmp_path / "data_manifest.json"
    s1 = IngestSummary(
        source_id="EPA_ATTAINS",
        display_name="EPA ATTAINS",
        license="public-domain",
        refresh_cadence="twice per cycle",
        state="PA",
        rows_written=10,
        last_refresh="2026-06-17T00:00:00+00:00",
        ok=True,
    )
    _update_manifest(mpath, [s1])
    s2 = IngestSummary(
        source_id="EPA_ATTAINS",
        display_name="EPA ATTAINS",
        license="public-domain",
        refresh_cadence="twice per cycle",
        state="PA",
        rows_written=20,
        last_refresh="2026-06-17T01:00:00+00:00",
        ok=True,
    )
    _update_manifest(mpath, [s2])
    data = json.loads(mpath.read_text(encoding="utf-8"))
    assert len(data["datasets"]) == 1
    assert data["datasets"]["EPA_ATTAINS::PA"]["rows_written"] == 20


def test_run_unknown_source_raises(tmp_path: Path) -> None:
    with open_store(tmp_path / "test.duckdb") as store:
        store.initialize_schema()
        with pytest.raises(ValueError):
            run(store, source="not-a-source", state="PA")


def test_run_continues_on_module_failure(tmp_path: Path, monkeypatch) -> None:
    """One source erroring out does not abort the dispatch; the summary
    records the error and other sources still run."""
    # Replace the default order with our fake module to keep this offline.
    from angler_ai.ingest import dispatch as disp

    class _RaisingFake(_FakeIngest):
        raise_exc = RuntimeError("simulated upstream 500")

    monkeypatch.setattr(
        disp, "_DEFAULT_ORDER",
        (("fake", _RaisingFake), ("fake_ok", _FakeIngest)),
    )
    db = tmp_path / "test.duckdb"
    with open_store(db) as store:
        store.initialize_schema()
        summaries = run(
            store,
            source="all",
            state="PA",
            manifest_path=tmp_path / "manifest.json",
        )
    assert len(summaries) == 2
    assert summaries[0].ok is False
    assert "500" in (summaries[0].error or "")
    assert summaries[1].ok is True
    assert summaries[1].rows_written == 7
