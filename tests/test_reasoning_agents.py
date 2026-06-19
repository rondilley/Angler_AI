"""Tests for the 3-agent reasoning pipeline.

Use a stand-in LLM handle (implements __call__) so tests don't load the
real llama.cpp model. The Analyst's tool dispatch hits real DuckDB tables
seeded by the test fixtures.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from angler_ai.features import open_store
from angler_ai.reasoning.agents import (
    AnalystAgent,
    LLMRunner,
    PlanningAgent,
    ProfileAgent,
    QueryPlan,
)
from angler_ai.reasoning.tools import ToolError


class _StubLLM:
    """Test stand-in LLM. Records prompts; returns canned replies."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def __call__(self, *, system: str, user: str, max_tokens: int, temperature: float) -> str:
        self.calls.append({"system": system, "user": user})
        if self.replies:
            return self.replies.pop(0)
        return "{}"


def test_profile_agent_parses_json_reply() -> None:
    """Profile uses _strip_json so fenced replies still parse."""
    stub = _StubLLM([
        '```json\n{"species": "brook trout", "state": "PA", "intent": "where_to_fish"}\n```'
    ])
    profile = ProfileAgent(LLMRunner(handle=stub)).parse("any query")
    assert profile.species_common == "brook trout"
    assert profile.state == "PA"
    assert profile.intent == "where_to_fish"


def test_profile_agent_handles_unparseable_reply() -> None:
    """Profile must NOT raise on garbage replies; falls back to defaults."""
    stub = _StubLLM(["this is not json at all"])
    profile = ProfileAgent(LLMRunner(handle=stub)).parse("any query")
    assert profile.intent == "where_to_fish"
    assert profile.species_common is None


def test_planning_agent_maps_intents_to_tool_sequences() -> None:
    pl = PlanningAgent()
    for intent in ("where_to_fish", "explain_reach", "regulation_qa", "anomaly_explanation"):
        plan = pl.plan(QueryPlan(intent=intent))
        assert plan.tools, f"empty tool plan for intent {intent}"


def test_planning_agent_defaults_huc8_and_state() -> None:
    plan = PlanningAgent().plan(
        QueryPlan(intent="where_to_fish"),
        default_huc8="02050206",
        default_state="PA",
    )
    assert plan.huc8 == "02050206"
    assert plan.state == "PA"


# ---------- Analyst with real feature store -------------------------------


def _seed_minimum_db(tmp_path: Path) -> Path:
    db = tmp_path / "t.duckdb"
    with open_store(db) as store:
        store.initialize_schema()
        now = datetime.now(timezone.utc).isoformat()
        conn = store.connect()
        # One species in the BRT registry
        conn.execute(
            "INSERT INTO brt_species "
            "(itis_tsn, scientific_name, common_name, presences, absences, "
            "prevalence, taxonomic_order, family, ingested_at) "
            "VALUES (1, 'Salvelinus fontinalis', 'brook trout', 100, 200, 0.3, "
            "'Salmoniformes', 'Salmonidae', ?)",
            [now],
        )
        # One reach in HUC8 02050206
        conn.execute(
            "INSERT INTO reaches (comid, reachcode, gnis_name, state_fips, "
            "huc8, huc10, huc12, stream_order, drainage_area_km2, source, ingested_at) "
            "VALUES (1, '02050206000001', 'Test Run', '42', "
            "'02050206', '0205020600', '020502060001', 2, 5.0, 'NHDPlus_HR', ?)",
            [now],
        )
        # V2 xwalk + BRT prior
        conn.execute(
            "INSERT INTO xwalk_v2_to_hr (comid_v2, comid_hr, confidence, method, ingested_at) "
            "VALUES (100, 1, 1.0, 'reachcode_exact', ?)",
            [now],
        )
        conn.execute(
            "INSERT INTO brt_priors (comid, species, probability, ingested_at) "
            "VALUES (100, 'Salvelinus fontinalis', 0.65, ?)",
            [now],
        )
    return db


def test_analyst_runs_real_tools_against_seeded_db(tmp_path: Path) -> None:
    db = _seed_minimum_db(tmp_path)
    stub = _StubLLM([
        "The model says brook trout probability is 0.65 at COMID 1."
    ])
    with open_store(db) as store:
        analyst = AnalystAgent(LLMRunner(handle=stub), store)
        plan = QueryPlan(
            species_scientific="Salvelinus fontinalis",
            species_common="brook trout",
            state="PA",
            huc8="02050206",
            intent="where_to_fish",
            tools=(
                "get_reaches_in_huc8", "get_top_species",
                "get_regulations", "get_stocking_history", "get_temperature",
            ),
        )
        resp = analyst.respond(plan, original_query="brook trout in PA")
    # The Analyst MUST surface a narrative and citation set
    assert resp.narrative
    assert "USGS Fluvial Fish BRT v2.0" in " ".join(resp.citations)
    # Reaches found means the get_reaches_in_huc8 tool ran successfully.
    assert resp.reaches_considered == 1
    # The Analyst should have included a get_top_species call for the focal reach.
    names = [c.name for c in resp.tool_call_log]
    assert "get_top_species" in names
    assert "get_reaches_in_huc8" in names
    # The empty regulation/stocking tables should yield honest errors, not fakes.
    error_names = {c.name for c in resp.tool_call_log if c.error}
    assert "get_regulations" in error_names
    assert "get_stocking_history" in error_names


def test_analyst_does_not_silently_fail_when_db_empty(tmp_path: Path) -> None:
    """Empty DB -> tool errors recorded; narrative still generated honestly."""
    db = tmp_path / "empty.duckdb"
    with open_store(db) as store:
        store.initialize_schema()
        stub = _StubLLM(["No reaches available."])
        analyst = AnalystAgent(LLMRunner(handle=stub), store)
        plan = QueryPlan(
            species_common="brook trout",
            state="PA",
            huc8="99999999",
            intent="where_to_fish",
            tools=("get_reaches_in_huc8", "get_top_species"),
        )
        resp = analyst.respond(plan, original_query="brook trout")
    # At least one tool error recorded.
    errors = [c for c in resp.tool_call_log if c.error]
    assert errors, "expected ToolError surface, got silent success"
