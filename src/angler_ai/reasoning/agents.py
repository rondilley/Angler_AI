"""3-agent reasoning pipeline adapted from MARSHA (Xie et al. 2025).

Profile -> Planning -> Analyst, all running on the locally-selected
llama.cpp model. The agents call REAL tools that query the DuckDB feature
store; no fabricated answers ever reach the user.

License of the prompt scaffolding: this file is original Angler_AI code;
the MARSHA paper architecture is the inspiration, not the prompts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from angler_ai.features.store import FeatureStore
from angler_ai.reasoning import tools as T
from angler_ai.reasoning.tools import ToolError

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """Output of the Profile agent + extended by the Planning agent."""

    species_scientific: str | None = None
    species_common: str | None = None
    state: str | None = None
    huc8: str | None = None
    county: str | None = None
    water_body_name: str | None = None
    date_window: tuple[str, str] | None = None
    intent: str = "where_to_fish"
    """One of: where_to_fish | explain_reach | regulation_qa | anomaly_explanation"""
    tools: tuple[str, ...] = ()


@dataclass(slots=True)
class ToolCallRecord:
    """One Analyst tool call. --explain surfaces these to the user."""

    name: str
    args: dict
    result_summary: str
    error: str | None = None


@dataclass(slots=True)
class AnalystResponse:
    """Final structured response. Always carries citations + intervals."""

    narrative: str
    citations: list[str] = field(default_factory=list)
    reaches_considered: int = 0
    tool_call_log: list[ToolCallRecord] = field(default_factory=list)
    plan: QueryPlan | None = None


# ---------------------------------------------------------------- LLM interface


@dataclass
class LLMRunner:
    """Adapter around llama.cpp's create_chat_completion.

    The injectable handle is whatever runtime is loaded (typically
    `InferenceRuntime.ensure_loaded(...).handle`, a `llama_cpp.Llama`).
    Callers that don't have a model loaded can pass a callable that
    accepts (system, user, max_tokens, temperature) and returns a str.
    """

    handle: Any

    def chat(
        self, *, system: str, user: str,
        max_tokens: int = 512, temperature: float = 0.2,
    ) -> str:
        # llama_cpp.Llama has create_chat_completion; other handles (test
        # stand-ins) implement __call__ with the same kwargs.
        if hasattr(self.handle, "create_chat_completion"):
            out = self.handle.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
            )
            return out["choices"][0]["message"]["content"]
        return self.handle(system=system, user=user,
                           max_tokens=max_tokens, temperature=temperature)


def _strip_json(text: str) -> str:
    """Best-effort JSON extraction from an LLM completion."""
    text = text.strip()
    if text.startswith("```"):
        # Strip fenced code block
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[: -3]
    s, e = text.find("{"), text.rfind("}")
    if s >= 0 and e > s:
        text = text[s : e + 1]
    return text.strip()


# ---------------------------------------------------------------- Profile agent


class ProfileAgent:
    """Parse natural-language query into a structured QueryPlan."""

    SYSTEM = (
        "You parse natural-language fishing questions for the Angler_AI tool "
        "into strict JSON. Extract: species (Latin scientific name if "
        "obvious, otherwise common name), state (two-letter code), county, "
        "water_body_name (e.g. 'West Branch Susquehanna', 'Pine Creek'), and "
        "intent. Valid intents: where_to_fish, explain_reach, "
        "regulation_qa, anomaly_explanation. Return ONLY a JSON object. "
        "Use null for missing fields. Do not add explanations or markdown."
    )

    def __init__(self, llm: LLMRunner) -> None:
        self.llm = llm

    def parse(self, query: str) -> QueryPlan:
        user = f"Query: {query!r}\n\nJSON:"
        raw = self.llm.chat(system=self.SYSTEM, user=user, max_tokens=256, temperature=0.0)
        log.debug("Profile raw: %s", raw)
        parsed: dict = {}
        try:
            parsed = json.loads(_strip_json(raw))
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning("Profile JSON parse failed: %s; raw=%r", exc, raw[:200])
        # Light validation
        species = parsed.get("species")
        common = None
        sci = None
        if isinstance(species, str):
            if " " in species and species[0].isupper():
                sci = species
            else:
                common = species
        return QueryPlan(
            species_scientific=sci,
            species_common=common,
            state=(parsed.get("state") or "").upper()[:2] or None,
            county=parsed.get("county"),
            water_body_name=parsed.get("water_body_name"),
            intent=(parsed.get("intent") or "where_to_fish"),
        )


# ---------------------------------------------------------------- Planning agent


class PlanningAgent:
    """Translate QueryPlan into a concrete tool-call ordering.

    For v0 we use deterministic mapping (intent -> tool sequence) which is
    more reliable than LLM-decided tool plans on small models. The Planning
    agent's responsibility is dressing the QueryPlan with concrete tool
    names; the actual scheduling happens in the Analyst agent.
    """

    INTENT_TOOLS: dict[str, tuple[str, ...]] = {
        "where_to_fish": (
            "get_reaches_in_huc8", "get_top_species", "get_regulations",
            "get_stocking_history", "get_temperature",
        ),
        "explain_reach": (
            "get_reach", "get_top_species", "get_temperature",
            "get_attains_status", "get_stocking_history",
        ),
        "regulation_qa": (
            "get_regulations", "get_stocking_history",
        ),
        "anomaly_explanation": (
            "hydrogem_flow_anomaly",
        ),
    }

    def plan(self, plan: QueryPlan, *, default_huc8: str = "02050206",
             default_state: str = "PA") -> QueryPlan:
        tools = self.INTENT_TOOLS.get(plan.intent, self.INTENT_TOOLS["where_to_fish"])
        return QueryPlan(
            species_scientific=plan.species_scientific,
            species_common=plan.species_common,
            state=plan.state or default_state,
            huc8=plan.huc8 or default_huc8,
            county=plan.county,
            water_body_name=plan.water_body_name,
            date_window=plan.date_window,
            intent=plan.intent,
            tools=tools,
        )


# ---------------------------------------------------------------- Analyst agent


class AnalystAgent:
    """Execute the tool plan against the real feature store, ground the LLM."""

    SYSTEM = (
        "You are the Analyst for Angler_AI. You are given REAL tool outputs "
        "from a US fisheries data pipeline (USGS BRT v2.0 priors, EPA ATTAINS, "
        "state stocking records, NHDPlus HR reaches). Write a short, "
        "ground-truthed response for the angler. Cite numeric values verbatim "
        "from the tool outputs. NEVER invent probabilities, dates, regulations, "
        "or species. If a value is unknown, say so. Include the hyperstability "
        "calibration note when probabilities appear."
    )

    def __init__(self, llm: LLMRunner, store: FeatureStore) -> None:
        self.llm = llm
        self.store = store

    def respond(self, plan: QueryPlan, original_query: str) -> AnalystResponse:
        call_log: list[ToolCallRecord] = []

        # Resolve species scientific name if only common is set.
        sci = plan.species_scientific
        common = plan.species_common
        if sci is None and common:
            row = self.store.connect().execute(
                "SELECT scientific_name, common_name FROM brt_species "
                "WHERE LOWER(common_name) = LOWER(?)",
                [common],
            ).fetchone()
            if row:
                sci, common = row[0], row[1]

        # Resolve a focal COMID for per-reach tools. For where_to_fish and
        # explain_reach, we pick the highest-probability reach for the
        # target species (if known) or the top-stream-order reach.
        focal_comid: int | None = None
        reaches_cache: list[T.Reach] | None = None
        if plan.huc8 and plan.intent in ("where_to_fish", "explain_reach"):
            try:
                reaches_cache = T.get_reaches_in_huc8(self.store, huc8=plan.huc8, limit=25)
                focal_comid = self._pick_focal_comid(reaches_cache, sci)
            except ToolError as exc:
                call_log.append(ToolCallRecord(
                    name="get_reaches_in_huc8",
                    args={"huc8": plan.huc8, "limit": 25},
                    result_summary="",
                    error=str(exc),
                ))

        # Run tools per plan.tools
        tool_results: dict[str, Any] = {}
        if reaches_cache is not None:
            tool_results["get_reaches_in_huc8"] = reaches_cache
            call_log.append(ToolCallRecord(
                name="get_reaches_in_huc8",
                args={"huc8": plan.huc8, "limit": 25},
                result_summary=f"[{len(reaches_cache)} reaches] focal=COMID {focal_comid}",
            ))

        for tool_name in plan.tools:
            if tool_name == "get_reaches_in_huc8":
                continue  # already done above
            try:
                result = self._dispatch_with_comid(tool_name, plan, sci, focal_comid)
                summary = self._summarize(tool_name, result)
                call_log.append(ToolCallRecord(
                    name=tool_name,
                    args=self._args_with_comid(tool_name, plan, focal_comid),
                    result_summary=summary,
                ))
                tool_results[tool_name] = result
            except ToolError as exc:
                call_log.append(ToolCallRecord(
                    name=tool_name,
                    args=self._args_with_comid(tool_name, plan, focal_comid),
                    result_summary="",
                    error=str(exc),
                ))

        # Build a compact ground-truthed context block for the LLM.
        context = self._build_context_block(tool_results, plan, sci, common)
        user = (
            f"User query: {original_query!r}\n\n"
            f"Tool outputs (ground truth - cite numbers exactly):\n{context}\n\n"
            "Write the response."
        )
        narrative = self.llm.chat(system=self.SYSTEM, user=user,
                                  max_tokens=512, temperature=0.3)

        citations = self._cite(tool_results, sci)
        reaches_considered = 0
        if "get_reaches_in_huc8" in tool_results:
            reaches_considered = len(tool_results["get_reaches_in_huc8"])

        return AnalystResponse(
            narrative=narrative.strip(),
            citations=citations,
            reaches_considered=reaches_considered,
            tool_call_log=call_log,
            plan=plan,
        )

    # ----- dispatch helpers -------------------------------------------------

    def _pick_focal_comid(
        self, reaches: list[T.Reach], species_scientific: str | None,
    ) -> int | None:
        """Choose a focal COMID for per-reach tools.

        Prefer the highest-probability reach for the target species, falling
        back to the highest-stream-order reach (= largest waterway).
        """
        if not reaches:
            return None
        if species_scientific:
            best: tuple[float, int] | None = None
            for r in reaches:
                try:
                    cp = T.get_catch_probability(self.store, r.comid, species_scientific)
                except ToolError:
                    continue
                if best is None or cp.point > best[0]:
                    best = (cp.point, r.comid)
            if best is not None:
                return best[1]
        # Fallback: largest stream order, then largest drainage.
        scored = sorted(
            reaches,
            key=lambda r: (r.stream_order or 0, r.drainage_area_km2 or 0),
            reverse=True,
        )
        return scored[0].comid

    def _dispatch_with_comid(
        self, tool_name: str, plan: QueryPlan,
        species_scientific: str | None, focal_comid: int | None,
    ) -> Any:
        match tool_name:
            case "get_reach":
                if focal_comid is None:
                    raise ToolError("get_reach requires a focal COMID.")
                return T.get_reach(self.store, comid=focal_comid)
            case "get_top_species":
                if focal_comid is None:
                    raise ToolError("get_top_species requires a focal COMID.")
                return T.get_top_species(self.store, comid=focal_comid, top_k=5)
            case "get_temperature":
                if focal_comid is None:
                    raise ToolError("get_temperature requires a focal COMID.")
                return T.get_temperature(self.store, comid=focal_comid)
            case "get_attains_status":
                if focal_comid is None:
                    raise ToolError("get_attains_status requires a focal COMID.")
                return T.get_attains_status(self.store, comid=focal_comid)
            case "get_stocking_history":
                return T.get_stocking_history(
                    self.store, water_body_name=plan.water_body_name,
                    state=plan.state or "PA", lookback_days=365,
                )
            case "get_regulations":
                return T.get_regulations(
                    self.store, state=plan.state or "PA",
                    species=plan.species_common, water_body_name=plan.water_body_name,
                )
            case "hydrogem_flow_anomaly":
                if plan.water_body_name:
                    return T.hydrogem_flow_anomaly(self.store, gauge_id=plan.water_body_name)
                # Default UC6 path: run on the published synthetic test set.
                status, truth = T.hydrogem_synthetic_test(sample_index=0)
                return {"status": status, "ground_truth": truth}
            case _:
                raise ToolError(f"Unknown tool: {tool_name}")

    def _args_with_comid(
        self, tool_name: str, plan: QueryPlan, focal_comid: int | None,
    ) -> dict:
        if tool_name in ("get_reach", "get_top_species", "get_temperature", "get_attains_status"):
            return {"comid": focal_comid}
        if tool_name == "get_stocking_history":
            return {"state": plan.state, "water_body_name": plan.water_body_name, "lookback_days": 365}
        if tool_name == "get_regulations":
            return {"state": plan.state, "species": plan.species_common, "water_body_name": plan.water_body_name}
        if tool_name == "hydrogem_flow_anomaly":
            return {"gauge_id": plan.water_body_name or "synthetic_test_index=0"}
        return {}

    @staticmethod
    def _summarize(tool_name: str, result: Any) -> str:
        if isinstance(result, list):
            return f"[{len(result)} items] first={result[0] if result else 'none'}"
        return str(result)[:200]

    def _build_context_block(
        self, results: dict[str, Any], plan: QueryPlan,
        sci: str | None, common: str | None,
    ) -> str:
        lines: list[str] = []
        if sci or common:
            lines.append(f"Species: {sci} ({common})")
        if plan.huc8:
            lines.append(f"HUC8: {plan.huc8}")
        if plan.state:
            lines.append(f"State: {plan.state}")
        if "get_reaches_in_huc8" in results:
            reaches = results["get_reaches_in_huc8"]
            lines.append(f"Reaches loaded: {len(reaches)} (showing first 5)")
            for r in reaches[:5]:
                lines.append(f"  COMID {r.comid}  {r.gnis_name or 'unnamed'}  "
                             f"order={r.stream_order}  drainage_km2={r.drainage_area_km2}")
        if "get_top_species" in results:
            tops = results["get_top_species"]
            lines.append("Top species priors (calibrated, hyperstability beta=0.23 applied):")
            for sp, cn, cp in tops:
                lines.append(f"  {sp} ({cn}): p={cp.point:.3f} [{cp.lower:.3f}, {cp.upper:.3f}]")
        if "get_temperature" in results:
            t = results["get_temperature"]
            lines.append(f"Temperature: {t['temperature_c']} C (source: {t['source']})")
        if "get_attains_status" in results:
            att = results["get_attains_status"]
            lines.append(f"EPA ATTAINS status records: {len(att)}")
            for a in att[:3]:
                lines.append(f"  cycle={a['cycle_year']}, status={a['status']!r}, "
                             f"parameter={a['parameter']!r}")
        if "get_stocking_history" in results:
            ev = results["get_stocking_history"]
            lines.append(f"Stocking events: {len(ev)} (showing 3 most recent)")
            for e in ev[:3]:
                lines.append(f"  {e.event_date}  {e.water_body_name}  {e.species}")
        if "get_regulations" in results:
            regs = results["get_regulations"]
            lines.append(f"Regulations rows: {len(regs)} (showing first 3)")
            for r in regs[:3]:
                lines.append(f"  {r.state}  water={r.water_body_id!r}  species={r.species!r}  "
                             f"special={r.special_regulation!r}  url={r.source_url}")
        if "hydrogem_flow_anomaly" in results:
            res = results["hydrogem_flow_anomaly"]
            # Either AnomalyStatus alone or dict with status + ground_truth.
            if isinstance(res, dict):
                a = res["status"]
                gt = res["ground_truth"]
                lines.append(
                    f"HydroGEM anomaly (synthetic-test sample): site={a.gauge_id} "
                    f"is_anomalous={a.is_anomalous} steps={a.n_anomalous_steps}/{a.total_steps} "
                    f"max_p={a.max_probability:.3f} threshold={a.threshold:.3f} "
                    f"model_version={a.model_version}"
                )
                lines.append(
                    f"Ground truth: {len(gt['segments'])} injected anomaly segment(s)."
                )
                for seg in gt['segments'][:3]:
                    lines.append(f"  segment kind={seg.get('kind')} start={seg.get('start')} end={seg.get('end')}")
            else:
                a = res
                lines.append(
                    f"HydroGEM anomaly: gauge={a.gauge_id} is_anomalous={a.is_anomalous} "
                    f"steps={a.n_anomalous_steps}/{a.total_steps} max_p={a.max_probability:.3f}"
                )
        return "\n".join(lines)

    def _cite(self, results: dict[str, Any], sci: str | None) -> list[str]:
        cites: list[str] = []
        if "get_top_species" in results or "get_temperature" in results:
            cites.append("USGS Fluvial Fish BRT v2.0 (DOI: 10.5066/P1UV25FW)")
            cites.append("Charbonneau et al. 2025 TAFS hyperstability (beta=0.23)")
        if "get_attains_status" in results and results["get_attains_status"]:
            cites.append("EPA ATTAINS (https://gispub.epa.gov/arcgis/rest/services/OW/ATTAINS_Assessment)")
        if "get_stocking_history" in results and results["get_stocking_history"]:
            cites.append("Pennsylvania Fish & Boat Commission TroutStocked")
        if "get_regulations" in results and results["get_regulations"]:
            cites.append("Pennsylvania Fish & Boat Commission regulations")
        if "hydrogem_flow_anomaly" in results:
            cites.append("HydroGEM (Ejokhan/HydroGEM on HuggingFace, arXiv 2512.14106)")
        return cites


