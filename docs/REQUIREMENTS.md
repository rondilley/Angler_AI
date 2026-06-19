# Angler_AI - v0 Requirements

**Status:** Stable.
**Date:** 2026-06-17 (Created), 2026-06-18 (Split from `requirements_and_success_criteria.md` for VIBE compliance).
**Author:** Ron Dilley
**Companion to:** `docs/ARCHITECTURE.md`, `docs/SUCCESS_CRITERIA.md`, `docs/v0_design.md`

This document captures what the system MUST, SHOULD, and COULD do (the requirements). Success criteria and acceptance methodology live in `docs/SUCCESS_CRITERIA.md`. Together the two are the verification spine for v0.

---

## Document Conventions

- **Priority:** MUST (v0 blocker if missing), SHOULD (target for v0, acceptable to defer if necessary), COULD (nice-to-have, deferrable to v1+).
- **Verification:** how we measure this requirement is met. Either an automated test, a manual test script, a metric measurement, or a documented review.
- **Rationale:** which research artifact, design-doc section, or architectural pillar drives this requirement.
- Each requirement has a stable ID (`FR-3.4`, `NFR-2.1`, etc.) so we can reference it from code, tests, and PR descriptions.
- Each milestone has a release gate (in SUCCESS_CRITERIA.md). v0 ships only when all gates pass.

---

## 1. Functional Requirements

### FR-1: Hardware Probe

| ID | Requirement | Priority | Verification | Rationale |
|---|---|---|---|---|
| FR-1.1 | The system detects CPU architecture (x86_64 / arm64 / Apple Silicon), physical and logical core count, and microarchitecture feature flags (AVX/AVX2/AVX-512, ARM NEON) via archspec + psutil. | MUST | Unit test on each reference platform | Pass 6, design 5.1 |
| FR-1.2 | The system reports total and available RAM. | MUST | Unit test | Pass 6 |
| FR-1.3 | The system detects NVIDIA GPUs and reports VRAM total and available, compute capability, and driver/CUDA version. | MUST | Unit test on CUDA reference machine | Pass 6 |
| FR-1.4 | The system detects Apple Silicon GPU and treats unified memory correctly (no separate VRAM number; a unified `memory_budget_for_inference` derived from total RAM and reserved-for-OS heuristic). | MUST | Unit test on Apple Silicon | Pass 6 |
| FR-1.5 | The system detects AMD GPUs via rocm-smi where ROCm is installed. | SHOULD | Manual test on AMD machine if available; otherwise defer to v1 | Pass 6 |
| FR-1.6 | The system detects Intel discrete and integrated GPUs via OpenVINO Device API where present. | COULD | Defer to v1 | Pass 6 |
| FR-1.7 | The system detects Intel NPU via OpenVINO GenAI where present. | COULD | Defer to v1 | Pass 6 (intel-npu-acceleration-library archived; route via OpenVINO GenAI) |
| FR-1.8 | The system reports the chosen llama.cpp backend variant (CPU / CUDA-X / Metal / Vulkan / ROCm) and the wheel index used. | MUST | Captured in `angler-ai probe` output | Pass 6, design 5.1 |
| FR-1.9 | The system enforces Python 3.10, 3.11, or 3.12 at install time and fails fast with a clear message on 3.13+. | MUST | Install script + entry point check | Pass 6 (llama-cpp-python wheel matrix) |

### FR-2: Inference Layer

| ID | Requirement | Priority | Verification | Rationale |
|---|---|---|---|---|
| FR-2.1 | The system loads GGUF models via llama-cpp-python. | MUST | Integration test | Pass 6, design 5.2 |
| FR-2.2 | The system supports text generation, embeddings, vision-language input, and ASR (Whisper.cpp). | MUST (text+embedding); SHOULD (VLM); COULD (ASR) | Per-modality smoke test | Pass 6 |
| FR-2.3 | The system supports OpenAI-compatible function/tool calling. | MUST | Integration test against the 8 tools enumerated in design 5.8 | Pass A (RAG agent template), design 5.8 |
| FR-2.4 | The system supports JSON-schema-constrained output. | MUST | Integration test producing structured `ReachQueryResult` | Pass 6 |
| FR-2.5 | The system exposes an OpenAI-compatible HTTP API on loopback (127.0.0.1 by default). | MUST | curl smoke test against `/v1/chat/completions` | Pass 6 |
| FR-2.6 | The system supports speculative decoding via draft model where one is available in the registry. | COULD | Defer to v1 | Pass 6 |

### FR-3: Model Registry and Router

| ID | Requirement | Priority | Verification | Rationale |
|---|---|---|---|---|
| FR-3.1 | The system maintains a curated registry of GGUF models with structured capability metadata (id, family, type, params, quants, capabilities, context length, license, source). | MUST | Schema test; registry must validate at startup | Pass 6, design 5.3 |
| FR-3.2 | The system selects a model per problem set based on capability requirements and the user's hardware budget. | MUST | Per-problem-set unit test on simulated HardwareProfiles | Design 5.3 |
| FR-3.3 | The system auto-downloads required GGUFs from HuggingFace Hub on first use, with optional hf_transfer acceleration. | MUST | Integration test with mocked HF Hub | Pass 6 |
| FR-3.4 | The system verifies SHA256 (or the integrity hash the registry provides) of downloaded models. | MUST | Unit test on a tampered file | Pass 6 |
| FR-3.5 | The system surfaces the model's license to the user at first download and records acknowledgment. | MUST | Manual test | Design 5.3, CR-3.1 |
| FR-3.6 | When no model in the registry fits the user's hardware for a requested task, the system emits an explicit "no model fits your hardware for `<task>`" error rather than silently degrading. | MUST | Test on a synthetic low-end HardwareProfile | Design 5.3 |
| FR-3.7 | The system supports hot-swapping models when a single problem-set query requires multiple model families (e.g., reasoning -> embedding -> VLM). | SHOULD | Manual test; can be served by llama-swap at v1 | Pass 6 |
| FR-3.8 | The model selection decision (candidate set, winner, hardware budget snapshot) is logged. | MUST | Log inspection | NFR-6.2 |

### FR-4: Data Ingestion

| ID | Requirement | Priority | Verification | Rationale |
|---|---|---|---|---|
| FR-4.1 | The system ingests USGS NWIS streamflow via the Water Data OGC API at 15-minute cadence on demand. | MUST | Integration test against the live API | Pass 1, design 5.4 |
| FR-4.2 | The system ingests NHDPlus HR hydrography (geometry + VAAs Hydroseq, LevelPathI, UpHydroseq, DnHydroseq, ArbolateSu, FromMeas, ToMeas - note `ArbolateSu` not `ArbolateSum`). | MUST | Schema test; row count > 0 for v0 launch states | Pass 1, pass 2 (peer review correction) |
| FR-4.3 | The system ingests EPA Water Quality Portal data. | MUST | Integration test | Pass 1 |
| FR-4.4 | The system ingests EPA ATTAINS impaired-waters assessments via the REST/JSON API and ArcGIS REST geospatial service, including Assessments-by-Catchment join to NHDPlus. | MUST | Integration test | Pass 1 (corrected: API key "transitioning to required" per peer review) |
| FR-4.5 | The system ingests NorWeST modeled stream temperature for the West states (ID, MT, NV, OR, UT, WA, WY, CA, CO, NM) - both mean August summer and full-year products. | MUST | Schema test; reach coverage > 0 in NorWeST states | Pass 1, pass 2 (corrected: NOT summer-only) |
| FR-4.6 | The system ingests EcoSHEDS Northeast Stream Temperature Model + Northeast Catchment Delineation (NECD) + Brook Trout Occupancy model, restricted to NECD catchments with drainage < 200 km^2. | MUST | Schema test; reach coverage > 0 in NECD states (Maine to Virginia) | Pass B |
| FR-4.7 | The system ingests USGS BRT fluvial fish SDM v2.0 (415 species, NHDPlusV2.1 COMID, public DOI 10.5066/P1UV25FW). | MUST | Schema test; row count matches published count for the dataset | Pass A |
| FR-4.8 | The system ingests Pennsylvania PFBC TroutStocked data via ArcGIS REST endpoint discovery (NOT hardcoded year-stamped URLs like `TroutStocked2024`). | MUST | Integration test that survives a hypothetical year-rollover | Pass B (year-stamped service name flagged) |
| FR-4.9 | The system maintains crosswalks where data sources use different reach identifiers: NHDPlusV2.1 COMID <-> NHDPlus HR COMID; EcoSHEDS NECD reach <-> NHDPlus HR COMID. | MUST | Coverage test (>=95% join success across v0 launch states) | Pass B (NECD distinct from HR), pass A (USGS BRT on V2.1) |
| FR-4.10 | Every ingest run persists raw extracts alongside normalized features so re-ingestion is possible without re-querying the source. | MUST | Filesystem inspection | NFR-5.1 |
| FR-4.11 | The system surfaces per-source license and attribution metadata for every ingested dataset. | MUST | Manifest inspection | CR-3.3 |
| FR-4.12 | Ingestion modules implement idempotent runs (running twice produces the same state). | MUST | Idempotency test | NFR-2.1 |
| FR-4.13 | The system ingests NOAA NWS daily forecast (api.weather.gov) for days 0-7 and Open-Meteo (api.open-meteo.com) for days 8-16, at any lat/lon. | MUST | Integration test | M7/M8 forecast pipeline |

### FR-5: Prediction Layer

| ID | Requirement | Priority | Verification | Rationale |
|---|---|---|---|---|
| FR-5.1 | The system produces reach-level catch probability via SSN2 `ssn_glm(family=binomial)` on NHDPlus HR preprocessed via SSNbler. Output includes a point estimate and 95% prediction interval. | MUST (v1) | Output schema test + non-trivial probability variance test | Pass 2 (SSN2 substrate) |
| FR-5.2 | The system surfaces USGS BRT v2.0 species-presence priors per COMID per species (top 10 default). | MUST | Output matches published BRT for sampled COMIDs | Pass A |
| FR-5.3 | The system produces per-reach temperature and flow nowcasts via a physics-guided recurrent GNN (Jia 2021 / He 2024 lineage) where direct measurements are absent. | SHOULD (v1) | Held-out gauge benchmark vs naive interpolation | Pass 2 |
| FR-5.4 | The system detects anomalies in USGS NWIS streams via a HydroGEM-style TCN-Transformer (or equivalent pretrained checkpoint). Output is a per-gauge anomaly flag with reconstruction error. | SHOULD | Detection of injected synthetic anomalies on a held-out subset | Pass 2 (preprint, directional) |
| FR-5.5 | The system produces season-/event-conditioned harvest probability via a Bayesian hurdle-gamma model in Stan for short-season fisheries (trout openers, salmon runs, put-and-take stocking windows). | SHOULD (v1) | Sample-from-posterior smoke test; report 80% predictive interval | Pass 2 (arXiv 2503.17293 framework) |
| FR-5.6 | The system surfaces multi-species conditional predictions via JSDM (HMSC-style) where co-occurrence data is available at the queried location. | COULD | Defer to v1 | Pass A (McLaughlin 2024) |
| FR-5.7 | Winter precipitation / flow anomaly is included as a first-class feature in any trout-reach prediction. | MUST | Feature presence audit in trained model artifact | Pass A (Kanno 2015 GCB) |
| FR-5.8 | Predictions explicitly tag the reach geography source (NHDPlus HR COMID) and the temperature source ('NWIS_obs' / 'NWIS_interp' / 'NWIS_interp_air_adjusted' / 'NWIS_air_projected' / 'NorWeST' / 'EcoSHEDS_TEMP' / 'PG-GNN' / 'not_modeled'). | MUST | Schema test on output | Pass 2 (NorWeST caveat) |
| FR-5.9 | The system produces a per-reach per-day relative suitability index in [0, 1] over a 14-16 day forecast window by multiplying BRT calibrated prior x species thermal niche x flow factor x seasonal phenology x flow z-score anomaly. Output is renamed `suitability_index` not `catch_probability` until M4 catch-data validation (multi-AI review STAT-02 / PV-6). | MUST | Schema test on `DailyScore` output | M7/M8 forecast pipeline |

### FR-6: Calibration Layer

| ID | Requirement | Priority | Verification | Rationale |
|---|---|---|---|---|
| FR-6.1 | The system records the Charbonneau 2025 hyperstability constant (beta=0.23) on every `CalibratedProbability` for provenance. At v0 the correction is NOT applied to the BRT path (cpue_weight=0); applying p^(1/0.23) to a unit-interval presence probability is a statistical category error per multi-AI review STAT-01. v1 will replace BRT with SSN2 ssn_glm(binomial) trained on observed CPUE, at which point a non-zero weight is justified. | MUST | Unit test: BRT path raw 0.8 equals calibrated 0.8 at v0; `basis.sources` includes `hyperstability:not_applied(reason=BRT_presence_probability_v0)` | Pass A (hyperstability) + multi-AI review |
| FR-6.2 | The system produces a calibrated prediction interval on every probability output (point + lower + upper at a stated confidence). | MUST | Schema test; no user-facing surface can emit a probability without an interval | Pass A |
| FR-6.3 | The system records and surfaces the calibration basis: raw value, applied hyperstability constant, source-weight mix (CPUE-derived vs fisheries-independent), and uncertainty source. | MUST | Inspect `basis` field on every output | Pass A, design 5.7 |
| FR-6.4 | The system structurally prevents stripping of the prediction interval downstream. The interval is propagated through the 5-factor suitability chain (`DailyScore.suitability_lower / upper`). | MUST | Code audit; `tests/test_forecast_scoring.py::test_interval_propagated_through_factor_chain` regression | Design 5.7 + multi-AI review STAT-02 |
| FR-6.5 | The system supports conformal prediction or isotonic regression on SSN2 outputs once a verified recipe is available. | COULD | Defer to v1 | Pass A open question |
| FR-6.6 | The 5-factor multiplicative product is clamped to [0, 1]. `seasonal_factor` is capped at 1.0 with an audit tag (`capped_from_X.XX` in `seasonal_source`) so the chain cannot exceed 1.0. | MUST | `tests/test_forecast_scoring.py::test_seasonal_factor_above_1_is_capped` | Multi-AI review STAT-03 |

### FR-7: Reasoning Layer

| ID | Requirement | Priority | Verification | Rationale |
|---|---|---|---|---|
| FR-7.1 | The system accepts natural-language queries scoped by body of water, county, state, species, and date window. | MUST | UC1 acceptance script | Mission, design 5.8 |
| FR-7.2 | The system implements a 3-agent sequential pipeline: Profile (parse query) -> Planning (decide tools) -> Analyst (produce response). | MUST | Per-agent unit test plus end-to-end UC1 | Pass A (MARSHA template) |
| FR-7.3 | The system runs the agents on the locally-selected llama.cpp model by default. | MUST | Integration test | Architectural pillar 2026-06-17 |
| FR-7.4 | The system supports opt-in escalation to a SaaS frontier model (Anthropic / OpenAI / Gemini / Mistral / xAI per the keys present in repo root) when network and explicit user consent are both available. | SHOULD | Manual test | Architectural pillar 2026-06-17 |
| FR-7.5 | Responses are structured: ranked reaches, per-reach calibrated probability, contextual flow/temp/regs/stocking, alerts, and citations to the tool calls that produced the data. | MUST | UC1, UC2 acceptance scripts | Design 5.8 |
| FR-7.6 | Tool surface exposes at least the 8 functions enumerated in design 5.8 (`get_reaches_in_county`, `get_reach_features`, `get_catch_probability`, `get_flow_anomaly_status`, `get_regulations`, `get_stocking_history`, `search_papers`, `explain_prediction`). | MUST | Tool registry test | Design 5.8 |
| FR-7.7 | The system generates a per-(water, species) narrative summary for the forecast pipeline that explicitly names BEST DAY and WORST DAY with the exact dates and explains WHY by referencing factor differences (mean water temp vs species thermal niche, flow factor, seasonal factor, anomaly factor). | MUST | Per-narrative format audit | Per-day output milestone |
| FR-7.8 | LLM-generated narratives are constrained against fabrication: never invent species, reach counts, scores, dates, or citations; never refer to the output as a "catch probability"; never claim "thermal not modeled" alongside a numeric thermal_factor. | MUST | Tier 2 multi-AI review item; automated FactorBreakdown-vs-narrative diff (pending) | Multi-AI review PV-2/3/4 |

### FR-8: Ethics & Disclosure Layer

| ID | Requirement | Priority | Verification | Rationale |
|---|---|---|---|---|
| FR-8.1 | The system suppresses reach-level probability output for bull trout in all v0 launch states and falls back to a coarse-grained (HUC-10 or coarser) presentation with an explicit suppression notice. | MUST | Acceptance test: bull trout query in Idaho returns suppressed, not reach-level | CR-1, pass A (GBIF / USGS BRT precedent) |
| FR-8.2 | The system applies tiered suppression to ESA-Threatened and state-SoC native salmonids per CR-1.1 / CR-1.2 (see `sensitive_species_seed.csv`). At HUC-10 (CR-1.1): bull trout, Apache trout, Gila trout, Paiute cutthroat, Lahontan cutthroat. At HUC-12 (CR-1.2): greenback, Rio Grande, westslope, Yellowstone, Bonneville, Colorado River cutthroat. Common non-native trout at reach. | MUST | Acceptance test against at least one species per tier | CR-1, decision log 2026-06-17, multi-AI review FB-7/FB-8 |
| FR-8.3 | The system masks CRITFC-managed waters (and any other tribally-managed waters surfaced via partner data) and emits a tribal-sovereignty notice with a redirect to the tribe's data resources page. | MUST | Acceptance test: query intersecting Yakama Reservation returns sovereignty notice, not reach data | Pass B (CRITFC AGOL-only) |
| FR-8.4 | The system surfaces EPA ATTAINS impaired-waters status as a "water quality alert" tag on every affected reach in user-facing output, regardless of conversion impact. | MUST | Acceptance test against a known impaired reach | Pass 2 (deliberate anti-marketing trade-off) |
| FR-8.5 | The system records and surfaces every model's license at first download with explicit user acknowledgment. | MUST | Manual test | CR-3.1 |
| FR-8.6 | The system supports configurable spatial blurring (default HUC-10) for opted-in sensitive habitat categories. | SHOULD | Configuration test | CR-1 |

### FR-9: User Interface

| ID | Requirement | Priority | Verification | Rationale |
|---|---|---|---|---|
| FR-9.1 | The CLI exposes at minimum: `probe`, `pull-models`, `ingest`, `ask`, `map`, `forecast`, `species`, `serve`, `status`. | MUST | CLI smoke test | Design 5.10 |
| FR-9.2 | The local HTTP API serves OpenAI-compatible chat endpoints plus Angler_AI-specific routes (`/v1/reaches`, `/v1/probability`, `/v1/anomaly`, `/v1/regulations`, `/v1/map.geojson`). | MUST | curl tests against each endpoint | Design 5.10 |
| FR-9.3 | The `map` command and `/v1/map.geojson` endpoint emit a GeoJSON FeatureCollection with reach geometry plus `probability`, `lower`, `upper`, `basis`, `temperature_source`, `alerts`. | MUST | Schema test | Design 5.10 |
| FR-9.4 | The `forecast` command produces a per-(water, species) directory structure: `summary.png` (best-day-over-window aggregate) + `daily/<YYYY-MM-DD>.png` (one PNG per day in the 14-16 day forecast window). Each map is matplotlib + RdYlGn colormap on [0, 1] with the colorbar label "Relative suitability index (0-1, NOT a probability)". | MUST | Filesystem test + manual visual review | Per-day output milestone |
| FR-9.5 | The CLI supports `--explain` flag adding the Reasoning Layer's chain of tool calls and citations to the response. | SHOULD | Manual test | UC2 |
| FR-9.6 | The CLI supports JSON output (`--json`) for programmatic consumers. | SHOULD | Schema test | NFR-7 |
| FR-9.7 | A first-run onboarding flow walks the user through probe -> pull-models -> ingest. | SHOULD | Manual test | NFR-1, UX |

---

## 2. Non-Functional Requirements

### NFR-1: Performance

| ID | Requirement | Priority | Verification |
|---|---|---|---|
| NFR-1.1 | Hardware probe completes in under 5 seconds on any v0 reference platform. | MUST | Timed test |
| NFR-1.2 | First-run model acquisition for the default tier (qwen3.5-small-9b Q4_K_M, ~6 GB) completes in under 15 minutes on a 100 Mbps connection. | SHOULD | Timed test |
| NFR-1.3 | A single-reach probability query (UC1, narrow geo) returns in under 30 seconds on the default v0 hardware tier (12-20 GB VRAM, qwen3.5-35b-a3b IQ2_M). | MUST | Timed test |
| NFR-1.4 | USGS NWIS real-time flow lookup returns in under 5 seconds when the upstream API is available. | MUST | Timed test |
| NFR-1.5 | A full reasoning-layer response (UC1, UC2) returns in under 60 seconds on the default hardware tier. | MUST | Timed test |
| NFR-1.6 | Ingestion of a full state's PFBC TroutStocked dataset completes in under 5 minutes. | SHOULD | Timed test |
| NFR-1.7 | Cold-start of the inference layer (model load into VRAM) completes in under 60 seconds for any v0-tier model. | SHOULD | Timed test |

### NFR-2: Reliability

| ID | Requirement | Priority | Verification |
|---|---|---|---|
| NFR-2.1 | Data ingestion is idempotent: re-running a successful ingest produces the same DuckDB state. | MUST | Idempotency test |
| NFR-2.2 | Failed external API calls surface explicit, structured error messages with the failing endpoint, HTTP status, and a suggested remediation. No silent failure. | MUST | Negative test against simulated upstream failure |
| NFR-2.3 | The hardware probe succeeds on all v0 reference platforms; failure to detect non-critical components (NPU, secondary GPU) does not abort the probe. | MUST | Negative test simulating missing nvidia-ml-py / missing OpenVINO |
| NFR-2.4 | Pinned dependencies (nvidia-ml-py at specific minor, Python 3.10-3.12, llama-cpp-python wheel index) are enforced at install time. | MUST | Install pipeline test |
| NFR-2.5 | When upstream USGS NWIS is unreachable, the system serves the last cached value with an explicit staleness warning, not a stale value silently. | MUST | Network-failure test |
| NFR-2.6 | DuckDB feature store is recoverable from a clean re-ingest in case of corruption. | SHOULD | Manual test |

### NFR-3: Privacy & Security

| ID | Requirement | Priority | Verification |
|---|---|---|---|
| NFR-3.1 | All inference runs locally by default. No user query, location, or session data leaves the machine unless the user explicitly opts in to SaaS escalation (FR-7.4) per query. | MUST | Network capture test |
| NFR-3.2 | API keys for SaaS providers (Anthropic, OpenAI, Mistral, xAI, Gemini) are stored locally (`*.key.txt` per repo convention or env vars) and never transmitted to any party other than the named provider. | MUST | Code audit + network capture test |
| NFR-3.3 | User-entered geo coordinates are not persisted beyond the current session unless the user explicitly saves a named query. | MUST | Storage audit |
| NFR-3.4 | Downloaded GGUF files are SHA-verified per FR-3.4. | MUST | (Duplicated for emphasis) |
| NFR-3.5 | The system transmits no telemetry by default. An opt-in error reporting path may be added in v1 with explicit consent. | MUST | Network capture test |
| NFR-3.6 | The HTTP API binds to 127.0.0.1 by default; binding to 0.0.0.0 requires an explicit `--bind 0.0.0.0` flag and surfaces a warning. | MUST | Default binding test |

### NFR-4: Portability

| ID | Requirement | Priority | Verification |
|---|---|---|---|
| NFR-4.1 | The system runs on Windows 11 (x86_64). | MUST | Reference platform test |
| NFR-4.2 | The system runs on macOS 13+ (Apple Silicon arm64). | MUST | Reference platform test |
| NFR-4.3 | The system runs on Linux x86_64 with glibc 2.31+. | MUST | Reference platform test |
| NFR-4.4 | The system does NOT require root / admin privileges for normal operation. | MUST | Run-as-user test |
| NFR-4.5 | After data and models are pulled, the system operates fully offline. | MUST | Airplane-mode test |
| NFR-4.6 | The system runs on Linux arm64. | COULD | Defer to v1 |
| NFR-4.7 | The system runs on Windows arm64 (Copilot+ PCs). | COULD | Defer to v1+ (NPU detection is also v1+) |

### NFR-5: Maintainability

| ID | Requirement | Priority | Verification |
|---|---|---|---|
| NFR-5.1 | Every ingestion module declares source URL, refresh cadence, license, and discovery pattern in a structured form (Python dataclass or similar). | MUST | Schema test |
| NFR-5.2 | Calibration constants (BC hyperstability beta default 0.23) are configurable with documented sources. | MUST | Config test |
| NFR-5.3 | Each component (HP, IL, MR, DI, FS, PL, CL, RL, EL, UI) is independently testable with mocked dependencies. | MUST | Unit-test layout audit |
| NFR-5.4 | Type hints on all public APIs; mypy clean on the public surface. | SHOULD | CI check |
| NFR-5.5 | Ruff clean (default rule set) on the codebase. | SHOULD | CI check |

### NFR-6: Observability

| ID | Requirement | Priority | Verification |
|---|---|---|---|
| NFR-6.1 | Every external API call is logged with timestamp, endpoint, HTTP status, bytes transferred, and duration. | MUST | Log inspection |
| NFR-6.2 | Every model selection decision is logged with candidate set, winner, quant chosen, and hardware budget snapshot. | MUST | Log inspection |
| NFR-6.3 | Every calibration adjustment is logged with raw -> calibrated values, beta used, and source weights. | MUST | Log inspection |
| NFR-6.4 | A `status` CLI command and `/v1/status` API endpoint expose: currently loaded model, hardware backend, last ingest timestamps per source, disk usage by category. | MUST | Manual test |
| NFR-6.5 | Logs default to a local file under `${XDG_STATE_HOME:-~/.local/state}/angler_ai/logs/` and rotate at 50 MB. | SHOULD | Filesystem test |

### NFR-7: Extensibility

| ID | Requirement | Priority | Verification |
|---|---|---|---|
| NFR-7.1 | A new state stocking source is addable by writing a single ingestion module conforming to the source schema (NFR-5.1) without modifying any existing code. | MUST | Add a second state (e.g., NJ) end-to-end as a verification |
| NFR-7.2 | A new GGUF model is addable via a registry YAML/JSON entry without code changes. | MUST | Registry-only addition test |
| NFR-7.3 | A new problem set is addable by declaring its capability requirements; the Router handles selection automatically. | MUST | New-problem-set test |
| NFR-7.4 | New prediction components plug into the Prediction Layer via a stable interface returning `CalibratedProbability`. | SHOULD | Interface test |

---

## 3. Data Requirements

### DR-1: Substrate

| ID | Requirement | Priority |
|---|---|---|
| DR-1.1 | NHDPlus HR COMID is the canonical reach identifier for all internal feature joins. | MUST |
| DR-1.2 | NHDPlusV2.1 COMID <-> NHDPlus HR COMID crosswalk exists locally, with >=95% join success across v0 launch states. | MUST |
| DR-1.3 | EcoSHEDS NECD reach <-> NHDPlus HR COMID crosswalk exists locally for the EcoSHEDS coverage area, with documented join-success rate per state. | MUST |
| DR-1.4 | Where USGS BRT (NHDPlusV2.1) lacks a clean HR mapping for a given COMID, the system surfaces the lack-of-join rather than silently emitting a prediction. | MUST |

### DR-2: Provenance

| ID | Requirement | Priority |
|---|---|---|
| DR-2.1 | Every persisted feature row carries a `source` column identifying the origin (`USGS_NWIS`, `EPA_ATTAINS`, `NorWeST`, `EcoSHEDS_TEMP`, `EcoSHEDS_BTO`, `USGS_BRT_V2.0`, `PA_PFBC_TROUT_STOCKED`, etc.). | MUST |
| DR-2.2 | Every persisted feature row carries an `ingested_at` timestamp. | MUST |
| DR-2.3 | Every prediction emitted by the Prediction Layer carries a `model_id` and `model_version`. | MUST |
| DR-2.4 | A `data_manifest.json` at the feature-store root enumerates every dataset's source, license, refresh cadence, and last successful refresh. | MUST |
| DR-2.5 | No fabricated values. Missing data returns the honest `not_modeled:<reason>` tag and `factor=1.0` (in the suitability chain). v1 stubs raise `NotImplementedError` rather than returning placeholders. | MUST |

### DR-3: Freshness

| ID | Requirement | Priority |
|---|---|---|
| DR-3.1 | USGS NWIS real-time flow is no older than 30 minutes at query time, or the value is flagged stale. | MUST |
| DR-3.2 | NHDPlus HR is refreshed annually or upon new USGS vintage release. | SHOULD |
| DR-3.3 | NorWeST, EcoSHEDS Temp/BTO, USGS BRT are refreshed per release (release events trigger a refresh job). | SHOULD |
| DR-3.4 | EPA ATTAINS is refreshed twice per state assessment cycle (event-driven on state submission + EPA final action). | SHOULD |
| DR-3.5 | EPA Water Quality Portal is refreshed weekly. | SHOULD |
| DR-3.6 | State stocking data is refreshed per state cadence; PA PFBC TroutStocked is checked weekly via the discovery pattern. | MUST |

### DR-4: Storage

| ID | Requirement | Priority |
|---|---|---|
| DR-4.1 | All feature data is persisted in a single DuckDB file under `${XDG_DATA_HOME:-~/.local/share}/angler_ai/features.duckdb`. | MUST |
| DR-4.2 | Raw extracts (shapefiles, GeoJSON, CSVs) are preserved under `${XDG_DATA_HOME:-~/.local/share}/angler_ai/raw/<source>/`. | MUST |
| DR-4.3 | Model files are stored under `${XDG_CACHE_HOME:-~/.cache}/angler_ai/models/`. | MUST |
| DR-4.4 | A `disk_budget` config setting allows the user to set a maximum size for raw + features + models combined; the system warns at 80% and refuses non-essential downloads at 100%. | SHOULD |

---

## 4. Compliance & Ethics Requirements

### CR-1: Sensitive Species

| ID | Requirement | Priority |
|---|---|---|
| CR-1.1 | Bull trout, Apache trout, Gila trout, Paiute cutthroat, and Lahontan cutthroat (all ESA-Threatened) reach-level probabilities are suppressed by default in every v0 launch state. Coarse-grained presentation at HUC-10. | MUST |
| CR-1.2 | ESA-listed native cutthroat subspecies (greenback in CO, Rio Grande in NM) coarse-grained at HUC-10. State species-of-concern native cutthroat (westslope, Yellowstone, Bonneville, Colorado River) coarse-grained at HUC-12. Common non-native trout (rainbow, brown, brook outside native range) at reach. | MUST |
| CR-1.3 | Any other ESA-listed freshwater species encountered during data ingest triggers a policy gate; the species is added to `sensitive_species` with `suppress_level=reach` until a human review reclassifies. | MUST |
| CR-1.4 | The suppression policy and the list of suppressed species are surfaced in `status` output and in any response that would have included a suppressed species. | MUST |
| CR-1.5 | User override of suppression for a single query requires explicit acknowledgment recorded in the local log; no global "disable suppression" config exists. | MUST |

### CR-2: Tribal Sovereignty

| ID | Requirement | Priority |
|---|---|---|
| CR-2.1 | The system does NOT scrape, hot-link, or redistribute CRITFC ArcGIS Online layers. | MUST |
| CR-2.2 | Queries intersecting tribally-managed waters surface a sovereignty notice and a redirect to the tribe's published data resources, NOT reach-level data. | MUST |
| CR-2.3 | The tribal-waters mask is sourced from public USGS / state datasets only (no scraped tribal data). | MUST |
| CR-2.4 | The system supports future opt-in partnership integrations where a tribe provides explicit data terms; no v0 partnership is assumed. | SHOULD |

### CR-3: Licensing

| ID | Requirement | Priority |
|---|---|---|
| CR-3.1 | Every model's license is surfaced at first download with user acknowledgment recorded. | MUST |
| CR-3.2 | Codebase is Apache-2.0. | MUST |
| CR-3.3 | Every bundled dataset's license is tracked in the data manifest. | MUST |
| CR-3.4 | State data published without an explicit license (e.g., PA PFBC service descriptor) is treated conservatively: attribute, do not redistribute the raw layer. | MUST |

### CR-4: Honesty

| ID | Requirement | Priority |
|---|---|---|
| CR-4.1 | Probability presentation always includes a calibrated interval. | MUST |
| CR-4.2 | Calibration is mandatory and cannot be disabled. | MUST |
| CR-4.3 | EPA ATTAINS impaired-water status is surfaced as an alert regardless of conversion or trip-intent impact. | MUST |
| CR-4.4 | Sources are cited in user-facing responses. The Analyst agent's response includes per-claim tool-call citations. | MUST |
| CR-4.5 | When a refuted research claim is referenced anywhere in code or docs, the refutation is referenced alongside it. (e.g., do NOT cite Unsloth Dynamic 2.0 MMLU numbers verbatim - measure ourselves.) | MUST |
| CR-4.6 | The 5-factor multiplicative suitability composite is presented as a "relative suitability index in [0, 1]", NOT a "catch probability", until M4 catch-data validation produces a defensible probability calibration. (Multi-AI review STAT-02 / PV-6.) | MUST |

---

## 5. Out of Scope (v0) - Explicit

These are intentionally excluded from v0 to keep the scope shippable:

- Lakes, reservoirs, ponds
- Saltwater
- Mobile apps (iOS, Android)
- On-device fine-tuning (LoRA), distillation, model merging - deferred to v2
- Heterogeneous bipartite GNN-SDM (architecture verified, efficacy refuted in pass 2)
- Time-series foundation models for hydrology (lose to domain LSTMs as of 2026)
- AMD ROCm, Intel SYCL, Intel NPU, Apple Neural Engine, Qualcomm Hexagon, Windows Copilot+ NPU detection
- Python 3.13 and 3.14 (llama-cpp-python wheels not available at audit time)
- CUDA 12.6, 12.7, 12.8 backends (wheel gap; fall back to Vulkan)
- Pacific Northwest anadromous queries (StreamNet/CAX integration deferred to v1)
- Eastern mainstem trout (drainage > 200 km^2; outside EcoSHEDS BTO)
- Multi-state regulation Q&A (single-state PA exemplar in v0; expansion in v1)
- Anglers Atlas / MyCatch ingestion (peer-reviewed but transferability gap; defer)
- Tribal partnership integrations (no v0 partnership assumed; sovereignty mask only)
- Telemetry / analytics / crash reporting
- Multi-user / cloud-hosted variants
- Web UI / GUI / desktop wrapper (v1+ Tauri, v1+ FastAPI Vite, etc.)

---

## 6. Resolved Decisions (2026-06-17)

The previously-open product questions are resolved. See `docs/v0_design.md` section 9 for the canonical decision table; the rationale is in the chat record dated 2026-06-17. Summary of how each decision touches this document:

| Decision | Affected requirements |
|---|---|
| ATTAINS surfacing default-on with "Water Quality Alert" tag | FR-8.4, CR-4.3 (no change) |
| Tiered cutthroat: HUC-10 ESA, HUC-12 SOC, reach for non-native | FR-8.2 updated, CR-1.2 updated |
| MyCatch deferred to v1 | No FR change; v1 candidate documented in design 6 |
| V2.1 <-> HR crosswalk: pull USGS / PSS, derive as fallback | DR-1.2 unchanged (95% target) |
| Default 12-20 GB tier model: Qwen3.5-35B-A3B IQ2_M | Router default in design 6; M1 gate runs acceptance suite |
| Tribal partnership pathway: ship sovereignty mask + notice, defer outreach to v1 | CR-2.4 unchanged (SHOULD with v1 timeline) |
| State data license: attribute, link, no redistribution | CR-3.4 unchanged; reinforced by DR-4.2 (user runs ingest locally) |
| v0 launch geography: PA + VA + Idaho (California -> v1) | Per-milestone acceptance and held-out test sets updated to reference ID |

## 7. Resolved Decisions (2026-06-18, multi-AI review)

The 2026-06-17 multi-AI review (5 reviewers, 22 high/critical findings, 20 survived 3-voter adversarial verification) drove the following updates:

| Decision | Affected requirements |
|---|---|
| Rename output from `catch_probability` to `suitability_index` until M4 catch-data validation | FR-5.9, CR-4.6 added; FR-9.4 updated |
| Propagate CalibratedProbability interval through 5-factor suitability chain | FR-6.4 updated to reference DailyScore.suitability_lower / upper |
| Clamp the multiplicative chain to [0, 1]; cap seasonal_factor at 1.0 with audit tag | FR-6.6 added |
| Drop Charbonneau hyperstability from BRT path (cpue_weight=0) | FR-6.1 updated to reflect not_applied at v0; SSN2 path retains the constant |
| Add Apache trout, Gila trout, Paiute cutthroat at HUC-10 (CR-1.1); reclassify Lahontan cutthroat from state-soc/huc12 to ESA-Threatened/huc10 | CR-1.1 updated; FR-8.2 updated |
| LLM narrative fabrication discipline (no "catch probability", no thermal-modeled / not-modeled contradictions, base_p vs thermal_factor distinction) | FR-7.7, FR-7.8 added |

---

## 8. Glossary

(References `docs/v0_design.md` section 10 glossary. Additional terms specific to this document:)

| Term | Meaning |
|---|---|
| MUST / SHOULD / COULD | RFC-2119-style priority levels |
| Acceptance suite | Set of tests run as the v0 release gate (see `docs/SUCCESS_CRITERIA.md`) |
| Brier score | Mean squared error of probabilistic prediction vs observed binary outcome; lower is better; 0.25 = uninformed |
| Reliability diagram | Per-bin calibration plot of predicted vs observed frequencies |
| HUC-10 / HUC-12 | Hydrologic Unit Code at 10-digit (subwatershed) / 12-digit (subwatershed-segment) granularity, USGS standard |
| Discovery pattern | Ingestion approach where the source URL is resolved at runtime (e.g., probing for `TroutStocked<year>` services) rather than hardcoded |
| Suitability index | The 5-factor multiplicative composite in [0, 1] produced by the forecast pipeline. NOT a calibrated catch probability at v0. |
