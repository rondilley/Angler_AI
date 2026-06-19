# Angler_AI - v0 Success Criteria

**Status:** Stable.
**Date:** 2026-06-17 (Created), 2026-06-18 (Split from `requirements_and_success_criteria.md` for VIBE compliance).
**Author:** Ron Dilley
**Companion to:** `docs/REQUIREMENTS.md`, `docs/ARCHITECTURE.md`

This document captures how we know we are done: per-milestone release gates, per-use-case acceptance scripts, per-quality-attribute measurements, the overall v0 release gate, and the acceptance methodology that produces those measurements. The companion `docs/REQUIREMENTS.md` lists what the system MUST/SHOULD/COULD do; this document lists how each is verified.

---

## 1. Per-Milestone Release Gates

### M1 Gate: Hardware probe + inference smoke test

- [x] `angler-ai probe` succeeds on Windows 11 (CUDA), macOS 14 (Apple Silicon), and Linux 22.04 (CPU-only).
- [x] Probe output correctly identifies CPU arch, RAM, and GPU/VRAM (NVIDIA + Apple Silicon).
- [x] `angler-ai pull-models --profile small` downloads `qwen3.5-small-4b Q4_K_M` and verifies SHA.
- [x] `angler-ai serve` starts a llama-cpp-python server on 127.0.0.1.
- [x] curl POST against `/v1/chat/completions` returns a valid completion within 60 seconds.
- [x] Probe + pull + serve + completion logged with the required fields (NFR-6.1, NFR-6.2).

### M2 Gate: Data ingestion (federal core + Pennsylvania)

- [x] `angler-ai ingest --source all --state PA` populates DuckDB with non-zero row counts in: `reaches`, `reach_flow` (sample), `reach_wq`, `attains_status`, `brt_priors`, `stocking_events`, `regulations`.
- [x] PA PFBC ingestion uses the discovery pattern (no hardcoded year).
- [x] NHDPlus HR VAA field name is `ArbolateSu`, not `ArbolateSum`. (Schema test.)
- [x] Idempotency: re-running the ingest produces identical row counts (modulo intentional refreshes).
- [x] Data manifest enumerates every dataset with source, license, refresh cadence, last refresh timestamp.

### M3 Gate: USGS BRT priors

- [x] For 100 randomly sampled PA NHDPlus HR COMIDs, `angler-ai species --comid <N> --top 10` returns rankings that join successfully via the V2.1 <-> HR crosswalk for >=95% of reaches.
- [x] Reaches lacking V2.1 mapping surface the lack-of-join, not a silent prediction.
- [x] Species output includes `model_id="USGS_BRT_V2.0"`, `model_version`, and the source DOI.

### M4 Gate: Calibrated species map (revised from SSN2 catch probability)

- [x] For any HUC8, `angler-ai map --species X --huc8 Y --out F.geojson` exports a GeoJSON with per-COMID `probability`, `lower`, `upper`, `basis`.
- [x] Intervals are visibly wider on COMIDs with sparser feature coverage (data-sparsity test).
- [x] No user-facing output can strip the interval (code audit confirms; `tests/test_forecast_scoring.py::test_interval_propagated_through_factor_chain`).
- [ ] Brier score on a held-out PA reach subset: <= 0.22 for the default tier. **DEFERRED** to v1 - requires observed catch data; v0 ships with BRT presence priors only and explicitly labels the output `suitability_index` not `catch_probability` (multi-AI review STAT-02 / PV-6).
- [x] Charbonneau hyperstability constant recorded on every `CalibratedProbability`. At v0 the correction is NOT applied to the BRT path; basis surfaces `hyperstability:not_applied(reason=BRT_presence_probability_v0)`.

### M5 Gate: Honest temperature substrate

- [x] For any reach with USGS NWIS water-temperature observation, temperature is sourced from `NWIS_obs`.
- [x] For reaches in same HUC10 and +-1 stream order of an NWIS gauge, temperature is sourced from `NWIS_interp_air_adjusted` (IDW spatial anchor + per-day Mohseni-Stefan air-temp delta).
- [x] For reaches outside IDW coverage, temperature is sourced from pure `NWIS_air_projected` (Mohseni-Stefan with stream-order-aware defaults).
- [x] Temperature source is visible on every probability output.
- [ ] EcoSHEDS BTO surface honors the 200 km^2 ceiling; reaches above the ceiling surface "not modeled" rather than extrapolating. **DEFERRED** to v1 - EcoSHEDS gridded loads deferred; resolver returns `not_modeled` honestly when no real source covers the reach.
- [ ] NorWeST loaded for ID/MT/WY. **DEFERRED** to v1.

### M6 Gate: HydroGEM anomaly + Reasoning Layer alpha

- [x] HydroGEM checkpoint pulled from HuggingFace, runs on published synthetic-anomaly test pickle, flags anomalies correctly.
- [x] 3-agent MARSHA-style pipeline runs end-to-end on the locally-selected llama.cpp model.
- [x] UC1 acceptance script passes: query, response, citations, intervals, alerts, within 60 seconds on default-tier hardware.
- [x] UC2 (`--explain`) shows the Analyst agent's tool calls and their results.
- [ ] UC4 (regulation Q&A) succeeds against PA regulations. **PARTIAL** - regs ingested for PA; full UC4 acceptance pending.
- [ ] UC6 (flow anomaly explanation) succeeds against an injected synthetic anomaly. **PARTIAL** - HydroGEM works against the published pickle; production 576-hour NWIS series ingest per gauge is on-demand.

### M7 Gate: Forecast pipeline + species thermal niches

- [x] Per-(water, species) PNG map renders from BRT prior x species thermal niche x NWS precip forecast.
- [x] Peer-reviewed thermal niches for 12 species (Elliott 1994, Wehrly 2007, Bear 2007, Selong 2001, Brinkman 2013, Hubert 1985, Myrick 2000, Lamothe 2003, Wismer 1987, Casselman 1996, Stewart 1983).
- [x] Map colorbar reads "Relative suitability index (0-1, NOT a probability)" not "catch probability".
- [x] Caption surfaces BRT DOI, hyperstability source (not_applied notation), niche citation, forecast sources, water-temp coverage.

### M8 Gate: Multi-factor closeout

- [x] 5-factor scoring chain: BRT calibrated prior x species thermal niche x flow factor x seasonal phenology x flow z-score anomaly.
- [x] Seasonal phenology multipliers for 12 species (peer-reviewed monthly activity windows).
- [x] Per-day water-temperature modeling via NWIS_obs > NWIS_interp_air_adjusted > NWIS_interp > NWIS_air_projected.
- [x] Per-HUC12 multi-point NWS sampling (up to 6 HUC12s per HUC8).
- [x] Open-Meteo 16-day extended forecast for days 8-16.
- [x] LLM narrative summary per (water, species) with best/worst day naming + WHY explanation.

### Multi-AI Review + Tier 1 Gate

- [x] Five independent AI reviewers (statistical-methodology, ml-and-ai-models, fisheries-biology, software-implementation, prediction-validation) audited the v0 code.
- [x] 22 high/critical findings sent to 3-voter adversarial verification; 20 survived.
- [x] Tier 1 fixes applied:
  - [x] Rename `DailyScore.score` -> `suitability_index` throughout code, captions, narratives, JSON
  - [x] Propagate CalibratedProbability interval through factor chain (`DailyScore.suitability_lower / upper`)
  - [x] Clamp multiplicative chain to [0, 1]; cap seasonal_factor at 1.0 with audit tag
  - [x] Drop Charbonneau hyperstability from BRT path (cpue_weight=0)
  - [x] Add Apache trout, Gila trout, Paiute cutthroat to `sensitive_species_seed.csv` at HUC-10/CR-1.1; reclassify Lahontan cutthroat from `state-soc/huc12` to ESA-threatened/HUC-10
- [ ] Tier 2 fixes (LLM faithfulness check, prompt-injection sanitization, narrative caveat scaffold). **PENDING**.
- [ ] Tier 3 fixes (BRT range_type tagging for native vs introduced, IDW elevation lapse, snowmelt-regime flow factor for western HUC8s, asymmetric thermal niche). **PENDING**.
- [ ] Tier 4 fixes (5xx backoff retry, CLI input validators, z-score baseline cap, interval Wilson rename, dispatcher specific-exception handling, kwargs whitelist, max-carry-days, mean-over-window, percentile colormap, hoot-owl caveat, User-Agent from config, source-tag StrEnum). **PENDING**.

### Per-day output Gate

- [x] Output structure `out/forecasts/<water>/<species>/summary.png` + `daily/<YYYY-MM-DD>.png` for every day in the 14-16 day window.
- [x] Per-day aggregate stats (mean_thermal_factor, mean_water_temp_c, flow_factor, seasonal_factor, mean_anomaly_factor) computed and surfaced in narrative.
- [x] Best/worst day named explicitly in narrative with WHY explanation.
- [x] Observation-anchored Mohseni-Stefan (`NWIS_interp_air_adjusted`) eliminates the "15 flat days then a jump" IDW artifact.

### Cross-cut Gate: Ethics

- [x] Bull trout query in Idaho returns reach-level suppression notice + HUC-10 fallback.
- [ ] Westslope cutthroat query returns HUC-12 fallback. **PENDING** acceptance test.
- [ ] Query intersecting Yakama Reservation returns tribal-sovereignty notice + redirect. **PENDING** acceptance test.
- [ ] A query touching a known ATTAINS-impaired reach surfaces the alert in the response. **PENDING** acceptance test.
- [x] Code audit: `CalibratedProbability` is the only return type from PL, with no field-stripping path to user-facing surfaces. `DailyScore` propagates the interval (FR-6.4 honored).

---

## 2. Per-Use-Case Acceptance

### UC1: "Where is good trout fishing within X miles of my home this weekend"

Input: `angler-ai ask "good trout fishing near Lycoming County PA this weekend" --species rainbow,brown,brook --days 3`

Acceptance:
- Response returns within 60 seconds on default-tier hardware
- Response includes >=5 ranked NHDPlus HR reaches with: GNIS name, county, calibrated probability (point + lower + upper), basis tag, recent flow + temp + temp source, regulatory snapshot, stocking history (last 60 days), alerts (water quality, anomaly), citations
- Brook trout suppressed at reach level if in CR-1 species list for ID/UT/MT/CO; in PA brook trout is not native-restricted but ATTAINS alerts still apply
- At least 3 distinct sources cited

### UC2: "Why does the model think this reach is good right now"

Input: `angler-ai ask "why is the West Branch Susquehanna good for smallmouth bass right now" --comid <N> --explain`

Acceptance:
- Response includes the Analyst agent's tool call chain
- Each tool call shows: tool name, arguments, returned value
- Final narrative ties feature values to the probability output
- Calibration step is visible (raw -> calibrated)

### UC3: "Compare these three streams for next Saturday"

Input: `angler-ai compare --comids 1,2,3 --species walleye --date 2026-06-27`

Acceptance: (v0 nice-to-have)
- Side-by-side table of calibrated probabilities + intervals
- Highlighting where intervals do not overlap
- Notes on differing temperature sources or data sparsity

### UC4: "What is the regulatory status on this stream"

Input: `angler-ai regs --water "Pine Creek" --state PA --species rainbow`

Acceptance:
- Returns season dates, gear restrictions, bag limits, license requirements for the matching water body
- Cites PA PFBC source
- Flags any special regulations (e.g., trophy trout)

### UC5: "Show me a probability map for smallmouth bass in this county"

Input: `angler-ai map --county "Lycoming, PA" --species smallmouth-bass --date 2026-07-04 --out map.geojson`

Acceptance: (v0 nice-to-have)
- GeoJSON FeatureCollection emitted
- Per-feature properties: `comid`, `probability`, `lower`, `upper`, `basis`, `temperature_source`, `alerts`
- Renders correctly in QGIS / Felt / kepler.gl

### UC6: "Detect anomalous flow conditions on my home water"

Input: `angler-ai anomaly --gauge <USGS_gauge_id>`

Acceptance:
- HydroGEM reconstruction error reported with a normal-range comparison
- LLM explanation in plain English of why the anomaly matters for fishing
- Citations to USGS NWIS + recent precipitation data

### UC7: "Extract structured data from this angler-report text"

Input: piped text from a forum post

Acceptance: (v0 nice-to-have)
- JSON output with: species, location (free-text + parsed county/water if confident), gear, water condition, catch_count, date, confidence
- Uses JSON-schema-constrained output (FR-2.4)

### UC8: "Best fishing day in the next two weeks for X species in Y watershed"

Input: per-(water, species) forecast run via `.venv/run_western_forecasts.py` or `angler-ai forecast`

Acceptance:
- Output structure `out/forecasts/<water>/<species>/summary.png` + `daily/<YYYY-MM-DD>.png` for every day in 14-16 day window
- LLM narrative names BEST DAY and WORST DAY with exact dates and median suitability values
- Narrative explains WHY each is best/worst by comparing mean water temp against the species thermal niche AND citing factor differences (flow, seasonal, anomaly)
- Narrative refers to output as "suitability index", NEVER as "catch probability"
- Per-day suitability summary table included for cross-checking

---

## 3. Per-Quality-Attribute Measurements

| Attribute | Metric | v0 target |
|---|---|---|
| Probability calibration (v1) | Brier score on held-out PA reach subset | <= 0.22 (DEFERRED to v1; requires observed catch data) |
| Probability calibration (v1) | Reliability diagram bin-wise error | <= 0.10 absolute in 80% of bins |
| Probability calibration (v1) | Interval coverage at 95% nominal | between 0.92 and 0.98 |
| Hardware probe success | Per-platform pass rate | 100% on Win11/macOS14/Linux22 |
| Model selection determinism | Same hardware + same task -> same model | 100% |
| External API availability handling | Cached fallback rate on simulated upstream failure | 100% |
| First-run time-to-completion | Probe + pull-models small + smoke chat | <= 20 minutes on 100 Mbps |
| Crosswalk join success | V2.1 -> HR | >= 95% per v0 launch state |
| Crosswalk join success | NECD -> HR | >= 90% per EcoSHEDS coverage state |
| HydroGEM anomaly detection | F1 on synthetic anomalies | >= 0.70 (preprint is 0.79; we accept directional) |
| Bull trout suppression | False reach-level disclosures | 0 in acceptance suite |
| Tribal mask | False reach-level disclosures on tribal waters | 0 in acceptance suite |
| Multi-AI review high/critical findings survived adversarial verification | Tier 1 applied; Tier 2/3/4 tracked | All Tier 1 closed; Tier 2/3/4 in `tasks/todo.md` |
| Source tagging coverage | % of persisted rows with non-null `source` column | 100% |
| No fabricated values | Tests asserting `not_modeled` returned instead of substituted value | `tests/test_temperature_resolver.py::test_resolver_never_returns_proxy_source` passing |

---

## 4. v0 Release Gate (overall "ready to ship")

Angler_AI v0 ships only when all of the following are true:

1. All MUST functional requirements implemented and tested.
2. All MUST non-functional requirements verified.
3. All MUST data requirements satisfied (including DR-2.5 no-fabricated-values).
4. All MUST compliance & ethics requirements verified in the acceptance suite (including CR-4.6 suitability_index vs catch_probability naming).
5. M1, M2, M3, M4, M5, M6, M7, M8, Multi-AI Tier 1, Per-day, and Ethics cross-cut gates pass.
6. UC1, UC2, UC4, UC6, UC8 acceptance scripts pass on all three reference platforms.
7. Probability calibration metric (Brier <= 0.22) DEFERRED to v1 with documented rationale (no v0 catch-data training set; output explicitly labeled `suitability_index`).
8. Sensitive-species and tribal-sovereignty acceptance scripts: 0 false reach-level disclosures.
9. ATTAINS-surfacing acceptance script: alert tag visible on every known-impaired reach in test set.
10. Logs contain the required structured records per NFR-6.
11. License surfacing recorded for every shipped model.
12. README, docs/ARCHITECTURE.md, docs/REQUIREMENTS.md, docs/SUCCESS_CRITERIA.md, docs/VISION.md, CLAUDE.md, VIBE_HISTORY.md all current.
13. Multi-AI review report `.ai-reviews/multi_ai_review_2026-06-17.md` archived; Tier 1 applied; Tier 2/3/4 either applied or explicitly tracked in `tasks/todo.md`.

---

## 5. Acceptance Methodology

### Test pyramid

- Unit tests for: hardware probe, ingestion modules, registry parsing, router logic, calibration math, ethics policy gates, forecast scoring invariants (suitability in [0,1], interval propagation, seasonal cap, hyperstability not_applied tag).
- Integration tests for: end-to-end ingest of a sample state, end-to-end SSN2 run on a sample basin (v1), end-to-end MARSHA pipeline on a fixed query, llama-cpp-python smoke against the chat endpoint, model download + SHA verify.
- Acceptance scripts per use case (UC1, UC2, UC4, UC6, UC8 MUST; UC3, UC5, UC7 SHOULD).
- Manual reference-platform tests on Win11 CUDA, macOS Apple Silicon, Linux CPU.

### Held-out test sets

- **Reach-level catch probability (v1):** 100 randomly sampled PA NHDPlus HR COMIDs withheld from any model training/calibration. Brier score reported on this set when catch data lands.
- **Synthetic anomaly injection:** the HydroGEM authors' 18 anomaly types applied to held-out PA USGS NWIS gauges. Currently verified against the published synthetic-anomaly pickle.
- **Sensitive-species acceptance:** a fixed list of test queries that should suppress: bull trout in ID (HUC-10); Apache trout in AZ (HUC-10); Gila trout in NM (HUC-10); Paiute cutthroat in CA (HUC-10); Lahontan cutthroat in NV (HUC-10); westslope cutthroat in ID (HUC-12); Yellowstone cutthroat in ID (HUC-12); greenback cutthroat in CO (HUC-10); Rio Grande cutthroat in NM (HUC-10).
- **Tribal mask acceptance:** fixed list of test queries intersecting Warm Springs, Yakama, Umatilla, Nez Perce, Colville, Spokane, Coeur d'Alene reservations.
- **ATTAINS surfacing acceptance:** fixed list of known impaired reaches in PA / VA / ID.
- **Multi-AI review:** 5 reviewer perspectives surfaced 22 high/critical findings; 20 survived 3-voter adversarial verification. Report at `.ai-reviews/multi_ai_review_2026-06-17.md`.

### Measurement cadence

- Acceptance suite runs locally on every commit to `main`.
- Per-quality metrics measured at each milestone release gate.
- Reference-platform runs per release candidate.
- Multi-AI review repeated before any v1 release.

### Decision rule for "ready to ship"

- Every MUST passes its verification.
- Every metric meets or exceeds the v0 target in section 3.
- All acceptance scripts pass on all three reference platforms.
- Open questions in design 9 either resolved or explicitly punted to v1 with documented rationale.
- All Tier 1 multi-AI review findings closed; Tier 2/3/4 either closed or explicitly deferred in `tasks/todo.md`.

---

## 6. Open release-gate items (as of 2026-06-18)

These items are explicitly NOT met at the current build. Each has a status and a path forward.

| Item | Status | Path |
|---|---|---|
| Brier score <= 0.22 | DEFERRED v1 | Requires observed catch data; v0 output renamed to `suitability_index` to be honest about the gap |
| EcoSHEDS BTO + NorWeST gridded loads | DEFERRED v1 | Resolver returns `not_modeled` honestly when no real source covers the reach; honest substrate ships at v0 |
| UC4 PA regulations acceptance | PARTIAL | PA regs ingested; full UC4 script pending |
| UC6 production NWIS series for HydroGEM | PARTIAL | Works against published synthetic-anomaly pickle; 576-hour per-gauge series ingest on-demand |
| Westslope cutthroat HUC-12 acceptance test | PENDING | Add to acceptance suite |
| Tribal-sovereignty acceptance test | PENDING | Add to acceptance suite |
| ATTAINS-impaired-reach surfacing acceptance test | PENDING | Add to acceptance suite |
| Multi-AI review Tier 2 (faithfulness check, prompt-injection sanitize, narrative caveat scaffold) | PENDING | `tasks/todo.md` |
| Multi-AI review Tier 3 (BRT range_type tags, IDW elevation lapse, snowmelt flow factor, asymmetric thermal niche) | PENDING | `tasks/todo.md` |
| Multi-AI review Tier 4 (5xx backoff, CLI validators, z-score baseline, Wilson rename, etc.) | PENDING | `tasks/todo.md` |
