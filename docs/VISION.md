# Angler_AI - v0 Vision

**Status:** Stable.
**Date:** 2026-06-17 (Created from the initial project prompt), 2026-06-18 (Refined for VIBE compliance + multi-AI review terminology updates).
**Author:** Ron Dilley
**Companion to:** `docs/ARCHITECTURE.md`, `docs/REQUIREMENTS.md`, `docs/SUCCESS_CRITERIA.md`

This document captures what Angler_AI is FOR - the problem the tool exists to solve and the angler whose day it tries to improve. Requirements (`docs/REQUIREMENTS.md`) translate this vision into MUST/SHOULD/COULD obligations; the Architecture (`docs/ARCHITECTURE.md`) is how the code organizes itself to deliver them; Success Criteria (`docs/SUCCESS_CRITERIA.md`) is how we know we are done.

---

## 1. The problem in one sentence

US recreational anglers planning a fishing trip on a river or stream have to stitch together NHDPlus hydrography, USGS real-time flow, EPA water-quality alerts, state stocking schedules, state regulations, weather forecasts, and species-specific biology from 7+ separate websites - and there is no single tool that fuses them into a reach-level, time-windowed, calibrated-honest answer to "where and when should I go this week."

## 2. The angler

The serious freshwater angler fishes rivers and streams more than 20 days per year, mostly trout and warmwater (bass, panfish, walleye). They want reach-level intel, not POI pins. They currently use TroutRoutes + USGS WaterWatch + state agency websites + fishing forums in parallel and they pay for tools that genuinely save trip-planning time.

Secondary persona: the fisheries-curious technologist who fishes recreationally, wants a tool they can extend or audit, and cares about explainability and calibration.

Tertiary persona: the fisheries researcher / conservation org who uses the same data substrates we do, wants to reproduce reach-level joins, and is alert to sensitive-species disclosure norms.

## 3. The product vision

Angler_AI is a **local-first, hardware-adaptive, calibration-honest** tool that turns the public federal hydrography + USGS species distributions + peer-reviewed thermal biology + real-time NOAA weather + observed water temperature into a **per-reach, per-day, relative suitability index map** for any US river or stream, filterable by body of water, county, and state.

It does this by running entirely on the user's hardware: a hardware probe detects what compute is available (CPU, RAM, GPU, NPU), the model registry picks an appropriately-sized GGUF on first use, llama.cpp loads it on the right backend (CUDA / Metal / Vulkan / ROCm / CPU), and a DuckDB feature store anchors every reach-level join on NHDPlus HR COMID. The LLM is used for natural-language query parsing and narrative summarization; it is **never** used to produce the underlying probabilities or indices.

The output explicitly does not pretend to be more than it is. At v0 the multiplicative 5-factor suitability composite (BRT presence prior x species thermal niche x flow condition x seasonal phenology x flow anomaly) is presented as a **relative ranking in [0, 1]**, NOT a calibrated catch probability. Calibrated catch probability is v1 work once SSN2 ssn_glm(binomial) is trained on observed catch data; until then, applying the Charbonneau hyperstability correction to a BRT presence probability would be a statistical category error and is explicitly disabled.

## 4. The four product pillars (in priority order)

### Pillar 1 - Local-first

All inference and all data live on the user's machine. SaaS frontier-model escalation is opt-in per query, never the default. No telemetry. No surprise network calls after data is pulled. The tool runs in airplane mode once the feature store is populated.

### Pillar 2 - Hardware-adaptive

Auto-detect CPU microarchitecture, RAM, GPU vendor + VRAM + CUDA runtime, NPU. Select the llama.cpp backend variant and GGUF quant that fits the machine. Do not hardcode a model choice anywhere outside the registry. When no model fits the hardware, emit an explicit "no model fits your hardware for `<task>`" error - never silently degrade.

### Pillar 3 - Honest calibration and provenance

Every probability is a `CalibratedProbability` with a structurally-enforced 95% interval that no surface can strip (FR-6.4). The interval is propagated through the multiplicative chain so downstream consumers see lower and upper alongside the point estimate. Every factor in the scoring chain carries a source tag; missing data returns the honest `not_modeled:<reason>` tag rather than a substituted value. No fabricated values, anywhere. The Charbonneau 2025 hyperstability constant is recorded on every reach but explicitly not applied to the BRT path at v0 (statistical category error per multi-AI review STAT-01).

### Pillar 4 - Honest ethics

Sensitive-species suppression is a hard gate, not a suggestion. Bull trout, Apache trout, Gila trout, Paiute cutthroat, Lahontan cutthroat at HUC-10 per CR-1.1. Greenback, Rio Grande, westslope, Yellowstone, Bonneville, Colorado River cutthroat at HUC-12 per CR-1.2. Tribally-managed waters are masked with a redirect to the tribe's own data resources (no scraping of CRITFC layers). EPA ATTAINS impaired-water status is surfaced as an alert by default, regardless of trip-intent impact.

## 5. What "shippable v0" looks like

A user installs Angler_AI on their laptop. The hardware probe runs in under 5 seconds. The model registry downloads an appropriately-sized GGUF in under 15 minutes on a 100 Mbps connection. The user runs `angler-ai ingest` for their state of interest, which populates the local DuckDB feature store from USGS NHDPlus HR, USGS NWIS, EPA ATTAINS, EPA WQP, USGS BRT v2.0, and state stocking (PA at v0; ID/MT/WY/VA at v1).

The user then asks:

> "Good brown trout fishing on the Big Hole River next two weeks"

and gets back:
- A per-day, per-reach colored PNG map for every day in the 14-16 day forecast window (Big Hole brown trout: `out/forecasts/Big_Hole/brown_trout/daily/2026-06-18.png` ... `2026-07-03.png`)
- A `summary.png` showing the best-day-over-window aggregate
- A `narratives.md` paragraph that names the best day and the worst day explicitly, with the exact dates AND a WHY explanation comparing the day's mean water temperature against the brown trout thermal niche (Elliott 1994: T_optimum=13.9, T_preferred_upper=19, T_lethal_upper=25) plus the day's flow factor, seasonal factor, and anomaly factor
- All driven by real federal data with source tags surfaced in the map caption
- Calibrated 95% interval preserved end-to-end so the user can see uncertainty alongside the point estimate

The user runs `angler-ai ask "why does the upper Big Hole score higher than the lower"` and the 3-agent reasoning pipeline (Profile -> Planning -> Analyst) calls real tools against the DuckDB feature store and returns a narrative grounded in real factor values with per-claim tool-call citations - no fabricated values.

The user can drop the GeoJSON output into QGIS / Felt / kepler.gl for further visualization. The CLI emits structured JSON for programmatic consumers. The HTTP API serves OpenAI-compatible chat endpoints plus Angler_AI-specific routes for `/v1/reaches`, `/v1/probability`, `/v1/anomaly`, `/v1/regulations`, `/v1/map.geojson`.

## 6. What v0 is intentionally NOT

These are excluded from v0 to keep the scope shippable. Each is a deliberate scope decision, not an oversight. The full list is in `docs/REQUIREMENTS.md` section 5; the headline exclusions:

- **No lakes / reservoirs / saltwater.** Rivers and streams only. Pattern overlap exists but is deferred.
- **No catch probability claim.** The 5-factor composite is a relative suitability index in [0, 1] at v0. Calibrated catch probability requires SSN2 ssn_glm(binomial) trained on observed catch data; deferred to v1.
- **No mobile apps.** Server-side / desktop only at v0.
- **No on-device fine-tuning / LoRA / distillation / model merging.** v2.
- **No web UI / GUI.** CLI + local FastAPI + GeoJSON / PNG / Markdown output files. v1+ for Tauri or Vite frontend.
- **No telemetry.** Ever, by default.
- **No CRITFC scraping.** Tribal sovereignty mask + redirect at v0; partnership integrations are v1+ with explicit tribal opt-in.
- **No multi-state regulation Q&A.** Single-state PA exemplar at v0; expansion at v1.
- **No Pacific NW anadromous queries.** StreamNet/CAX integration is v1.

## 7. What "v1 and beyond" looks like

The CalibratedProbability and `SpeciesPrior` interfaces are designed to be swap-in-stable. v1 work in priority order:

1. **SSN2 ssn_glm(binomial) on observed catch data** replaces the BRT prior as `base_p`. The `CalibratedProbability` interface holds; the suitability index becomes a calibrated catch probability and gets renamed back. M4 Brier <= 0.22 release gate is measured at this point.
2. **NorWeST + EcoSHEDS gridded temperature loads** replace the Mohseni-Stefan + IDW substrate with peer-reviewed model output for the majority of US reaches. Resolver priority list already includes the slots.
3. **State stocking + regulations for ID / MT / WY / VA** following the PA PFBC discovery pattern.
4. **PG-GNN nowcasts for per-reach temperature and flow** when training data and compute are available.
5. **HydroGEM in the forecast pipeline** (576-hour 12-channel feature engineering hookup; currently called only from the `ask` agent).
6. **Asymmetric thermal niche** anchored on `t_lethal_upper_c` per multi-AI review FB-1.
7. **Hoot-owl / regulation-closure caveats** on every map caption.
8. **Pacific Northwest anadromous queries** via StreamNet / CAX.
9. **Web UI** (Tauri or Vite).

## 8. The bet

The bet is that **honest calibration + real federal data + per-day temporal forecasting + local-first execution** is a defensible product position that competing apps (TroutRoutes, FishAngler, Navionics, Anglers Atlas, OnX Fish) do not occupy. Each of those tools makes one or two of those bets; none makes all four. The market gap is anglers who fish rivers more than 20 days a year, want to see calibrated probabilities rather than POI pins, care about the data lineage, and would rather their data stay on their machine.

The bet is NOT that v0 is a finished product. v0 is the substrate: hardware-adaptive inference, a populated feature store, calibrated species priors, a relative-suitability scoring layer, observation-anchored temperature modeling, an honest per-day forecast pipeline, an LLM narrative generator, and a hard sensitive-species ethics gate. v1 is when this substrate gets fed observed catch data and the suitability index becomes a calibrated probability.

## 9. Non-goals

- Not a SaaS web app. The tool runs on the user's machine.
- Not a replacement for state regulation books. We surface and link to authoritative sources; we do not legally arbitrate.
- Not a calibrated catch-probability estimator at v0. The multiplicative composite is honestly a relative suitability index. The literature warns recreational CPUE is hyperstable; v1 will replace BRT with SSN2 trained on observed catch data and a defensible probability calibration.
- Not a biomass estimator. Recreational CPUE is hyperstable; we surface calibrated probabilities with explicit intervals, not point biomass.
- Not a social-features clone. Fishbrain and Anglers Atlas already serve catch-logging social features; that is not the product we are building.

---

## 10. Provenance of this vision

This document is the codified form of the initial product prompt from the user, refined through:

- Six deep-research artifacts (`research/01..06_*.md`) that enumerated the data substrate, peer-reviewed habitat science, hyperstability mechanism, and the four-pillar architecture
- The 2026-06-17 design conversation that resolved ten previously-open product questions (logged in `docs/v0_design.md` section 9)
- The 2026-06-17 multi-AI review (5 reviewers, 22 high/critical findings, 20 survived adversarial verification) that flagged the "catch probability" framing as a statistical category error and produced the `suitability_index` rename
- The 2026-06-18 per-day output expansion and the observation-anchored Mohseni-Stefan temperature fix that addressed user feedback on flat-IDW behavior

If the vision changes during build, update this document AND `docs/REQUIREMENTS.md` AND `docs/SUCCESS_CRITERIA.md`. The three are a triad: vision tells you why the requirements exist; requirements tell you what success looks like; success criteria tell you how to measure it.
