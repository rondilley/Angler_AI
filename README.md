# Angler_AI

Local-first, hardware-adaptive tool for researching, analyzing, and mapping fishing in US rivers and streams. Produces per-reach, per-day **relative suitability index** maps that fuse federal hydrography, USGS species distributions, peer-reviewed thermal niches, real-time weather, observed water temperature, and species-specific biology. Inference runs locally via llama.cpp; the tool auto-detects your hardware and selects appropriate models on first use.

**Status:** v0 working through M8 + Tier 1 multi-AI review fixes. Hardware-adaptive llama.cpp inference, real federal data ingest, per-day relative suitability maps over a 14-16 day window, observation-anchored water-temperature modeling, and a 3-agent reasoning pipeline with best/worst day narratives are all functional.

The output is a **relative suitability index in [0, 1]**, NOT a calibrated catch probability. Multi-AI review (2026-06-17) flagged the calibrated-probability framing as a statistical category error until M4 catch-data validation - the documents and code have been renamed accordingly.

## Design, requirements, and audits

- [**Architecture**](docs/ARCHITECTURE.md) - current state of the 10-component system; read this first
- [**Vision**](docs/VISION.md) - what the tool is FOR
- [**Requirements**](docs/REQUIREMENTS.md) - stable-ID functional, non-functional, data, ethics requirements
- [**Success criteria**](docs/SUCCESS_CRITERIA.md) - per-milestone release gates and per-use-case acceptance
- [Original design doc](docs/v0_design.md) - 2026-06-17; some sections superseded by ARCHITECTURE.md, kept as historical design rationale
- [Forecast analysis - what goes into the maps](docs/forecast_analysis.md)
- [Multi-AI review report (2026-06-17)](.ai-reviews/multi_ai_review_2026-06-17.md)
- [Tasks - open punch list](tasks/todo.md)
- [Lessons - self-improvement loop](tasks/lessons.md)
- [Research artifacts](research/) - six deep-research passes that anchor the design

## What this is not

- Not a SaaS web app. The tool runs on your machine; data ingest, model inference, and predictions all happen locally.
- Not a replacement for state regulation books. We surface and link to authoritative sources; we do not legally arbitrate.
- Not a calibrated catch-probability estimator at v0. The multiplicative composite is honestly a relative suitability index in [0, 1]. The literature warns recreational CPUE is hyperstable; v1 will replace BRT with SSN2 ssn_glm(binomial) trained on observed catch data and a defensible probability calibration.

## Architectural pillars

1. **Local-first.** All inference, all data, all predictions on your machine. SaaS frontier-model escalation is opt-in per query.
2. **Hardware-adaptive.** Auto-detect CPU, RAM, GPU, NPU. Select llama.cpp backend variant (CUDA / Metal / Vulkan / ROCm / CPU) and download appropriately-sized GGUF models on first use.
3. **No fabricated values.** Every factor carries a source tag. Missing data returns `factor=1.0` tagged `not_modeled:<reason>` rather than substituting a value. This is enforced in code, not just policy.
4. **Calibrated provenance.** Every species presence carries a `CalibratedProbability` with structurally-enforced 95% interval. The interval is propagated through the 5-factor suitability chain (FR-6.4) so no surface can strip it. The Charbonneau 2025 hyperstability constant is recorded on every reach but explicitly NOT applied to the BRT path at v0 (applying `p^(1/0.23)` to a unit-interval presence probability is a category error per multi-AI review STAT-01).
5. **Honest ethics.** Bull trout, Apache trout, Gila trout, Paiute cutthroat, Lahontan cutthroat all suppressed at HUC-10 per CR-1.1. Greenback/Rio Grande/Westslope/Yellowstone/Bonneville/Colorado River cutthroat suppressed at HUC-12 per CR-1.2. Tribal-managed waters masked with redirect; EPA ATTAINS impaired-water status surfaced as an alert by default.

## License

GPL-3.0 for the codebase. Each ingested data source and each downloaded model carries its own license, surfaced at acquisition time.

## Python version

Python 3.10, 3.11, or 3.12. (3.13+ pending llama-cpp-python wheel availability.)

## Quick start

```bash
git clone https://github.com/yourrepo/Angler_AI
cd Angler_AI
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev,prediction]"

# pick the llama-cpp-python wheel for your hardware:
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125

angler-ai probe                                  # detect hardware + backend
angler-ai pull-models --profile auto             # download GGUF models
angler-ai ingest --source nhdplus --state ID --huc8 17040203   # one HUC8 at a time
angler-ai ingest --source v2_xwalk --state ID --huc8 17040203
angler-ai ingest --source brt --state ID                       # one-time BRT load
angler-ai species --comid <reach_comid>          # top species at a reach
angler-ai map --species "brown trout" --huc8 17040203 --out hf.geojson
angler-ai forecast --species "brown trout" --huc8 17040203 --out hf_brown.png
angler-ai ask "good brown trout fishing on the Henrys Fork next week"
```

For multi-water per-day analysis with summary + 16 daily maps + LLM narrative per (water, species), see `.venv/run_western_forecasts.py`.

## What ships in v0

- **Hardware probe**: archspec + pynvml + CUDA runtime detection; selects llama.cpp wheel index variant (CPU / CUDA cu118/cu121/cu122/cu123/cu124/cu125/cu130/cu132 / Metal / Vulkan).
- **Local inference**: llama.cpp via llama-cpp-python, OpenAI-compatible HTTP server.
- **Federal data ingest** (idempotent, into a local DuckDB feature store):
  - USGS NHDPlus HR hydrography (any HUC8, any of 50 states)
  - USGS NHDPlus V2.1 -> HR crosswalk via shared REACHCODE
  - USGS BRT v2.0 fluvial fish SDM (419 species, 112M predictions; DOI 10.5066/P1UV25FW)
  - USGS NWIS continuous water-temperature observations (parameter 00010)
  - USGS NWIS streamflow + discharge (parameter 00060)
  - EPA Water Quality Portal + EPA ATTAINS impaired-waters
  - NOAA NWS daily weather forecast (api.weather.gov, days 0-7)
  - Open-Meteo 16-day extended forecast (days 8-16; tagged `OpenMeteo_daily`)
  - PA PFBC trout stocking (PA only at v0; ID/MT/WY at v1)
- **Per-day suitability index** (`out/forecasts/<water>/<species>/`):
  - `summary.png` - best-day-over-window aggregate
  - `daily/<YYYY-MM-DD>.png` - one PNG per day in the 14-16 day window
  - 5-factor multiplicative chain: BRT calibrated prior x species thermal niche x flow factor x seasonal phenology x flow z-score anomaly
  - Per-reach 95% interval propagated through the factor chain
  - Output clamped to [0, 1]; seasonal_factor capped at 1.0 with audit tag
  - Renders via matplotlib + RdYlGn (red = low suitability, green = high)
- **Water-temperature modeling**:
  - Source priority: `NWIS_obs` (direct gauge measurement) > `NWIS_interp_air_adjusted` (IDW spatial anchor + per-day Mohseni-Stefan air-temp delta) > `NWIS_interp` (pure IDW) > `NWIS_air_projected` (Mohseni-Stefan only) > `not_modeled`
  - IDW restricted to same HUC10 AND within +-1 stream order of the receiving reach (no mainstem gauge contaminating a cold cirque tributary)
  - Mohseni-Stefan defaults stratified by stream order: small headwater (alpha=16, beta=12), medium (alpha=19, beta=14), mainstem (alpha=22, beta=15)
- **Peer-reviewed species thermal niches** for 12 species: Elliott 1994 (brown), Wehrly 2007 (brook), Bear 2007 (cutthroat), Selong 2001 (bull), Brinkman 2013 (mountain whitefish), Hubert 1985 / Lamothe 2003 (arctic grayling), Myrick 2000 (rainbow), Wismer 1987 (smallmouth), Casselman 1996 (northern pike), Stewart 1983 (lake trout).
- **Peer-reviewed species seasonal phenology** for the same 12 species: monthly activity multipliers from documented spawning + peak-feeding windows.
- **LLM narrative per (water, species)** describing the temporal trajectory: explicit best day and worst day with WHY explanation grounded in per-day factor breakdown (mean water temp vs species thermal niche, flow factor change, seasonal factor, anomaly).
- **Reasoning agent** (`angler-ai ask`): MARSHA-style 3-agent pipeline (Profile -> Planning -> Analyst) grounded in real feature-store tool outputs.
- **Calibration** (mandatory): 95% interval propagated through the factor chain at the type level.
- **Ethics**: bull trout HUC-10 suppression; Apache, Gila, Paiute cutthroat, Lahontan cutthroat HUC-10 per CR-1.1; remaining native cutthroats HUC-12 per CR-1.2; ATTAINS surfaced by default; tribal-waters mask.

## Honest gaps and what's deferred to v1

- **The suitability index is a relative ranking in [0, 1], NOT a calibrated catch probability.** Multi-AI review confirmed this is the correct framing at v0. v1 will replace BRT with SSN2 ssn_glm(binomial) trained on observed catch and produce a calibrated probability.
- **No fabricated values.** Reaches without modeled water temperature score `thermal_factor=1.0` (neutral) with `not_modeled:<reason>` source tag. NorWeST/EcoSHEDS gridded predictions, PG-GNN nowcasts, SSN2 spatial stream-network models are designed but not loaded at v0.
- **State stocking + regulations**: PA only at v0. ID/MT/WY/VA deferred to v1.
- **HydroGEM anomaly detection**: real model loaded and tested against the published synthetic-anomaly pickle. Production 576-hour NWIS series ingest per gauge is on-demand via `ask`; the forecast pipeline uses a simpler statistical z-score anomaly detector on the same gauge data.
- **NOAA forecast horizon**: NWS provides ~7 days of daily forecast; days 8-16 use Open-Meteo (real second-source extended forecast, tagged distinctly).
- **Asymmetric thermal niche**: the v0 Gaussian bell is symmetric; salmonid thermal physiology has asymmetric warm-tail collapse. Tier 3 fix per multi-AI review FB-1 / PV-9.
- **LLM faithfulness check**: minor narrative drift can occur (e.g., conflating thermal_factor with base_p). Tier 2 fix per multi-AI review PV-2/3/4 (automated FactorBreakdown-vs-narrative diff is on the roadmap).

## Status of v0 milestones

- [x] M1 - Hardware probe + inference smoke test
- [x] M2 - Federal core + PA ingestion (NHDPlus HR, V2 xwalk, ATTAINS, NWIS, WQP, PA PFBC)
- [x] M3 - USGS BRT priors integration (419 species, V2-to-HR crosswalk)
- [x] M4 - Calibrated species probability map (GeoJSON, interval propagation)
- [x] M5 - Honest temperature substrate (NWIS_obs + resolver; NorWeST/EcoSHEDS deferred to v1, NO proxy fabrication)
- [x] M6 - HydroGEM + MARSHA-style 3-agent reasoning
- [x] **M7** - NOAA forecast + species thermal niches + PNG forecast map (initial single-aggregate output)
- [x] **M8** - Multi-factor scoring closeout: seasonal phenology, Mohseni-Stefan, NWIS_interp, real discharge ratio, flow z-score anomaly, per-HUC12 NWS sampling, Open-Meteo days 8-16, LLM narrative
- [x] **Multi-AI review + Tier 1 fixes** - rename score -> suitability_index, propagate interval through factor chain, clamp to [0,1], drop hyperstability from BRT path, add missing ESA species to sensitive-species seed
- [x] **Per-day maps** - one PNG per day in the 14-16 day window, plus summary; best/worst day narrative with WHY explanation
- [x] **Observation-anchored Mohseni-Stefan** - per-day temporal variation on IDW-anchored reaches (`NWIS_interp_air_adjusted` source tag)

See [SUCCESS_CRITERIA.md](docs/SUCCESS_CRITERIA.md) for per-milestone release gates and the full v0 acceptance suite.
