# Forecast: what goes into the maps

This document describes exactly what data, models, and detectors produce the
per-reach per-day **relative suitability index** maps under `out/forecasts/`.

**Per-day update (2026-06-18):** the driver now writes a `summary.png`
(best-day-over-window aggregate) AND a `daily/<YYYY-MM-DD>.png` for every
day in the 14-16 day forecast window. 16 daily maps per (water, species)
plus 1 summary = 17 PNGs per (water, species). Best/worst day are surfaced
in the per-(water, species) LLM narrative with a WHY explanation grounded
in the per-day factor breakdown.

**Observation-anchored Mohseni-Stefan (2026-06-18):** the IDW spatial
interpolator now anchors on the most recent NWIS_obs value and adds a
per-day delta driven by the Mohseni-Stefan air-temperature projection.
Source tag `NWIS_interp_air_adjusted`. This fixes a "15 flat days then a
jump" artifact in watersheds where IDW dominated.

**Tier 1 update (2026-06-18, multi-AI review):** the output is now a
"relative suitability index" in [0, 1], explicitly NOT a calibrated catch
probability. The Charbonneau hyperstability correction has been dropped
from the BRT path (cpue_weight=0) because applying `p^(1/0.23)` to a
unit-interval presence probability is a statistical category error. The
95% calibrated interval is propagated through the factor chain
(`suitability_lower` / `suitability_upper` on `DailyScore`) so FR-6.4
holds end-to-end. The multiplicative product is clamped to [0, 1] and the
seasonal_factor cap at 1.0 is enforced (was previously [0.3, 1.1]).

**M8 update (2026-06-17):** the scoring chain was extended from 3 factors
(base x thermal x flow) to 5 factors (base x thermal x flow x seasonal x
anomaly). Per-day water temperature, per-HUC12 forecast sampling, real
discharge ratio, Open-Meteo days 8-16, and an LLM-generated narrative
summary are now wired. 100% of reaches have modeled water temperature
via the resolver priority NWIS_obs > NWIS_interp_air_adjusted > NWIS_interp
> NWIS_air_projected.

## Output structure

```
out/forecasts/
  <water>/                                          # Big_Hole, Madison_and_Firehole, ...
    <species>/                                      # brown_trout, arctic_grayling, ...
      summary.png                                   # best-day-over-window aggregate
      daily/
        2026-06-18.png                              # one map per day
        2026-06-19.png
        ...
        2026-07-03.png
  narratives.md                                     # one narrative per (water, species)
                                                    # with best/worst day WHY + per-day table
```

## 1. End-to-end data flow (per-day, per-reach, per-species)

```
        USGS BRT v2.0              Peer-reviewed niches + phenology
        boosted regression         (Elliott, Wehrly, Bear, Selong,
        tree presence prior        Brinkman, Lamothe, Myrick, ...)
        (419 species,                  |               |
        ~270k V2 COMIDs)               |               |
              |                        v               v
              v                  thermal niche    seasonal phenology
        hyperstability         (Gaussian bell)   (monthly multiplier
        NOT applied at v0           |             capped at 1.0)
        (cpue_weight=0)             |                  |
              |                     |                  |
              v                     v                  v
        base_p (raw BRT)      thermal_factor    seasonal_factor
              |                in [0.01, 1.0]    in [0.3, 1.0]
              |                     ^                  ^
              |                     |                  |
              |              per-day water temp        |
              |          (NWIS_obs > IDW air-adj       |
              |             > IDW > Mohseni)           |
              |                     ^                  |
              |                     |                  |
              |              NWIS_obs (param 00010)    |
              |              IDW within HUC10 and      |
              |              +-1 stream order +        |
              |              Mohseni-Stefan air        |
              |              delta on top              |
              |                                        |
              |       NWS+OpenMeteo precip prob        |
              |       (api.weather.gov + open-meteo)   |
              |             |                          |
              |             v                          |
              |       flow_factor in [0.6, 1.0]        |
              |       OR real-discharge ratio          |
              |       (NWIS 00060 recent vs            |
              |        30-day baseline)                |
              |             |                          |
              |             |        USGS NWIS         |
              |             |        flow z-score      |
              |             |        anomaly per gauge |
              |             |             |            |
              |             |             v            |
              |             |       anomaly_factor     |
              |             |       in [0.7, 1.0]      |
              |             |             |            |
              v             v             v            v
       suitability_index(reach, day)  =  CLAMP_[0,1](
              base_p * thermal_factor * flow_factor
              * seasonal_factor * anomaly_factor
       )
       suitability_lower / upper propagated through same chain
       from the BRT CalibratedProbability 95% interval (FR-6.4)
              |
              v
       NHDPlus HR              matplotlib + RdYlGn
       reach geometry  ----->  LineCollection
       (MULTILINESTRING)       per-day PNG render
                               +
                               summary.png
                               (max over the window per reach)
                               +
                               LLM narrative
                               (best/worst day with WHY explanation,
                                grounded in per-day FactorBreakdown)
```

The scoring chain is deterministic Python. The LLM (Llama 3.2 3B Q4_K_M on
RTX 5090, hardware-adaptive) is used ONLY to produce the per-(water, species)
narrative summary; it does not produce any probabilities or indices.

## 2. Source data, ranked by their role

Every value passed to a multiplier has a source tag in the score's
FactorBreakdown. Missing data returns factor=1.0 with `not_modeled` tag.

### 2.1 USGS NHDPlus High Resolution -- reach geometry substrate

- **Endpoint:** `https://hydro.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/MapServer`
- **License:** public domain
- **Role:** Every reach drawn in the maps comes from NHDPlus HR.
  MULTILINESTRING geometry, with `reachcode`, `gnis_name`, `huc8`, `huc10`,
  `huc12`, `stream_order`, `drainage_area_km2`, `state_fips`, `county_fips`.
- **Loaded for the 5 waters:** 14,804 reaches across 4 HUC8s
  (17040203 Henrys Fork; 10020007 Madison + Firehole; 10020004 Big Hole;
  10020005 Jefferson).

### 2.2 USGS NHDPlus V2.1 -- crosswalk source

- **Endpoint:** `https://watersgeo.epa.gov/arcgis/rest/services/NHDPlus_NP21/NHDSnapshot_NP21/MapServer/0`
- **License:** public domain
- **Role:** USGS BRT v2.0 predictions are keyed by NHDPlus V2.1 COMID; our
  geometry table is keyed by NHDPlus HR COMID. The two versions share the
  14-digit REACHCODE, so we build `xwalk_v2_to_hr` with two tiers:
    - `reachcode_exact` (confidence 1.0)
    - `huc10_proximity` (confidence 0.5) for HR reaches finer than V2
- **Loaded for the 5 waters:** 17,883 crosswalk rows.

### 2.3 USGS BRT v2.0 -- the species presence prior (THE base probability)

- **Source:** USGS ScienceBase item `6760bf81d34e03058f220b48`,
  `fluvial_fish_brt_predictions_v2_0.parquet.gzip` (1.06 GB) plus
  `fish_list_v2_0.csv`
- **DOI:** 10.5066/P1UV25FW
- **Citation:** Yu, S.L., Cooper, A.R., Ross, J., McKerrow, A.J.,
  Wieferich, D.J., Infante, D.M. 2023. Fluvial fish native distributions
  using NHDPlusV2.1 and Boosted Regression Tree models. USGS SIR 2023-5088.
- **License:** public domain (US government work)
- **Role:** the `base_p` in every score. 419 species x ~270,000 V2 reaches =
  ~112 million predictions. Each prediction is a Boosted Regression Tree
  output `P(species present at this reach)` against ~30 NHDPlus V2.1
  landscape and EROM covariates (stream order, drainage area, gradient,
  mean annual flow, mean annual air temperature, percent impervious cover,
  geology, climate, etc.).
- **Critical caveat:** BRT is fit on NATIVE-RANGE occurrence records.
  Introduced populations are excluded. Rainbow trout returns "NO PRIORS"
  for Henrys Fork / Madison / Big Hole / Jefferson because the model
  treats those headwaters as outside the species' native Pacific
  distribution. (This is the same reason smallmouth bass returns NO PRIORS
  for parts of PA where the species is now ubiquitous but historically
  introduced.)
- **Loaded for the 5 waters:** 16,491 species-reach prior rows.

### 2.4 USGS NWIS water temperature -- observed water temp at gauges

- **Endpoint:** `https://api.waterdata.usgs.gov/ogcapi/v0/collections/continuous`
- **Parameter code:** `00010` (degrees Celsius at the sensor)
- **License:** public domain
- **Role:** real observed water temperature, daily-averaged over a 7-day
  lookback at every USGS stream gauge in the HUC8 that records 00010.
  Joined to the nearest NHDPlus HR reach inside the HUC8 by `ST_Distance`
  and inserted into `reach_temperature` with `source='NWIS_obs'`.
- **Coverage:** sparse. Of 4,000 reaches in a typical HUC8, only the
  handful that contain a real USGS gauge (typically 1-10) end up with a
  water-temp value. **All other reaches surface `thermal_factor=1.0`
  with `not_modeled` tag** -- we do not interpolate or fabricate.
- **What we DON'T have at v0:** NorWeST gridded predictions (West),
  EcoSHEDS NECD / Letcher hierarchical Bayesian temperature (East), and
  the physics-guided GNN nowcast. These are designed and stubbed; ingest
  is v1 work.

### 2.5 NOAA NWS daily forecast -- precipitation probability + air temp

- **Endpoint:** `https://api.weather.gov/points/{lat},{lon}` then
  `forecast` URL
- **License:** public domain (NOAA)
- **Role:** sampled ONCE per HUC8 at the watershed centroid (mean of all
  loaded reach centroids). Yields ~7 real NWS NDFD forecast days; days
  8-14 are filled by persistence-projecting the last NWS day with the
  explicit source tag `persistence_projection`.
- **Used:** daily precipitation-probability percent. (Air temp is loaded
  but not currently fed into the flow_factor.)
- **Limitation:** a single watershed-level forecast point applied to every
  reach in the HUC8. Microclimate variation within the HUC8 is not
  resolved.

### 2.6 Peer-reviewed species thermal niches

- **Storage:** `src/angler_ai/prediction/thermal_niches.py`
- **License:** literature citations (values are factual)
- **Role:** species-specific thermal preference curve. For each species we
  encode `(T_optimum, T_preferred_upper, T_lethal_upper,
  T_preferred_lower)` from peer-reviewed papers.
- **Sources (cited inline in the code):**

  | Species | Citation | T_opt | Preferred upper | Lethal upper |
  |---|---|---|---|---|
  | Brown trout | Elliott 1994 (Oxford UP) | 13.9 C | 19 | 25 |
  | Rainbow trout | Myrick & Cech 2000 Rev Fish Biol Fisheries 10 | 15 | 20 | 25 |
  | Brook trout | Wehrly et al. 2007 TAFS 136:365 | 14 | 18 | 24 |
  | Cutthroat trout | Bear, McMahon, Zale 2007 TAFS 136:1113 | 13 | 18 | 24 |
  | Bull trout | Selong et al. 2001 TAFS 130:1026 | 10 | 13 | 16 |
  | Arctic grayling | Lamothe & Magee 2003; Hubert et al. 1985 | 10 | 15 | 23 |
  | Mountain whitefish | Brinkman et al. 2013 TAFS 142:824 | 11 | 17 | 23 |
  | Smallmouth bass | Wismer & Christie 1987 GLFC SP 87-3 | 24 | 28 | 33 |
  | Northern pike | Casselman & Lewis 1996 CJFAS 53:161 | 19 | 25 | 30 |
  | Lake trout | Stewart et al. 1983 TAFS 112:751 | 10 | 15 | 20 |
- **Functional form:** symmetric Gaussian bell anchored on T_optimum with
  sigma fit so `suitability(T_preferred_upper) ~= 0.5` and
  `suitability(T_lethal_upper) ~= 0.1`. Returns `1.0` at exact optimum,
  clipped to `[0.01, 1.0]`. Pure deterministic math; no ML training.

### 2.7 Charbonneau 2025 hyperstability constant

- **Source:** Charbonneau et al. 2025 TAFS 154(4):339
- **Geography:** 14 British Columbia steelhead streams, 1972-2019
- **Value:** beta = 0.23 (standard error ~0.05)
- **Role:** mandatory calibration for any CPUE-derived probability.
  Applied at the SpeciesPrior layer with `cpue_weight=0.5`:
  `p_calibrated = 0.5 * p_raw^(1/beta) + 0.5 * p_raw`
- **Why:** Erisman et al. 2011 "illusion of plenty"; the same hyperstable
  pattern across walleye, bass, and trout. Charbonneau confirms beta=0.23
  produces ~40% CPUE decline from 50% true population decline, ~77% CPUE
  decline from 90% true decline. Without this correction the BRT prior
  would overstate catchability everywhere.

## 3. The actual probability math (M8 5-factor chain)

Each reach segment in the PNG is colored by the MAX over the 14-16 day
forecast window of a 5-factor multiplicative score:

```
score(reach, day) = base_p * thermal_factor * flow_factor * seasonal_factor * anomaly_factor
PNG color = max over the forecast window
```

Score chain in detail:

| Term | Source / value | When it falls back to 1.0 |
|---|---|---|
| `base_p` | USGS BRT v2.0 + Charbonneau 2025 hyperstability (beta=0.23, cpue_weight=0.5) | reach has no V2 join (row omitted, not faked) |
| `thermal_factor` | Gaussian niche bell evaluated at per-day water temp from: **NWIS_obs** (gauge measurement) > **NWIS_interp** (IDW from gauges in same HUC10) > **NWIS_air_projected** (Mohseni-Stefan logistic from forecast air temp) | species not in niche registry; carry-forward used to avoid gap-day 1.0 |
| `flow_factor` | Real discharge ratio (recent 7-day mean / 30-day baseline median), OR NWS precip-probability 3-bin proxy if discharge ingest fails | both unavailable |
| `seasonal_factor` | Species monthly activity from published phenology (spawning attenuation, peak-feeding boost) | species not in phenology registry |
| `anomaly_factor` | Statistical z-score on recent vs baseline gauge discharge; flagged at z>=2.0; mapped to reaches by nearest gauge | no discharge data in HUC8 |

Inputs and their sources:

(legacy 3-factor table superseded by the M8 5-factor table above.)

The 95% prediction interval on `base_p` is also computed but is not
currently encoded in the PNG color. It is preserved on the underlying
`CalibratedProbability` and is available via `angler-ai map` (GeoJSON)
and through the API.

## 4. ML / AI detectors actually in this codebase

This section names every ML/statistical detector present in the
codebase, and whether each is used by the forecast pipeline that
produced the colored maps. "Used in maps" = called by the
`forecast` CLI path; "loaded" = available to other commands.

| Detector | Type | Role | Used in maps |
|---|---|---|---|
| USGS BRT v2.0 | Boosted Regression Tree species SDM | base_p; the species-presence prior | YES |
| Hyperstability correction | Statistical (Charbonneau 2025 closed form) | mandatory CPUE downweight on base_p | YES |
| Naive beta Wilson interval | Statistical | 95% prediction interval on base_p | computed but not encoded in PNG color |
| Species thermal-niche bell | Analytical Gaussian | thermal_factor | YES |
| **Mohseni-Stefan logistic** | Closed-form air-to-water temperature regression (Mohseni Stefan Erickson 1998 WRR 34) | per-day water-temp projection from forecast air temp | YES (M8) |
| **IDW spatial temp interpolation** | Inverse-distance weighting from NWIS_obs gauges in same HUC10 | per-day water-temp on reaches without a direct gauge | YES (M8) |
| **Species seasonal phenology** | Published monthly activity table | seasonal_factor | YES (M8) |
| Real discharge ratio | recent 7-day mean / 30-day baseline median | flow_factor (when discharge ingest succeeds) | YES (M8) |
| Flow z-score anomaly | Statistical, recent vs baseline gauge discharge | anomaly_factor; reaches mapped to nearest gauge | YES (M8) |
| Flow modifier from precip | Heuristic 3-bin rule (Bell 2022 review) | flow_factor (fallback if no discharge data) | YES |
| **HydroGEM TCN-Transformer** | TCN encoder + Transformer + TCN decoder + MultiScaleAnomalyHead, CC-BY-NC-4.0 | streamflow anomaly detection on 576-hour USGS NWIS windows | NO (loaded; available to the `ask` reasoning agent; flow z-score is the forecast-pipeline equivalent) |
| SSN2 spatial stream-network GLM | Bayesian `ssn_glm(family=binomial)` via R | reach-level binary catch prob with stream-network autocorrelation | NO (NotImplementedError stub at v0; designed for v1) |
| PG-GNN | Physics-guided graph neural network | water-temperature and flow nowcast per reach | NO (NotImplementedError stub at v0; designed for v1) |
| EcoSHEDS BTO / NorWeST | gridded peer-reviewed temperature models | per-reach temperature substrate | NO (NotImplementedError stub at v0; designed for v1) |
| Stan hurdle gamma harvest model | Bayesian hurdle-gamma | per-trip harvest probability | NO (NotImplementedError stub at v0) |

## 5. Pattern detection

"Pattern detection" in the strict ML sense means the HydroGEM anomaly
detector. It is in the codebase at `src/angler_ai/prediction/hydrogem.py`
+ `hydrogem_arch.py`:

- Vendored from the Ejokhan/HydroGEM HuggingFace repo
- License: CC-BY-NC-4.0 (non-commercial)
- Architecture: TCN encoder -> Transformer -> TCN decoder, plus a
  MultiScaleAnomalyHead, trained on USGS NWIS streamflow series
- Tested end-to-end against the published synthetic-anomaly pickle
- Available via `angler-ai ask` -- the Analyst agent has a `hydrogem`
  tool that runs it on real USGS NWIS data for a gauge

**HydroGEM is NOT currently invoked by the `forecast` CLI** that produced
the colored maps. The forecast pipeline is static-prior-driven; it
does not currently flag flow anomalies on the date window. That linkage
is straightforward to add (feed an anomaly-degraded flow_factor for
reaches whose gauge is anomalous) and is the natural next step.

## 6. LLM / SLM models

**M8 update:** the forecast pipeline now invokes the locally-loaded
Llama 3.2 3B Instruct Q4_K_M (or whatever the hardware-adaptive router
selected for the current machine) AFTER each PNG is rendered to produce
a 4-6 sentence narrative summary grounded in the real numeric outputs.
The narratives are written to `out/forecasts/narratives.md` alongside
the PNGs. The LLM is constrained by a strict system prompt that forbids
inventing species, reach counts, scores, or citations, and requires it
to surface the source tags verbatim (e.g. "thermal suitability was
modeled for 312 of 312 reaches via NWIS_interp").

The LLM is NOT involved in producing the probabilities themselves.
The 5-factor multiplication is deterministic Python: ingest -> project ->
multiply -> render. The LLM only summarizes the resulting numbers.

LLMs ARE used by the separate `angler-ai ask` command (the M6 Reasoning
Layer). For that command:

- **Model selection** is hardware-adaptive via `src/angler_ai/registry/`
  router. On an RTX 5090 (32 GB VRAM) the default is **Llama 3.2 3B
  Instruct, Q4_K_M GGUF**, loaded via llama.cpp.
- **Architecture:** MARSHA-style 3-agent pipeline.
  - **Profile agent** parses the user's NL query into intent, species,
    geographic filters
  - **Planning agent** decides which tools to call and in what order
  - **Analyst agent** issues tool calls against the DuckDB feature store
    (`brt_priors`, `attains_status`, `stocking_events`, `regulations`,
    `hydrogem`) and synthesizes a narrative grounded in the real outputs
- **Tools available to the Analyst:** species_priors, temperature
  resolver, ATTAINS lookup, stocking lookup, regulations lookup,
  HydroGEM anomaly, sensitive-species check, ethics gate.
- **Grounding rule:** Tools raise `ToolError` when data is missing; the
  Analyst surfaces the error in the response rather than fabricating.

Tier table for model selection (from the v0 design doc):

| VRAM tier | Default text model | Quant |
|---|---|---|
| < 6 GB | Llama 3.2 3B Instruct | Q4_K_M |
| 6-12 GB | Qwen3.5 9B / 4B | Q4_K_M |
| 12-20 GB | Qwen3.5 35B A3B / 27B | IQ2_M |
| 20-32 GB | Qwen3.5 35B A3B | Q4_K_XL |
| > 32 GB | Qwen3.5 122B A10B / 35B | IQ2_XXS |

Hardware probing uses archspec (CPU microarchitecture) + pynvml (NVIDIA
GPU) + a CUDA-runtime probe via ctypes.WinDLL. The probe drives both
the llama-cpp-python wheel index variant (CUDA cu118/cu121/cu122/cu123/
cu124/cu125/cu130/cu132, Metal, Vulkan, or CPU) and the GGUF quant
choice.

## 7. Honest limits of the M8 forecast

What was a 10-item gap list at M7 has been substantially closed at M8.
The remaining honest limits:

1. **The BRT prior is a STATIC native-distribution probability.** It does
   not vary by date. Date variation now comes from thermal_factor (which
   does vary per day via Mohseni-Stefan), flow_factor (per-day NWS precip),
   seasonal_factor (per-month phenology), and anomaly_factor (current-vs-
   baseline z-score at the gauge). Base_p itself remains static. To make
   BRT date-conditional we would need to retrain it on a per-month
   stratification, which is v1 work.
2. **Mohseni-Stefan uses default salmonid-stream parameters** (alpha=22,
   beta=15, gamma=0.18, mu=2). The fitting code is in place but at v0
   we do not yet have paired (T_air, T_water) history per reach to fit
   reach-specific parameters. This is a real-data limit: NWIS_obs gives
   us water temp but we have not yet stored paired centroid air temp
   in the DB.
3. **IDW spatial interpolation is within HUC10 only.** Reaches in a HUC10
   with NO gauge fall back to Mohseni-Stefan air projection. This is
   honest: HUC10 is a defensible neighborhood for thermal homogeneity.
4. **No state stocking or regulation overlays for ID/MT/WY.** PA PFBC
   stocking is loaded; ID / MT / WY state agencies are not. IDFG ingest
   module is a NotImplementedError stub pending IDFG ArcGIS Hub probe.
5. **HydroGEM TCN-Transformer is loaded but called only by `ask`.** The
   forecast pipeline uses a simpler statistical flow z-score detector
   for anomaly. HydroGEM's 576-hour fixed-window 12-channel input
   pipeline is v1 work.
6. **NorWeST gridded + EcoSHEDS NECD temperatures are not loaded.** They
   would replace the Mohseni-Stefan projection with peer-reviewed model
   output. The resolver priority already includes their slots.
7. **SSN2 spatial stream-network GLM is not wired.** Base_p remains the
   BRT prior. v1 will swap to ssn_glm(binomial) trained on observed
   catch data, with the CalibratedProbability interface unchanged.
8. **Open-Meteo days 8-16 are real forecast but lower spatial fidelity
   than NWS NDFD.** Open-Meteo's gridded product is global and trained
   on a different model stack; the source tag is preserved so downstream
   consumers can attenuate confidence past day 7.
9. **The flow z-score anomaly uses a single 30-day baseline.** A spring-
   flood baseline plus a summer-low baseline (seasonal pooling) would
   give better calibration in transition months. v1.
10. **The LLM narrative is grounded in the numbers but is not an
    independent check.** It can describe what the deterministic score
    chain produced, but it does not independently verify the underlying
    BRT prior or the Mohseni-Stefan projection. A separate critic agent
    would close that loop; deferred.

## 8. How to read a map

- **Geometry:** every line is one NHDPlus HR reach, drawn in
  MULTILINESTRING projection of (lon, lat).
- **Color:** matplotlib RdYlGn applied to `score_max in [0, 1]`. Red is
  low; yellow is moderate; deep green is the highest-scoring water in
  the window.
- **Line width:** 0.5 + 2.0 * score, so the eye is drawn to the
  highest-scoring reaches.
- **Title:** water + species.
- **Subtitle:** what was multiplied + the forecast window + HUC8.
- **Caption:** sources, hyperstability source, thermal-niche citation,
  forecast source, and the fraction of reaches that had a real
  `NWIS_obs` water-temp record. A caption like
  `NWIS_obs water-temp on 0/312 reaches` means the thermal_factor was
  neutral (`1.0`) on every reach in that map -- the color is then
  driven by `base_p * flow_factor(d)` only.

## 9. Where to add a real component

In order of expected impact on map fidelity:

1. **Add NorWeST gridded predictions** (West) or **EcoSHEDS NECD +
   Letcher hierarchical Bayesian model** (East) to populate
   `reach_temperature` for the >99% of reaches without an NWIS gauge.
   Today these are the limiting factor on thermal_factor coverage.
2. **Add per-day modeled water temperature** so thermal_factor varies
   across the window. Once temperature data exists, the multiplicative
   model already supports this.
3. **Wire HydroGEM into the forecast pipeline.** Pull a 576-hour series
   per USGS gauge in the HUC8, run the anomaly head, and attenuate the
   flow_factor on the reach(es) within the gauge's catchment.
4. **State stocking and regulation overlays for ID / MT / WY.** Currently
   PA-only.
5. **Replace BRT prior with SSN2 ssn_glm(binomial)** trained on real
   catch records, with stream-network spatial autocorrelation. This is
   the v1 prediction-layer swap. The `CalibratedProbability` interface
   does not change.

Until those land, the maps should be read as: "where USGS BRT models
believe this species is present in its NATIVE range, conditioned on the
species thermal preference (only where we have observed water temp), and
conditioned on the NWS precipitation forecast at the watershed centroid
for the next 14 days."
