-- Angler_AI feature-store schema (DuckDB).
--
-- All reach-level joins anchor on `comid` (NHDPlus HR). Crosswalks for
-- NHDPlusV2.1 (USGS BRT) and NECD (EcoSHEDS) attach via xwalk tables
-- maintained by the ingest layer (DR-1.2, DR-1.3).
--
-- Every persisted row carries `source` (DR-2.1) and `ingested_at` (DR-2.2).
-- Predictions carry `model_id` and `model_version` (DR-2.3).

INSTALL spatial;
LOAD spatial;

CREATE TABLE IF NOT EXISTS reaches (
    comid                BIGINT PRIMARY KEY,
    reachcode            VARCHAR,
    gnis_name            VARCHAR,
    state_fips           VARCHAR,
    county_fips          VARCHAR,
    huc8                 VARCHAR,
    huc10                VARCHAR,
    huc12                VARCHAR,
    stream_order         INTEGER,
    drainage_area_km2    DOUBLE,
    geometry             GEOMETRY,
    source               VARCHAR NOT NULL,
    ingested_at          TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS reaches_reachcode_idx ON reaches(reachcode);

-- NHDPlusV2.1 <-> HR crosswalk (DR-1.2).
CREATE TABLE IF NOT EXISTS xwalk_v2_to_hr (
    comid_v2             BIGINT NOT NULL,
    comid_hr             BIGINT NOT NULL,
    confidence           DOUBLE,
    method               VARCHAR,
    ingested_at          TIMESTAMP NOT NULL,
    PRIMARY KEY (comid_v2, comid_hr)
);

-- EcoSHEDS NECD <-> NHDPlus HR crosswalk (DR-1.3).
CREATE TABLE IF NOT EXISTS xwalk_necd_to_hr (
    necd_reach_id        VARCHAR NOT NULL,
    comid_hr             BIGINT NOT NULL,
    confidence           DOUBLE,
    method               VARCHAR,
    ingested_at          TIMESTAMP NOT NULL,
    PRIMARY KEY (necd_reach_id, comid_hr)
);

-- Modeled or observed stream temperature per reach per date.
CREATE TABLE IF NOT EXISTS reach_temperature (
    comid                BIGINT NOT NULL,
    date                 DATE NOT NULL,
    temperature_c        DOUBLE,
    uncertainty_c        DOUBLE,
    source               VARCHAR NOT NULL,
    -- Source is one of: 'NorWeST', 'EcoSHEDS_TEMP', 'PG-GNN', 'NWIS_interp'.
    ingested_at          TIMESTAMP NOT NULL,
    PRIMARY KEY (comid, date, source)
);

-- USGS NWIS streamflow snapshots. 15-min cadence is queried on-demand;
-- this table caches recent values per gauge. comid is backfilled later from
-- the NHDPlus join; nullable until then.
CREATE TABLE IF NOT EXISTS reach_flow (
    comid                BIGINT,
    gauge_id             VARCHAR NOT NULL,
    ts                   TIMESTAMP NOT NULL,
    discharge_cfs        DOUBLE,
    gauge_height_ft      DOUBLE,
    source               VARCHAR NOT NULL DEFAULT 'USGS_NWIS',
    ingested_at          TIMESTAMP NOT NULL,
    PRIMARY KEY (gauge_id, ts)
);

-- EPA Water Quality Portal discrete WQ samples joined to nearest reach.
-- comid is backfilled from a later spatial join; nullable until then.
CREATE TABLE IF NOT EXISTS reach_wq (
    comid                BIGINT,
    sample_date          DATE,
    parameter            VARCHAR NOT NULL,
    value                DOUBLE,
    unit                 VARCHAR,
    source               VARCHAR NOT NULL DEFAULT 'EPA_WQP',
    org_id               VARCHAR,
    ingested_at          TIMESTAMP NOT NULL
);

-- EPA ATTAINS impaired-waters status joined to NHDPlus catchment.
CREATE TABLE IF NOT EXISTS attains_status (
    comid                BIGINT NOT NULL,
    cycle_year           INTEGER NOT NULL,
    status               VARCHAR,
    parameter            VARCHAR,
    state_305b_url       VARCHAR,
    source               VARCHAR NOT NULL DEFAULT 'EPA_ATTAINS',
    ingested_at          TIMESTAMP NOT NULL
);

-- USGS NAS (Nonindigenous Aquatic Species) occurrence summary, per HUC8.
-- Used as a presence-only fallback prior for non-native species in HUC8s
-- where USGS BRT v2.0 has no row (BRT v2.0 only models native distributions).
-- PK is (huc8, scientific_name) - the summary collapses per-record history
-- into a (first, last, n_records) tuple. Per-record history is v1 (Tier 2).
-- Establishment filter applied at ingest: only Established/Stocked/Collected
-- records contribute to the summary; Eradicated/Failed records are dropped.
CREATE TABLE IF NOT EXISTS nas_occurrences (
    huc8                 VARCHAR NOT NULL,
    scientific_name      VARCHAR NOT NULL,
    common_name          VARCHAR,
    status               VARCHAR,
    -- 'Established' | 'Stocked' | 'Collected' (subset of NAS status vocab
    -- after our establishment filter). Most-recent-record wins.
    year_first_observed  INTEGER,
    year_last_observed   INTEGER,
    n_records            INTEGER,
    source               VARCHAR NOT NULL DEFAULT 'USGS_NAS_V1.0',
    ingested_at          TIMESTAMP NOT NULL,
    PRIMARY KEY (huc8, scientific_name)
);
CREATE INDEX IF NOT EXISTS nas_occurrences_species_idx ON nas_occurrences(scientific_name);

-- Climatological water-temperature baseline per reach per month.
-- NorWeST (USFS RMRS, 1993-2011 mean August) and EcoSHEDS_TEMP (East,
-- hierarchical Bayesian) populate this. NOT a daily forecast - this is
-- the spatial anchor that water_temp_model.py uses as `T_water_baseline`
-- when projecting per-day water temps off the air-temp forecast.
-- Distinct from reach_temperature (daily/observed values).
CREATE TABLE IF NOT EXISTS reach_temp_baseline (
    comid                BIGINT NOT NULL,
    month                INTEGER NOT NULL,
    -- 1-12; NorWeST baseline is month=8 (August mean) at v0
    baseline_temp_c      DOUBLE NOT NULL,
    source               VARCHAR NOT NULL,
    -- 'NorWeST' | 'EcoSHEDS_TEMP' | future
    ingested_at          TIMESTAMP NOT NULL,
    PRIMARY KEY (comid, month, source)
);
CREATE INDEX IF NOT EXISTS reach_temp_baseline_comid_idx ON reach_temp_baseline(comid);

-- USGS BRT v2.0 fluvial fish SDM priors. Per species per COMID.
CREATE TABLE IF NOT EXISTS brt_priors (
    comid                BIGINT NOT NULL,
    species              VARCHAR NOT NULL,
    probability          DOUBLE NOT NULL,
    auc                  DOUBLE,
    model_version        VARCHAR NOT NULL DEFAULT 'USGS_BRT_V2.0',
    ingested_at          TIMESTAMP NOT NULL,
    PRIMARY KEY (comid, species)
);

-- USGS BRT v2.0 species metadata loaded from fish_list_v2_0.csv. Used for
-- common-name <-> scientific-name lookup in the species/map CLI.
CREATE TABLE IF NOT EXISTS brt_species (
    itis_tsn             BIGINT,
    scientific_name      VARCHAR PRIMARY KEY,
    common_name          VARCHAR,
    presences            INTEGER,
    absences             INTEGER,
    prevalence           DOUBLE,
    taxonomic_order      VARCHAR,
    family               VARCHAR,
    ingested_at          TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS brt_species_common_idx ON brt_species(common_name);

-- State stocking events. PA PFBC discovery seeds v0; VA + ID at M2.
-- comid is backfilled from the NHDPlus join in a follow-up pass; nullable here.
CREATE TABLE IF NOT EXISTS stocking_events (
    comid                BIGINT,
    state                VARCHAR NOT NULL,
    event_date           DATE,
    species              VARCHAR NOT NULL,
    count                INTEGER,
    water_body_name      VARCHAR,
    source               VARCHAR NOT NULL,
    ingested_at          TIMESTAMP NOT NULL
);

-- State fishing regulations. Keyed on state + water + species + date window.
CREATE TABLE IF NOT EXISTS regulations (
    state                VARCHAR NOT NULL,
    water_body_id        VARCHAR,
    comid                BIGINT,
    species              VARCHAR,
    season_start         DATE,
    season_end           DATE,
    gear_restrictions    VARCHAR,
    bag_limit            INTEGER,
    license_required     BOOLEAN,
    special_regulation   VARCHAR,
    source_url           VARCHAR,
    source               VARCHAR NOT NULL,
    ingested_at          TIMESTAMP NOT NULL
);

-- Sensitive-species policy table (CR-1). Loaded from the seed CSV in
-- src/angler_ai/ethics/data/sensitive_species_seed.csv at first run.
CREATE TABLE IF NOT EXISTS sensitive_species (
    species_id           VARCHAR PRIMARY KEY,
    common_name          VARCHAR NOT NULL,
    scientific_name      VARCHAR NOT NULL,
    status               VARCHAR NOT NULL,
    -- 'ESA-threatened', 'ESA-endangered', 'ESA-candidate', 'state-soc'
    suppress_level       VARCHAR NOT NULL,
    -- 'reach', 'huc12', 'huc10', 'huc8'
    rationale            VARCHAR,
    source_doc           VARCHAR,
    -- e.g. 'CR-1.1', 'CR-1.2'
    ingested_at          TIMESTAMP NOT NULL
);

-- Tribal-waters mask. Loaded from public USGS / state datasets only (CR-2.3).
CREATE TABLE IF NOT EXISTS tribal_mask (
    region_id            VARCHAR PRIMARY KEY,
    tribe_name           VARCHAR NOT NULL,
    geometry             GEOMETRY,
    redirect_url         VARCHAR,
    source               VARCHAR NOT NULL,
    ingested_at          TIMESTAMP NOT NULL
);

-- Model selection decisions log (NFR-6.2). Append-only.
CREATE TABLE IF NOT EXISTS model_selection_log (
    ts                   TIMESTAMP NOT NULL,
    problem_set          VARCHAR NOT NULL,
    chosen_model_id      VARCHAR NOT NULL,
    chosen_quant         VARCHAR NOT NULL,
    candidates_considered INTEGER,
    hardware_budget_mb   INTEGER,
    reason               VARCHAR
);

-- Calibration adjustments log (NFR-6.3). Append-only.
CREATE TABLE IF NOT EXISTS calibration_log (
    ts                   TIMESTAMP NOT NULL,
    comid                BIGINT,
    species              VARCHAR,
    raw_p                DOUBLE,
    calibrated_p         DOUBLE,
    beta_used            DOUBLE,
    cpue_weight          DOUBLE,
    source               VARCHAR
);
