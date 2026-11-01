# Exploratory Data Analysis (EDA) Engineer Handoff Guide

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Phase:** Phase 2 — Exploratory Data Analysis & Feature Verification  
**Target Role:** Senior EDA Engineer  
**Handoff Author:** Exclusive Data Engineer  
**Date:** 2026-11-01  
**Status:** **READY FOR IMMEDIATE EDA COMMENCEMENT**  

---

## 1. Project Objective & Data Engineering Deliverables

The primary objective of Phase 2 is to predict **high-impact supply chain logistics bottlenecks** and forecast **multi-step vessel arrival traffic** across the global maritime trade network. 

The Data Engineering team has completed the acquisition, cleansing, profiling, feature engineering, and temporal/graph packaging stages. The downstream EDA Engineer does **not** need to reacquire or reprocess any datasets.

### Primary Data Engineering Deliverables Available to EDA:
1. **Primary Unified Daily Panel:** `DL/data/staging/supplementary_acquisition/processed/unified_temporal/port_daily_unified.parquet` (5,280,205 rows, 91 columns).
2. **Temporal Sequence Index:** `DL/data/staging/supplementary_acquisition/processed/temporal_sequences/sequence_metadata.parquet` (5,181,085 sequences, $L=30, H=7$).
3. **Partition Manifest:** `DL/data/staging/supplementary_acquisition/processed/temporal_sequences/temporal_split_manifest.csv` (Train/Val/Test boundaries).
4. **North American Multimodal Hinterland Network:** `DL/data/staging/supplementary_acquisition/processed/graph_ready/north_american_graph_*.parquet` (251,646 nodes, 303,016 edges).
5. **Caribbean Regional Maritime Graph:** `DL/data/staging/supplementary_acquisition/processed/graph_ready/caribbean_graph_*.parquet` (528 nodes, 9,969 directed lanes).
6. **Weekly Congestion Benchmark:** `DL/data/staging/supplementary_acquisition/processed/adb_port_traffic_weekly.parquet` (3,990 rows, 89 columns).
7. **Master 166-Feature Dictionary:** `DL/data engineer/manifests/final_feature_dictionary.csv`.

---

## 2. Primary Dataset Architecture (`port_daily_unified.parquet`)

### 2.1 File Location & Structural Dimensions
* **Exact Path:** `DL/data/staging/supplementary_acquisition/processed/unified_temporal/port_daily_unified.parquet`
* **File Format:** Apache Parquet (Snappy compression, 97.68 MB on disk)
* **Total Rows:** **5,280,205 records**
* **Total Columns:** **91 columns**
* **Entity Coverage:** Exactly **2,065 commercial seaports** across **180 sovereign nations**
* **Temporal Extent:** **2019-01-01 to 2025-12-31** (7 contiguous calendar years)
* **Frequency:** Daily ($t$)
* **Panel Balance:** **100% Balanced.** Every port contains exactly **2,557 contiguous calendar days**.
* **Primary Key:** `(portwatch_id, date_utc)` (Verified: 0 duplicate records).

### 2.2 Semantic Taxonomy of Variables
The EDA engineer must strictly distinguish between the four conceptual layers of variables in this table:

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Layer 1: Observed Operational Measurements (PortWatch AIS Ground Truth)│
  │ • portcalls: Total commercial vessel arrivals inside port limits       │
  │ • portcalls_container, tanker, dry_bulk, general_cargo, roro           │
  │ • import_cargo, export_cargo, import_container, export_container       │
  │   (Physical count / tonnage observables; non-negative; 0 = calm day)   │
  ├────────────────────────────────────────────────────────────────────────┤
  │ Layer 2: Derived Historical Activity & Structural Signals              │
  │ • total_cargo_tonnage: import_cargo + export_cargo                     │
  │ • pw_calls_rolling_7d_mean, 14d_mean, 30d_mean: Trailing backward-only│
  │ • bts_daily_expected_vessel_calls: Design arrival baseline             │
  │ • bts_capacity_pressure_ratio_14d: Rolling calls / daily expected calls│
  ├────────────────────────────────────────────────────────────────────────┤
  │ Layer 3: Contextual Exogenous Shocks & Proxies                         │
  │ • noaa_storm_event_count_50km, marine_wind_event_count, flood_flag     │
  │   (Spatial proximity exposure within 50km/100km radius; US only)       │
  │ • emdat_active_disaster_count, macro_disaster_active (Country level)   │
  │ • era5_wind_speed_mean_ms, t2m_mean_c, msl_mean_hpa, precipitation    │
  │ • era5_swh_mean_m, era5_wave_proxy_type (Conditional offshore proxy)   │
  ├────────────────────────────────────────────────────────────────────────┤
  │ Layer 4: Supervised Target Horizon (Lookahead Labels)                  │
  │ • target_portcalls_7d_future_sum: sum(calls_{t+1} ... calls_{t+7})     │
  │ • target_container_calls_7d_future_sum: sum(cont_{t+1} ... cont_{t+7}) │
  │ • target_cargo_tonnage_7d_future_sum: sum(cargo_{t+1} ... cargo_{t+7})│
  │ • target_portcalls_lead_1d through lead_7d: Multi-horizon lead vector  │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Supervised Targets & Temporal Sequences

### 3.1 Forecasting Targets
* **Primary Target:** `target_portcalls_7d_future_sum`
  - Formula: $\sum_{k=1}^7 \text{portcalls}(i, t+k)$
  - Physical Meaning: Total cumulative vessel traffic arriving at port $i$ over the upcoming 7-day planning horizon.
  - Distribution: Heavy right-skewed (Mean = 16.9, Median = 5.0, P90 = 39.0, Max = 1,024).
* **Multi-Horizon Lead Vector:** `target_portcalls_lead_1d` to `lead_7d`
  - Allows multi-step ahead trajectory forecasting ($t+1, t+2, \dots, t+7$).
* **Null Boundary Behavior:**
  - Because the panel terminates on `2025-12-31`, target values for the terminal 7 days ($t \in [2025\text{-}12\text{-}25, 2025\text{-}12\text{-}31]$) are strictly populated as **`NaN`** ($2,065 \times 7 = 14,455 \text{ records}$).
  - All earlier dates have 100% complete target labels.

### 3.2 Sequence Metadata & Split Boundaries
* **Exact Path:** `DL/data/staging/supplementary_acquisition/processed/temporal_sequences/sequence_metadata.parquet`
* **Total Quarantined Sequences:** **5,181,085 instances**
* **Window Specifications:** Lookback $L = 30$ days ($[t-29, t]$), Horizon $H = 7$ days ($[t+1, t+7]$).
* **Partition Split Rules & Date Spans:**
  - **Train Partition (71.34%, 3,696,350 sequences):**
    - Reference date $t \in [2019\text{-}01\text{-}30, 2023\text{-}12\text{-}24]$.
    - Input window $[t-29, t]$ spans `2019-01-01` to `2023-12-24`.
    - Target window $[t+1, t+7]$ ends strictly on or before **`2023-12-31`**.
  - **Validation Partition (14.35%, 743,400 sequences):**
    - Reference date $t \in [2023\text{-}12\text{-}31, 2024\text{-}12\text{-}24]$.
    - Target window $[t+1, t+7]$ strictly contained within **`2024-01-01` to `2024-12-31`**.
  - **Test Partition (14.31%, 741,335 sequences):**
    - Reference date $t \in [2024\text{-}12\text{-}31, 2025\text{-}12\text{-}24]$.
    - Target window $[t+1, t+7]$ strictly contained within **`2025-01-01` to `2025-12-31`**.
* **Pruned Boundary Crossing Instances:** Exactly **24,780 candidate sequences** that crossed annual partition boundaries were pruned to guarantee strict temporal quarantine.

---

## 4. Supporting & Benchmark Datasets

| Dataset | Exact Path | Rows | Cols | Key Identifiers | Temporal Span | Role in EDA & Modeling |
| :--- | :--- | :---: | :---: | :--- | :---: | :--- |
| **ADB Weekly Port Traffic** | `DL/data/staging/supplementary_acquisition/processed/adb_port_traffic_weekly.parquet` | 3,990 | 89 | `port_name`, `iso_year_week` | 2019–2022 (Weekly) | Analyze anchorage waiting time vs. berth service time across 19 major hubs. |
| **Indian Major Ports Dwell** | `DL/data/staging/supplementary_acquisition/processed/indian_port_dwell_monthly.parquet` | 363 | 20 | `portwatch_id`, `year_month` | 2021–2023 (Monthly) | Compare cargo land/customs dwell against vessel activity. |
| **AIS Monthly Vessel Dwell**| `DL/data/raw/ais_dwell/monthly_dwell_2015_2025.csv` | 660 | 13 | `Port`, `YearMonth` | 2015–2025 (Monthly) | Macro validation benchmark for 5 US port complexes. |
| **BTS Capacity Features** | `DL/data/staging/supplementary_acquisition/processed/bts/bts_port_capacity_features.parquet` | 309 | 15 | `canonical_portwatch_id`, `reporting_year` | 2016–2023 (Annual) | Physical baseline comparisons and capacity limits. |
| **NOAA Daily Shocks** | `DL/data/staging/supplementary_acquisition/processed/noaa/noaa_port_daily_shocks.parquet` | 143,192 | 14 | `portwatch_id`, `date_utc` | 2019–2025 (Daily) | Convective storm damage and marine gale frequencies for 56 US ports. |
| **EM-DAT Disaster Shocks** | `DL/data/staging/supplementary_acquisition/processed/emdat/emdat_country_daily_shocks.parquet` | 488,387 | 12 | `country_iso3`, `date_utc` | 2019–2025 (Daily) | Macro disaster event counts and economic impact tracking. |
| **Canonical Port Registry** | `DL/data engineer/registry/canonical_port_registry.csv` | 2,065 | 32 | `portwatch_id` | Static | Authoritative metadata (UN/LOCODE, lat/lon, country ISO). |

---

## 5. Spatial Graph Networks

### 5.1 North American Multimodal Hinterland Network
* **Node Table:** `DL/data/staging/supplementary_acquisition/processed/graph_ready/north_american_graph_nodes.parquet` (251,646 nodes, 3.04 MB)
  - 250,204 rail network junctions (NTAD Rail).
  - 1,401 freight intermodal transfer terminals (NTAD Intermodal).
  - 41 canonical deepwater port gateways (Canonical Registry).
* **Edge Table:** `DL/data/staging/supplementary_acquisition/processed/graph_ready/north_american_graph_edges.parquet` (303,016 edges, 11.49 MB)
  - 302,774 physical freight rail line segments.
  - 242 intermodal drayage connectors ($\le 50\text{ km}$ geodesic proximity).
* **EDA Focus:** Examine network diameter, degree distribution, Class I railroad share (BNSF, UP, CSX, NS), and intermodal connectivity choke points.

### 5.2 Caribbean Regional Maritime Network
* **Node Table:** `DL/data/staging/supplementary_acquisition/processed/graph_ready/caribbean_graph_nodes.parquet` (528 nodes, 31.6 KB)
  - 528 commercial seaports spanning 38 territories.
  - 18 canonical PortWatch global gateways.
* **Edge Table:** `DL/data/staging/supplementary_acquisition/processed/graph_ready/caribbean_graph_edges.parquet` (9,969 edges, 151.7 KB)
  - 9,969 empirical directed maritime shipping lanes.
  - 1,179 inter-canonical gateway corridors.
  - Total voyage traffic broken down by vessel type (Container, Tanker, Bulk, Feeder).
* **EDA Focus:** Transshipment hub centrality (Kingston, Freeport, Caucedo), voyage volume skewness, and feeder-to-hub connectivity.

---

## 6. Known Data-Quality Issues, Discrepancies & Epistemic Boundaries

The EDA engineer must account for the following data characteristics during exploratory analysis:

1. **Spatial Coverage Sparsity in Contextual Features:**
   - `has_era5_coverage = 1` for **5 ports** (0.24% of rows). 99.76% of rows have nulls.
   - `has_noaa_coverage = 1` for **56 ports** (2.71% of rows). Calm days are 0; non-US ports are `NaN`.
   - `has_bts_coverage = 1` for **36 ports** (1.74% of rows).
   - `has_emdat_coverage = 1` for **150 countries** (97.24% of rows). Island territories (e.g., Aruba, Bermuda) are `NaN`.
   - *Precaution:* Never compute global means without conditioning on the coverage flag.
2. **BTS Port Intersection Count (36 in Unified vs. 39 in BTS Table):**
   - The BTS capacity table contains 39 deepwater ports. However, exactly 36 intersect with `port_daily_unified.parquet`. Three ports—Valdez (`port1347`), Palm Beach (`port880`), and Lake Charles (`port613`)—are in the registry but have no active AIS records in PortWatch's daily panel.
3. **BTS Capacity Baseline Forward-Fill (2024–2025):**
   - BTS historical manifests are published through 2023. For 2024 and 2025, the 2023 structural capacity was forward-filled (`is_bts_forward_filled = 1`). This allows leak-free validation and testing against known baseline capacities, but terminal expansions in 2024–2025 are not reflected.
4. **Conditional Wave Proxies in ERA5:**
   - Wave grid cells for Savannah (57.6 km offshore), Houston (53.2 km), and NY/NJ (77.8 km) exceed the 40 km coastal threshold and are classified as `CONDITIONAL_OFFSHORE_PROXY`. They measure open-ocean approach swell rather than inner-berth water disturbance.
5. **Dwell Ground-Truth Reality (Prohibition of Daily Interpolation):**
   - Continuous daily vessel dwell does **not** exist in any available public dataset.
   - Monthly AIS dwell (`MeanDwellDays`) and weekly ADB turnaround times must be analyzed at their native frequency. **Do not attempt to interpolate monthly dwell into daily synthetic values.**
6. **EM-DAT Economic Damages Missingness (MNAR):**
   - Unreported disaster damages and fatalities in EM-DAT are preserved strictly as **`NaN`**. They must **never** be imputed as 0, as doing so treats un-surveyed catastrophic events as having zero economic impact.

---

## 7. Leakage Risks & Methodological Precautions

Do not describe the dataset as fully leakage-free without understanding how boundaries are enforced:

1. **Target Boundary Quarantine:**
   - Target values represent future sums ($t+1$ to $t+7$). In all exploratory analyses of features at time $t$, ensure `target_*` columns are **strictly excluded from predictor feature sets**.
2. **Feature Scaler Leakage:**
   - When computing normalization statistics (mean, variance, min, max, quantile bounds), compute them **strictly on the Train partition ($t \le \text{2023-12-24}$)**. Applying global scalers computed across 2019–2025 will leak future distribution shifts into training.
3. **Lookahead in Rolling Averages:**
   - Trailing rolling features (`pw_calls_rolling_7d_mean`, `14d_mean`, `30d_mean`) are strictly backward-looking ($[t-k+1, t]$). They contain zero forward leakage. However, when creating new rolling features during EDA, ensure `center=True` is **never** used.
4. **Transition Window Quarantine:**
   - When validating sequence models, adhere to the 24,780 excluded boundary-crossing sequences documented in `temporal_split_manifest.csv` to ensure no training sequence target overlaps with a validation input window.

---

## 8. Recommended Order of EDA Activities

The EDA Engineer should execute exploratory analyses in the following sequential order:

```
  Phase 1: Univariate Target & Throughput Distribution Analysis
  ├── 1.1 Compute target skewness, kurtosis, and zero-inflation proportions.
  ├── 1.2 Evaluate log1p and Box-Cox transformations for target_portcalls_7d_future_sum.
  └── 1.3 Compare arrival distributions across vessel types (Container, Tanker, Bulk).

  Phase 2: Temporal Stability, Seasonality & COVID Shock Drift Analysis
  ├── 2.1 Plot 7-year global time series (2019–2025) of total vessel calls.
  ├── 2.2 Quantify temporal distribution drift: Train (2019-2023) vs Val (2024) vs Test (2025).
  ├── 2.3 Decompose weekly and annual cyclicities using day_of_week and day_of_year.
  └── 2.4 Profile the 2021-2022 global supply chain bottleneck crisis period.

  Phase 3: Exogenous Shock Exposure & Elasticity Analysis
  ├── 3.1 Bivariate analysis: NOAA storm exposure (within 50km) vs. localized arrival drops.
  ├── 3.2 EM-DAT active disaster days vs. national-level port call anomalies.
  ├── 3.3 ERA5 wave height / wind speed correlation with vessel arrival volatility.
  └── 3.4 Capacity Pressure Ratio (bts_capacity_pressure_ratio_14d) threshold audit.

  Phase 4: Spatial Network & Graph Topology Exploration
  ├── 4.1 Caribbean graph: in-degree vs. out-degree centrality of transshipment hubs.
  ├── 4.2 Inter-canonical corridor voyage frequency distributions.
  └── 4.3 North American rail hinterland degree distribution and intermodal transfer distances.

  Phase 5: Cross-Resolution Benchmark Calibration
  ├── 5.1 Monthly AIS dwell (MeanDwellDays) correlation with monthly aggregated PortWatch calls.
  └── 5.2 Weekly ADB waiting time vs. service time elasticity across vessel size classes.
```

---

## 9. Quick-Start Ingestion Code Snippet for EDA

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Primary Dataset (Lean Column Selection for Fast EDA)
DATA_PATH = "DL/data/staging/supplementary_acquisition/processed/unified_temporal/port_daily_unified.parquet"
META_PATH = "DL/data/staging/supplementary_acquisition/processed/temporal_sequences/sequence_metadata.parquet"

eda_cols = [
    'portwatch_id', 'portwatch_name', 'country_name', 'country_iso3', 'date_utc', 'year',
    'portcalls', 'portcalls_container', 'total_cargo_tonnage',
    'pw_calls_rolling_7d_mean', 'pw_calls_rolling_14d_mean',
    'target_portcalls_7d_future_sum', 'target_container_calls_7d_future_sum',
    'has_noaa_coverage', 'noaa_storm_event_count_50km',
    'has_emdat_coverage', 'emdat_active_disaster_count',
    'has_bts_coverage', 'bts_capacity_pressure_ratio_14d'
]

print("Loading unified panel for EDA...")
df_panel = pd.read_parquet(DATA_PATH, columns=eda_cols)
print(f"Loaded: {len(df_panel):,} rows across {df_panel['portwatch_id'].nunique():,} ports.")

# 2. Load Sequence Split Metadata
df_meta = pd.read_parquet(META_PATH)
print("Sequence Partitions:")
print(df_meta['split_tag'].value_counts())

# 3. Example EDA: Profile Target Statistics across Train, Val, Test
train_ports = df_meta[df_meta['split_tag'] == 'train']['reference_date'].unique()
val_ports = df_meta[df_meta['split_tag'] == 'val']['reference_date'].unique()
test_ports = df_meta[df_meta['split_tag'] == 'test']['reference_date'].unique()

print("\nTrain Target Mean:", df_panel[df_panel['date_utc'].isin(train_ports)]['target_portcalls_7d_future_sum'].mean())
print("Val Target Mean:  ", df_panel[df_panel['date_utc'].isin(val_ports)]['target_portcalls_7d_future_sum'].mean())
print("Test Target Mean: ", df_panel[df_panel['date_utc'].isin(test_ports)]['target_portcalls_7d_future_sum'].mean())
```

---

## 10. Dataset Access, Distribution, Placement & Integrity Verification Guide

To balance developer ergonomics, repository performance, and GitHub's strict file size boundaries (50 MB warning, 100 MB hard reject), the project uses a three-tier dataset distribution architecture:

```
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ Tier 1: Direct Git Tracking (< 19 MB total)                                 │
  │ • All pipeline source code (27 scripts)                                     │
  │ • All 15 data engineering manifests and crosswalks (.csv)                  │
  │ • Canonical port registry & reference dictionaries (.csv)                  │
  │ • All 17 detailed data profiles, lineage docs, and handoff guides (.md)    │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ Tier 2: Git LFS Tracking (Model-Ready Processed Parquet Tables, ~165 MB)    │
  │ • Primary Unified Daily Panel: port_daily_unified.parquet (97.68 MB)        │
  │ • Temporal Sequence Index: sequence_metadata.parquet (50.77 MB)            │
  │ • Graph packages: north_american_graph_*.parquet (14.53 MB total)           │
  │ • Caribbean graph packages: caribbean_graph_*.parquet (0.17 MB total)       │
  │ • Supplementary benchmark parquets (ADB, Indian Dwell, BTS, NOAA, EM-DAT)   │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ Tier 3: External Object Storage (Oversized Raw Data Archives, ~950 MB total)│
  │ • IMF PortWatch Daily Vessel AIS: Daily_Ports_Data.csv (618.96 MB)          │
  │ • Caribbean Port Calls & Trajectories: portcalls.csv (197.57 MB)            │
  │ • NTAD Rail Lines: NTAD_North_American_Rail_Network_Lines.csv (41.63 MB)   │
  │ • NOAA Storm Events 2019-2025 gzip archives (7 files, ~80 MB)               │
  │ • EM-DAT Global Disaster Archive: 260430_emdat_archive.xlsx (7.76 MB)       │
  └─────────────────────────────────────────────────────────────────────────────┘
```

### 10.1 Fresh Clone Quickstart

#### Path A: Standard Git LFS Clone (Recommended)
If you have `git-lfs` installed (standard with GitHub Desktop, Homebrew, and apt):
```bash
# 1. Clone repository
git clone https://github.com/AbdulRahman1807/DS2.git
cd DS2

# 2. Pull all Phase 2 LFS tracked datasets (~165 MB)
git lfs pull

# 3. Verify datasets immediately
python3 -c "
import os, pyarrow.parquet as pq
p = 'DL/data/staging/supplementary_acquisition/processed/unified_temporal/port_daily_unified.parquet'
assert os.path.exists(p), 'Unified panel missing!'
meta = pq.read_metadata(p)
print(f'SUCCESS: Verified {meta.num_rows:,} rows and {meta.num_columns} columns in unified panel.')
"
```

#### Path B: External Mirror Download (Non-LFS Environments)
If your environment has Git LFS disabled or bandwidth restricted, all processed model-ready Parquet tables are mirrored as a single compressed archive on external storage:
* **Mirror Download URL:** `https://storage.googleapis.com/supply-chain-bottlenecks-phase2/phase2_processed_model_ready.tar.gz` (or Zenodo DOI: `10.5281/zenodo.XXXXXXX`)
* **Placement Instructions:**
```bash
# Download and unpack into the processed data directory:
mkdir -p DL/data/staging/supplementary_acquisition/processed
curl -L -o /tmp/phase2_processed_model_ready.tar.gz <DOWNLOAD_URL>
tar -xzf /tmp/phase2_processed_model_ready.tar.gz -C DL/data/staging/supplementary_acquisition/processed/
```

### 10.2 Authoritative Dataset Inventory & SHA-256 Checksums

Every model-ready dataset consumed by the EDA Engineer has been fingerprinted with SHA-256:

| Dataset Name | Relative Destination Path | Size (MB) | Rows | Cols | SHA-256 Fingerprint |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Primary Unified Daily Panel** | `DL/data/staging/supplementary_acquisition/processed/unified_temporal/port_daily_unified.parquet` | 97.68 | 5,280,205 | 91 | `7ddaab022c93c98239ff99c0239768965c6cc99f10e7ef7515f96662d8521d86` |
| **Sequence Metadata Index** | `DL/data/staging/supplementary_acquisition/processed/temporal_sequences/sequence_metadata.parquet` | 50.77 | 5,181,085 | 16 | `d6e2a09f0101f7a91bea85e8f5bb785b090eda8a914804bd6464eaaad923aee0` |
| **Temporal Split Manifest** | `DL/data/staging/supplementary_acquisition/processed/temporal_sequences/temporal_split_manifest.csv` | 0.00 | 3 | 18 | `2969476e1ecb669e0eab4b821e84aeadb67fb00b6c16494015d1fdabad434dff` |
| **NA Rail Hinterland Nodes** | `DL/data/staging/supplementary_acquisition/processed/graph_ready/north_american_graph_nodes.parquet` | 3.04 | 251,646 | 12 | `8abb022e94ecbcc16121175052f879184898aeff4a752396259ebbd342fefa71` |
| **NA Rail Hinterland Edges** | `DL/data/staging/supplementary_acquisition/processed/graph_ready/north_american_graph_edges.parquet` | 11.49 | 303,016 | 15 | `db4a383b7ba42365f1ae4b711ec9ed1dfb7a8a27f3615f1e78277824883c0ebd` |
| **Caribbean Maritime Nodes** | `DL/data/staging/supplementary_acquisition/processed/graph_ready/caribbean_graph_nodes.parquet` | 0.03 | 528 | 15 | `2e7f292ba8971b9883aaa066ef4c658f65cf90c02b4f6a6617f13ed55afbbccf` |
| **Caribbean Maritime Edges** | `DL/data/staging/supplementary_acquisition/processed/graph_ready/caribbean_graph_edges.parquet` | 0.14 | 9,969 | 17 | `20ac327c486ac662a088f55edccc8ca79d92c168cbb59dde38c09b5aeb3f0c89` |
| **ADB Weekly Port Traffic (Parquet)** | `DL/data/staging/supplementary_acquisition/processed/adb_port_traffic_weekly.parquet` | 0.48 | 3,990 | 89 | `96f2facf1f7b899164a2e68add5a04123ccb6b19f6ce0314f4174806990f712b` |
| **ADB Weekly Port Traffic (CSV)** | `DL/data/staging/supplementary_acquisition/processed/adb_port_traffic_weekly.csv` | 2.06 | 3,990 | 89 | `74a178aa41d3e3364513b48e6e68ebd0490ec30434e1cb53e1ca33886a9845f1` |
| **Indian Major Ports Dwell (Parquet)** | `DL/data/staging/supplementary_acquisition/processed/indian_port_dwell_monthly.parquet` | 0.02 | 363 | 20 | `0acf27f8bc79c44eee26f78ac9bb954ae91a07125c4bfd8d274ca5ceb4f82bbb` |
| **Indian Major Ports Dwell (CSV)** | `DL/data/staging/supplementary_acquisition/processed/indian_port_dwell_monthly.csv` | 0.06 | 363 | 20 | `de1df09a6b3334570237d9f58fb9c7dbac7c0e80c21419e73cd6dd624ffdc5c4` |
| **BTS Port Capacity Benchmark** | `DL/data/staging/supplementary_acquisition/processed/bts/bts_port_capacity_features.parquet` | 0.03 | 309 | 15 | `3537d7c65cecb520b18e1b47c15f118ddcb6aef5325e4dae2f8ee5365cd6a901` |
| **NOAA Daily Port Weather Shocks** | `DL/data/staging/supplementary_acquisition/processed/noaa/noaa_port_daily_shocks.parquet` | 0.27 | 143,192 | 14 | `cd0d26fd967e9fa2a0e9dbf9497c8efc0f688680ae713112ed9db628ee88c7ae` |
| **EM-DAT Disaster Shocks** | `DL/data/staging/supplementary_acquisition/processed/emdat/emdat_country_daily_shocks.parquet` | 0.56 | 488,387 | 12 | `814909cd562c4721727fd57e835b859bace7c4d8e92e24049c784d1bb91b10cb` |
| **Canonical Port Registry** | `DL/data engineer/registry/canonical_port_registry.csv` | 0.50 | 2,065 | 32 | `292d959c8000c6598483abac5db43476919833cd0b9f5957e1c18ce5bb51ebaa` |
| **Spatial Feature Governance** | `DL/data engineer/registry/spatial_feature_governance.parquet` | 0.16 | 2,065 | 31 | `df75823d98bda2a89cf293558b870ecc15b793a87b370e82bc4b0763163cf568` |
| **Master Feature Dictionary** | `DL/data engineer/manifests/final_feature_dictionary.csv` | 0.03 | 166 | 7 | `c1d4295e473cb058c005bd4dc79160718e58929341f9e44b8dc9e07091a87625` |

### 10.3 Automated Self-Verification Code

Run this script to automatically verify that all datasets are present and uncorrupted:

```python
import hashlib, os, sys
from pathlib import Path

# Repo root auto-detection
PROJECT_ROOT = Path(__file__).resolve().parents[2] if "eda engineer" in str(Path(__file__)) else Path.cwd()

EXPECTED = {
    "DL/data/staging/supplementary_acquisition/processed/unified_temporal/port_daily_unified.parquet": "7ddaab022c93c98239ff99c0239768965c6cc99f10e7ef7515f96662d8521d86",
    "DL/data/staging/supplementary_acquisition/processed/temporal_sequences/sequence_metadata.parquet": "d6e2a09f0101f7a91bea85e8f5bb785b090eda8a914804bd6464eaaad923aee0",
    "DL/data/staging/supplementary_acquisition/processed/temporal_sequences/temporal_split_manifest.csv": "2969476e1ecb669e0eab4b821e84aeadb67fb00b6c16494015d1fdabad434dff",
    "DL/data/staging/supplementary_acquisition/processed/graph_ready/north_american_graph_nodes.parquet": "8abb022e94ecbcc16121175052f879184898aeff4a752396259ebbd342fefa71",
    "DL/data/staging/supplementary_acquisition/processed/graph_ready/north_american_graph_edges.parquet": "db4a383b7ba42365f1ae4b711ec9ed1dfb7a8a27f3615f1e78277824883c0ebd",
    "DL/data/staging/supplementary_acquisition/processed/graph_ready/caribbean_graph_nodes.parquet": "2e7f292ba8971b9883aaa066ef4c658f65cf90c02b4f6a6617f13ed55afbbccf",
    "DL/data/staging/supplementary_acquisition/processed/graph_ready/caribbean_graph_edges.parquet": "20ac327c486ac662a088f55edccc8ca79d92c168cbb59dde38c09b5aeb3f0c89",
    "DL/data/staging/supplementary_acquisition/processed/adb_port_traffic_weekly.parquet": "96f2facf1f7b899164a2e68add5a04123ccb6b19f6ce0314f4174806990f712b",
    "DL/data/staging/supplementary_acquisition/processed/indian_port_dwell_monthly.parquet": "0acf27f8bc79c44eee26f78ac9bb954ae91a07125c4bfd8d274ca5ceb4f82bbb",
    "DL/data/staging/supplementary_acquisition/processed/bts/bts_port_capacity_features.parquet": "3537d7c65cecb520b18e1b47c15f118ddcb6aef5325e4dae2f8ee5365cd6a901",
    "DL/data/staging/supplementary_acquisition/processed/noaa/noaa_port_daily_shocks.parquet": "cd0d26fd967e9fa2a0e9dbf9497c8efc0f688680ae713112ed9db628ee88c7ae",
    "DL/data/staging/supplementary_acquisition/processed/emdat/emdat_country_daily_shocks.parquet": "814909cd562c4721727fd57e835b859bace7c4d8e92e24049c784d1bb91b10cb",
    "DL/data engineer/registry/canonical_port_registry.csv": "292d959c8000c6598483abac5db43476919833cd0b9f5957e1c18ce5bb51ebaa",
    "DL/data engineer/manifests/final_feature_dictionary.csv": "c1d4295e473cb058c005bd4dc79160718e58929341f9e44b8dc9e07091a87625"
}

print("=== VERIFYING DATASET INTEGRITY ===")
all_pass = True
for rel_path, expected_hash in EXPECTED.items():
    full_path = PROJECT_ROOT / rel_path
    if not full_path.exists():
        print(f"❌ MISSING: {rel_path}")
        all_pass = False
        continue
    
    h = hashlib.sha256()
    with open(full_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    actual_hash = h.hexdigest()
    
    if actual_hash == expected_hash:
        print(f"✅ PASS: {full_path.name}")
    else:
        print(f"⚠️ HASH MISMATCH: {rel_path} (Expected: {expected_hash[:12]}..., Got: {actual_hash[:12]}...)")
        all_pass = False

if all_pass:
    print("\n🎉 ALL 14 CORE DATASETS VERIFIED WITH 100% BIT-LEVEL INTEGRITY!")
else:
    print("\n❌ INTEGRITY AUDIT FAILED: Check missing files or run 'git lfs pull'.")
```

### 10.4 Upstream Raw Dataset Provenance & Placement Instructions

If a researcher wishes to re-run raw pipeline extraction from scratch (not required for EDA):
1. **IMF PortWatch Daily Data:** Download `Daily_Ports_Data.csv` (618.96 MB) from IMF PortWatch Data Portal and place under `DL/data/raw/portwatch/Daily_Ports_Data.csv`.
2. **Caribbean AIS Data:** Download `portcalls.csv` (197.57 MB) and `trajectories.csv` (72.56 MB) from Caribbean Open Maritime AIS repository and place under `DL/data/staging/supplementary_acquisition/raw/caribbean_ais/`.
3. **NOAA Storm Events:** Download annual gz files `StormEvents_details-ftp_v1.0_d*.csv.gz` (2019-2025, ~80 MB) from NOAA NCEI FTP and place under `DL/data/staging/supplementary_acquisition/raw/noaa_storm_events/`.
4. **NTAD Rail Network:** Download North American Rail Network Lines CSV (41.63 MB) from BTS NTAD Open Data and place under `DL/data/staging/supplementary_acquisition/raw/ntad_rail_network/`.
5. **EM-DAT Disaster Archive:** Download EM-DAT global disaster archive `260430_emdat_archive.xlsx` (7.76 MB) from EM-DAT portal and place under `DL/data/staging/supplementary_acquisition/raw/emdat/`.


