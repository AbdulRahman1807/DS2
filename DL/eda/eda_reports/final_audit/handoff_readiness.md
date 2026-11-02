# Stage 02 EDA — Modeling Handoff Readiness & Engineering Specification

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Phase:** Phase 2 (Deep Learning) — Stage 02 to Stage 03 Handoff  
**Target Audience:** Machine Learning, Deep Learning, Graph Neural Network, and Evaluation Engineers  
**Audit Directory:** `DL/eda/eda_reports/final_audit/`  
**Audit Date:** 2026-11-01  
**Audit Status:** READY WITH QUALIFICATIONS  

---

## 1. Confirmed Dataset & Sequence Specifications

Downstream sequence forecasters (LSTM, GRU, Temporal Fusion Transformers) and Spatio-Temporal Graph Neural Networks (ST-GNN) must configure their data pipelines to the following empirically verified dataset parameters:

| Parameter | Confirmed Value | Verification Source | Downstream Implementation Requirement |
| :--- | :--- | :--- | :--- |
| **Primary Panel Path** | `DL/data/staging/supplementary_acquisition/processed/unified_temporal/port_daily_unified.parquet` | Parquet metadata | Ingest as primary global daily table; strictly 1 row per `(portwatch_id, date_utc)`. |
| **Panel Dimensions** | **5,280,205 rows**, 91 columns | Verified on disk | Full Cartesian grid: 2,065 ports $\times$ 2,557 days (2019-01-01 to 2025-12-31). Zero missing rows. |
| **Sequence Metadata Path** | `DL/data/staging/supplementary_acquisition/processed/temporal_sequences/sequence_metadata.parquet` | Parquet metadata | Ingest as index for sliding window sequence generation. |
| **Sequence Volume** | **5,181,085 valid sequences** | Verified on disk | Exactly 24,780 boundary-crossing sequences have been quarantined by Data Engineering. |
| **Lookback Horizon ($L$)** | **$L = 30$ days** | Verified in Table 12 | Captures 4 complete weekly harmonics and monthly turning points. Form tensor lookback dimension. |
| **Forecast Horizon ($H$)** | **$H = 7$ days** | Verified in Table 13 | Multi-step lead vector $\mathbf{y}_{t+1:t+7} \in \mathbb{R}^{7}$ and/or cumulative sum scalar $y_{\text{sum}} \in \mathbb{R}$ (select modeling contract per Section 3). |

---

## 2. Eligible Predictor Feature Pool ($C$) & Absolute Exclusions

> [!CAUTION]
> **Total Schema Width vs Eligible Predictor Width ($C$):**  
> The Parquet file contains **91 columns**. Downstream engineers must **NEVER** set input channel dimension $C = 91$. Ingesting all columns will cause fatal target leakage and string type casting failures. Furthermore, downstream engineers must **NEVER hardcode $C = 72$** without documenting their feature selection and encoding choices: $C = 72$ represents the maximum numerical and binary candidate pool before availability filters, domain pruning, and categorical encoding are applied, not a guaranteed universal input width.

### Mandatory Column Exclusions (18 Columns):
1. **Future Target Variables (10 Columns — ZERO TOLERANCE FOR INPUT LEAKAGE):**
   - 7 Daily Leads: `target_portcalls_lead_1d`, `target_portcalls_lead_2d`, `target_portcalls_lead_3d`, `target_portcalls_lead_4d`, `target_portcalls_lead_5d`, `target_portcalls_lead_6d`, `target_portcalls_lead_7d`
   - 3 Cumulative Future Sums: `target_portcalls_7d_future_sum`, `target_container_calls_7d_future_sum`, `target_cargo_tonnage_7d_future_sum`
   *(Must be evaluated strictly as targets in supervised loss functions $\mathcal{L}(y, \hat{y})$; zero tolerance for inclusion in predictor tensors).*
2. **Metadata & String Identifiers (8 Columns):**
   - Calendar Partitions: `year`, `month`, `day`, `date_utc` (temporal indices; use engineered cyclic calendar features instead).
   - Entity Identifiers: `portwatch_id`, `portwatch_name`, `country_name`, `country_iso3` (string keys; index as node indices or learnable entity embeddings, never as raw continuous feature channels).

### Eligible Predictor Candidate Pool (73 Total Non-Target Columns):
The 73 non-target, non-key columns decompose into:
- **Continuous Numerical Predictors (66 Columns):**
  - Operational Ground Truth (12): `portcalls`, `portcalls_container`, `portcalls_dry_bulk`, `portcalls_general_cargo`, `portcalls_roro`, `portcalls_tanker`, `portcalls_cargo`, `import_container`, `import_cargo`, `export_container`, `export_cargo`, `total_cargo_tonnage`.
  - Trailing Rolling Means (5): `pw_calls_rolling_7d_mean`, `pw_calls_rolling_14d_mean`, `pw_calls_rolling_30d_mean`, `pw_container_rolling_7d_mean`, `pw_container_rolling_14d_mean`.
  - Calendar Indices (2): `day_of_week`, `day_of_year`.
  - Weather Reanalysis (15): Continuous ERA5 atmospheric and marine columns (wind components, wave heights/periods, temperature, sea-level pressure, precipitation).
  - NOAA Storm Shocks (10): Continuous wind/pressure and storm event count metrics.
  - EM-DAT National Disasters (10): Continuous/count disaster occurrence indicators.
  - BTS Hinterland Capacity (12): Continuous rail carload, intermodal lift, and TEU baseline metrics.
- **Binary Status & Sparsity Masks (6 Columns):**
  - `is_weekend`, `has_era5_coverage`, `has_noaa_coverage`, `has_emdat_coverage`, `has_bts_coverage`, `is_bts_forward_filled`.
- **String Categorical Column (1 Column):**
  - `era5_wave_proxy_type`: String identifier indicating the marine wave source (`native_era5_ocean`, `nearest_neighbor_era5_ocean`, `synthetic_calm_sheltered`). This column **cannot be ingested as a raw float**. It requires one-hot encoding (expanding input dimension by $K$ binary indicator channels) or explicit exclusion.

### Usable Input Channel Dimensions ($C$) by Modeling Strategy:
Downstream engineers must explicitly select and document one of the following feature subset configurations rather than assuming a universal dimension:
- **Core Universal Feature Set ($C = 20$):** Operational Ground Truth (12) + Trailing Rolling Means (5) + Calendar Indices and Flags (3: `day_of_week`, `day_of_year`, `is_weekend`). Provides 100% data density across all 2,065 ports and 2,557 days without missingness or imputation.
- **Global Macro Feature Set ($C = 30$):** Core Universal (20) + EM-DAT National Disaster Indicators (10). Provides high coverage across 2,008 ports (97.24% of ports).
- **Maximum Numerical/Binary Candidate Pool ($C = 72$):** All 66 continuous numerical features + 6 binary status/coverage masks (with `era5_wave_proxy_type` excluded).
- **One-Hot Encoded Candidate Pool ($C = 72 + K$):** Maximum numerical/binary candidate pool ($C=72$) plus $K$ one-hot categories for `era5_wave_proxy_type` (e.g. $C = 75$ for 3 one-hot categories).

---

## 3. Forecasting Target Families & Output Modeling Contracts

The unified panel dataset and sequence metadata provide two distinct target families spanning the 7-day forecast horizon:
1. **Daily 7-Step Lead Vector (7 Columns):** `target_portcalls_lead_1d`, `target_portcalls_lead_2d`, `target_portcalls_lead_3d`, `target_portcalls_lead_4d`, `target_portcalls_lead_5d`, `target_portcalls_lead_6d`, `target_portcalls_lead_7d`. These columns capture the expected day-by-day arrival sequence.
2. **Cumulative 7-Day Future Sums (3 Columns):** `target_portcalls_7d_future_sum` (primary operational bottleneck target), `target_container_calls_7d_future_sum`, and `target_cargo_tonnage_7d_future_sum`. These aggregate volume over the full window $[t+1, t+7]$.

Downstream ML and DL engineering leads must **select and formally document one of the following three modeling contracts** rather than hardcoding a single universal output tensor shape:

### Contract 1: Port-Call Cumulative Sum Only (Scalar Bottleneck Forecasting)
- **Target Variable:** `target_portcalls_7d_future_sum`
- **Output Tensor Shape:** $\mathbf{\hat{y}} \in \mathbb{R}^{B \times 1}$ (or scalar $\hat{y} \in \mathbb{R}^1$ per sequence).
- **Supervised Loss:** $\mathcal{L}(y, \hat{y})$ evaluated strictly on 7-day future sum port calls.
- **Operational Rationale:** Simplest and most direct formulation. Aligns with weekly macro-logistics bottleneck monitoring, smooths high-frequency day-to-day arrival jitter, and minimizes loss variance. Strongly recommended as the primary initial baseline for all deep sequence (LSTM/GRU) and spatial GNN models.

### Contract 2: Multi-Task Cumulative Sums (Throughput & Mode Vector)
- **Target Variables:** `target_portcalls_7d_future_sum`, `target_container_calls_7d_future_sum`, `target_cargo_tonnage_7d_future_sum`
- **Output Tensor Shape:** $\mathbf{\hat{y}} \in \mathbb{R}^{B \times 3}$
- **Supervised Loss:** $\mathcal{L}_{\text{multitask}} = w_1 \mathcal{L}(y_{\text{calls}}, \hat{y}_{\text{calls}}) + w_2 \mathcal{L}(y_{\text{container}}, \hat{y}_{\text{container}}) + w_3 \mathcal{L}(y_{\text{tonnage}}, \hat{y}_{\text{tonnage}})$
- **Operational Rationale:** Jointly models total traffic volume, containerization intensity, and deadweight cargo mass. Enables multi-task representation sharing across logistics modes while maintaining robust weekly aggregation.

### Contract 3: Daily Multi-Step Lead Vector (Trajectory Forecasting with Optional Sum Derivation)
- **Target Variables:** `target_portcalls_lead_1d` through `target_portcalls_lead_7d` (7 daily steps).
- **Output Tensor Shape:** $\mathbf{\hat{y}}_{1:7} \in \mathbb{R}^{B \times 7}$
- **Cumulative Sum Handling:** 
  - *Option 3A (Deterministic Post-Processing):* Derived post-hoc via sum $\hat{y}_{\text{sum}} = \sum_{k=1}^7 \hat{y}_{t+k}$.
  - *Option 3B (Joint Multi-Objective Optimization):* Output $\mathbf{\hat{y}} \in \mathbb{R}^{B \times 8}$ (or $\mathbb{R}^{B \times 7}$ with auxiliary sum loss), minimizing $\mathcal{L} = \mathcal{L}_{\text{daily}}(\mathbf{y}_{1:7}, \mathbf{\hat{y}}_{1:7}) + \lambda \mathcal{L}_{\text{sum}}(y_{\text{sum}}, \sum_{k=1}^7 \hat{y}_{t+k})$.
- **Operational Rationale:** Required only if downstream operational applications require day-specific berth allocation and pilotage scheduling rather than weekly aggregate capacity monitoring.

---

## 4. Confirmed Temporal Split Boundaries & Target-Window Partitioning

> [!IMPORTANT]
> **Clarifying Target-Window Separation vs Rolling Lookback Input Context:**  
> The partition strategy enforces strict **target-window separation**, not total isolation of all historical input observations. Sliding-window sequence models require $L=30$ days of historical lookback. Therefore, validation sequences forecasting early-2024 targets legitimately ingest historical observations from December 2023 as input context, and test sequences forecasting early-2025 targets legitimately ingest December 2024 observations. Target windows, however, remain strictly segregated across annual partitions.

The partitions in `sequence_metadata.parquet` and Table 08 are configured as follows:

| Partition | Sequence Count | Share % | Calendar Target Window | Reference Date Range ($t$) | Target End Date Range ($t+7$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | **3,696,350** | **71.34%** | **2019 to 2023 (5 Full Years)** | 2019-01-30 to 2023-12-24 | 2019-02-06 to 2023-12-31 |
| **VAL** | **743,400** | **14.35%** | **2024 (1 Full Year)** | 2023-12-31 to 2024-12-24 | 2024-01-07 to 2024-12-31 |
| **TEST** | **741,335** | **14.31%** | **2025 (1 Full Year)** | 2024-12-31 to 2025-12-24 | 2025-01-07 to 2025-12-31 |

### Target-Window Partitioning & Boundary Quarantine Mechanics:
- **Role of the 24,780 Quarantined Sequences:** Exactly 24,780 sliding sequences were pruned by Data Engineering ($12,390$ sequences at the 2023/2024 boundary and $12,390$ sequences at the 2024/2025 boundary). These sequences had reference dates $t$ whose 7-day future target windows $[t+1, t+7]$ straddled partition boundaries (e.g., target windows partially in 2023 and partially in 2024). Pruning them guarantees that **100% of target observations in TRAIN occur in 2019–2023, 100% of target observations in VAL occur in 2024, and 100% of target observations in TEST occur in 2025**.
- **Historical Input Context Across Boundaries:** Sequence models evaluating validation targets starting on 2024-01-01 (reference date 2023-12-31) ingest input lookback windows spanning 2023-12-02 to 2023-12-31. Similarly, test sequences forecasting targets starting on 2025-01-01 (reference date 2024-12-31) ingest input lookback windows spanning 2024-12-02 to 2024-12-31. This is standard and required in time-series evaluation: isolating input lookback at calendar boundaries would artificially blind models for the first 30 days of each evaluation period.
- **Scaler Fitting Protocol:** All feature transformers (`StandardScaler`, `RobustScaler`, min-max) must be fit strictly on the `TRAIN` sequence partition ($t \le \text{2023-12-24}$). The scaler parameters (mean, scale, quantiles) are frozen and applied forward to transform input sequences in `VAL` and `TEST`. Input observations from late December 2023 that serve as lookback context for early validation targets fall within the historical training timeline and are transformed using the frozen training scalers. Under no circumstances may observations from 2024 or 2025 be included in scaler fitting.

---

## 5. Historical Feature Availability Constraints & Evidential Categorization

To ensure realistic historical backtesting and prevent look-ahead bias, downstream model pipelines must respect feature availability constraints. The stated reporting latencies are categorized below by their evidential status:

### 1. Source-Documented Publication Latency (Contractual / Source Data Guarantees)
- **ECMWF Atmospheric & Marine Weather (ERA5 / ERA5T):** ECMWF documentation establishes that near-real-time ERA5T data is published with a **5-day latency**, while final consolidated ERA5 reanalysis is published with a **2 to 3-month latency**. In historical simulation, models must lag reanalysis inputs by $\ge 5$ days. For forward operational deployment ($t+1 \dots t+7$), models cannot ingest reanalysis actuals; they must ingest Numerical Weather Prediction (NWP) forecasts (e.g., ECMWF IFS or NOAA GFS).
- **NOAA NCEI Storm Events Database:** Formally published in monthly batch releases with a documented **60 to 90-day latency**. Furthermore, monetary damage figures (`noaa_storm_damage_property_sum`) undergo retrospective post-disaster insurance and governmental audits. Audited loss values must be excluded from predictor inputs.
- **CRED EM-DAT Macro Disasters:** Assembled and verified retrospectively with an audited publication lag of **3 to 6 months post-event**. Impact metrics (`total_damage_usd`, `total_deaths`) are periodically revised over subsequent years and must be excluded from predictor inputs.
- **US BTS Hinterland Port Performance:** Annual port capacity and intermodal reports are compiled retrospectively and published **12 to 18 months after calendar year-end** (e.g. the 2021 annual report is published in late 2022).

### 2. Empirically Observed Delay (Platform Processing Dynamics)
- **IMF PortWatch Operational Calls & Cargo:** Derived from UN Global Platform AIS satellite/terrestrial streams. Empirically observed ingestion and consolidation cadence on the PortWatch platform exhibits a **24 to 48-hour refresh delay** from vessel movement to public dashboard availability.

### 3. Conservative Engineering Assumptions (Backtesting Simulation Firewalls)
- **PortWatch $T-2$ Day Lag Rule:** In the absence of per-record system ingestion timestamps in the historical dump, a **mandatory 2-day backward lag** is imposed during backtesting. At forecast time $t$, operational call inputs are available only up to $t-2$, unless downstream deployments explicitly assume a localized, private, real-time AIS receiver at the port facility.
- **BTS $\ge 1$-Year Backward Lag Rule:** A pragmatic engineering firewall ($t_{\text{base}} \le t_{\text{eval}} - 1\text{ year}$) ensuring annual baselines are not used until the subsequent year has elapsed. BTS metrics must be treated as static structural node capacity features, not high-frequency dynamic predictors.
- **Active Warning Binary Flags as Contemporaneous Proxies:** While audited disaster damage figures are quarantined, active binary flags (`noaa_is_storm_active_50km`, `emdat_macro_disaster_active`) are permitted as contemporaneous proxies on the assumption that real-time storm warnings and emergency declarations are publicly broadcast at event onset.

### 4. Unverified Availability Assumptions (Supplementary Static Datasets)
- Supplementary research datasets (ADB Port Traffic, Zenodo US AIS Dwell, Mendeley Surabaya TOS, DGQI Indian Dwell) are static research aggregates and offline calibration assets. **No live operational feed exists for these datasets.** Downstream engineers must treat them strictly as external evaluation benchmarks and domain priors, never assuming real-time feature availability in production inference.

> [!TIP]
> **Simulation Constraints vs Production Deployment:**  
> These latency constraints represent rigorous simulation parameters for historical backtesting. Downstream engineers deploying models against live enterprise feeds (e.g., direct port TOS feeds or proprietary real-time AIS APIs) may relax the $T-2$ operational lag upon verifying live data ingestion latency.

---

## 6. Qualified Target Statistics & Exploratory Loss Functions

- **Target Statistics (`target_portcalls_7d_future_sum`):** Mean $= 16.95$, Median $= 5.0$, P90 $= 39.0$, Max $= 1,188$, Skewness $= 9.53$, Kurtosis $= 137.68$, Zero-inflation $= 18.74\%$.
- **Exploratory Loss Recommendation:** **Smooth L1 Loss (Huber Loss, $\beta=1.0$)** applied to $\log(1+y)$ transformed target values.
- **Modeling Qualification:** High skewness and kurtosis motivate robustness against extreme megahub throughput outliers, but Huber loss is a **modeling hypothesis**, not a mathematical mandate. Downstream engineers should benchmark:
  1. Smooth L1 Loss on $\log(1+y)$
  2. Standard MSE on $\log(1+y)$
  3. Tweedie Loss (compound Poisson-Gamma distribution for zero-inflated continuous targets)
  4. Negative Binomial / Zero-Inflated Poisson loss for discrete call counts.

---

## 7. Spatial Graph Limitations & Modeling Hypotheses

### North American Hinterland Network (`north_american_graph_*.parquet`):
- **Graph Topology:** 251,646 nodes, 303,016 edge rows (collapsing to 289,343 unique undirected edges in a simple graph; 13,673 multi-edges). Simple graph mean degree $= 2.30$; multigraph mean degree $= 2.41$. Density $= 9.58 \times 10^{-6}$.
- **Node Breakdown:** 250,204 rail junctions (99.43%), 1,401 intermodal facilities (0.56%), 41 deepwater ports (0.01%).
- **Connectivity:** All 41 ports link to 242 intermodal terminals via proximity-based transfer connectors (mean distance 10.07 km).
- **CRITICAL LIMITATION:** The 242 transfer edges are **inferred spatial proximity linkages (<30 km)**, NOT confirmed physical rail track or contracted drayage routes. Furthermore, the remaining 1,159 intermodal facilities have no rail edges in this file.
- **Architectural Hypothesis:** 2-to-3 layer GraphSAGE or Relational GCN with `distance_km` edge weights. Must be empirically validated against non-graph sequence baselines.

### Caribbean Maritime Network (`caribbean_graph_*.parquet`):
- **Graph Topology:** 528 nodes, 9,969 directed lanes. Density $= 0.0358$.
- **CRITICAL LIMITATION:** 121 catalog nodes have degree 0 (no recorded AIS voyages). The active route network consists of **407 ports**.
- **Giant SCC:** 394 nodes, diameter $= 6$ hops, average path length $= 2.498$ hops.
- **Architectural Hypothesis:** 1-to-2 layer ST-GATv2 with voyage attention weighting. Must be empirically tested against global persistence and LSTM baselines.

---

## 8. Supplementary Data Limitations & Operational Firewall

Downstream engineers must enforce strict conceptual firewalls between operational dwell metrics:
1. **Anchorage Waiting Time (ADB):** Offshore queue duration before pilot boarding. Surged $+171.2\%$ for Neopanamax vessels during the 2021 crunch.
2. **Berth Service Time (ADB):** Active cargo working time at berth.
3. **AIS Geofence Dwell (Zenodo):** Macro vessel residence time inside the entire port polygon.
   - *Correlation Finding:* Correlates with ADB turnaround at $r = +0.8373$ for LA/Long Beach, but has a negative correlation ($r = -0.1294$) for NY/New Jersey. **Do not treat Zenodo AIS geofence dwell as interchangeable with container ship turnaround.**
4. **Container Yard Dwell (Mendeley):** Time cargo containers sit in the terminal yard from vessel discharge to gate-out truck dispatch. Red Channel physical customs inspection doubles dwell from 3.18 to 5.85 days (+84%). Applies strictly to terminal gate/customs dynamics; do not equate with macro port congestion.
5. **Indian Port Dwell (DGQI):** Confined to 2020 calendar year across 13 commercial ports. Western Region exhibits a 3.08x export-to-import dwell ratio, while the national all-port commercial average is 1.53x.

---

## 9. Reference Baselines to Beat & Baseline Mathematical Equivalence

To demonstrate that Spatio-Temporal Graph Neural Networks or deep sequence architectures deliver genuine predictive value, downstream models must outperform established statistical and tabular baselines.

### Baseline Consolidation & Mathematical Equivalence for Cumulative Targets:
In previous documentation, "Trailing 7-Day Sum Persistence" and "Day-of-Week Seasonal Naive" were presented as distinct cumulative baselines. A rigorous mathematical analysis reveals that **for 7-day cumulative sum forecasting (Contract 1 and Contract 2), these two baselines are mathematically identical**:

Let $y(t)$ represent daily port calls at day $t$, and let the cumulative 7-day future target be $Y_{\text{sum}}(t) = \sum_{k=1}^7 y(t+k)$.
1. **Trailing 7-Day Window Sum Persistence:**
   $$\hat{Y}_{\text{persist}}(t) = \sum_{j=0}^6 y(t-j) = y(t) + y(t-1) + \dots + y(t-6)$$
2. **Day-of-Week Seasonal Naive Sum:** Each future day $t+k$ ($k \in \{1, \dots, 7\}$) is forecasted using the identical day-of-week from the previous week ($t+k-7$):
   $$\hat{Y}_{\text{seasonal-sum}}(t) = \sum_{k=1}^7 y(t+k-7) = y(t-6) + y(t-5) + \dots + y(t) = \sum_{j=0}^6 y(t-j)$$

Because any contiguous 7-day window contains exactly one Monday, one Tuesday, ..., and one Sunday, the sum of day-of-week seasonal naive predictions over a 7-day horizon collapses identically to the trailing 7-day sum:
$$\sum_{k=1}^7 y(t+k-7) \equiv \sum_{j=0}^6 y(t-j)$$
Consequently, they generate identical scalar forecasts and identical evaluation errors for 7-day future sum targets. They are consolidated into a single benchmark for cumulative targets.

### Divergence for Multi-Step Daily Vector Forecasting (Contract 3):
For downstream teams implementing **Contract 3** ($\mathbf{\hat{y}}_{1:7} \in \mathbb{R}^7$), these two baselines diverge substantially:
- **Daily Persistence Baseline (Flat Profile):** Assumes zero trend and zero intra-week variation, predicting a constant scalar across all 7 forward steps:
  $$\hat{y}_{t+k} = y(t) \quad \forall k \in \{1, \dots, 7\}$$
- **Day-of-Week Seasonal Naive (Cyclic Profile):** Replicates the complete intra-week daily arrival curve from the preceding 7 days:
  $$\hat{y}_{t+k} = y(t+k-7) \quad \forall k \in \{1, \dots, 7\}$$
  In liner shipping, where vessel arrivals follow scheduled weekly rotations, Day-of-Week Seasonal Naive provides a substantially stronger daily trajectory baseline than flat daily persistence.

### Consolidated List of Mandatory Baselines to Beat:

1. **Trailing 7-Day Sum Persistence / Day-of-Week Seasonal Naive Sum (Consolidated Cumulative Baseline):**
   $$\hat{Y}_{\text{sum}}(t) = \sum_{j=0}^6 y(t-j)$$
   Primary benchmark for Contract 1 and Contract 2. For Contract 3, evaluate both Flat Daily Persistence ($\hat{y}_{t+k} = y(t)$) and Cyclic Seasonal Naive ($\hat{y}_{t+k} = y(t+k-7)$).
2. **Historical Port-Mean Baseline:**
   Long-term historical mean port calls computed per port over the training period ($\bar{y}_i = \frac{1}{|T_{\text{train}}|} \sum_{t \in T_{\text{train}}} y_i(t)$). Benchmarks whether models capture port-specific scale vs dynamic temporal fluctuations.
3. **Gradient Boosted Decision Trees (LightGBM / CatBoost):**
   Tabular GBDT ingesting `pw_calls_rolling_7d_mean`, calendar features, difference residuals (`roll7 - roll30`), and static port metadata. Serves as the competitive non-deep tabular baseline.
4. **Standard Non-Spatial Sequence Models:**
   Unregularized LSTM and GRU sequence forecasters consuming temporal features without graph topology, isolating the marginal value of spatial graph message passing.
