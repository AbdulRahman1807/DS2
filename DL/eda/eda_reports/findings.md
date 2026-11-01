# Exploratory Data Analysis (EDA) Findings — Global Supply Chain Logistics Bottlenecks

**Project:** Stage 02 · Project 07: Global Seaport Throughput & Supply Chain Bottleneck Forecasting  
**Role:** Senior EDA Engineer  
**Target Deliverable Directory:** `DL/eda/eda_reports/`  
**Execution Pipeline:** `DL/eda engineer/02_eda_engineer.py`  
**Status:** Completed, Verified & 100% Grounded in Production Data Assets  

---

## 1. Executive Summary & Core Architectural Determination

This report presents the rigorous exploratory data analysis of the finalized Stage 02 Deep Learning data assets. The analysis evaluates **5,280,205 daily port observations across 2,065 commercial seaports**, **5,181,085 temporal sequences**, and **two multimodal/maritime spatial graphs** (North American hinterland and Caribbean shipping lanes).

### Key Empirical Determinations:
1. **Supervised Target Reality:** The primary forecasting target is `target_portcalls_7d_future_sum`—the cumulative vessel arrival count over the upcoming 7-day planning horizon ($H = 7$). Continuous daily dwell does not exist across global seaports in any public data source; therefore, high-frequency daily throughput sequence forecasting is paired with weekly and monthly macro dwell calibration benchmarks.
2. **Heavy Target Skewness (+9.53):** While the global median 7-day vessel arrival sum is 5.0 vessels, the distribution exhibits an extreme right-skewed tail reaching 1,188 vessels at megahubs like Singapore, Shanghai, and Rotterdam. Neural sequence models must apply log1p normalization or Smooth L1 loss to prevent gradient explosion.
3. **Strict Chronological Quarantine ($L=30, H=7$):** The 5.18M sequences are strictly partitioned chronologically: Train (71.34%, 2019–2023), Validation (14.35%, 2024), and Test (14.31%, 2025). Target distributions remain highly stable across all three partitions ($	ext{PSI} < 0.05$).
4. **Multimodal Spatial Graph Grounding:** The North American network provides 251,646 nodes and 303,016 edges linking 41 seaports to 1,401 intermodal facilities across Class I rail networks, establishing physical propagation paths for Spatio-Temporal Graph Neural Networks (ST-GNNs).

---

## 2. Dataset Architecture & Structural Dimensions

The Data Engineering pipeline integrates IMF PortWatch AIS, ERA5 atmospheric/oceanographic reanalysis, NOAA storm events, CRED EM-DAT disasters, and BTS capacity baselines into an authoritative, leak-free panel:

| Architectural Property | Verified Dataset Measurement |
| :--- | :--- |
| **File Path** | `DL/data/.../unified_temporal/port_daily_unified.parquet` |
| **Total Rows** | **5,280,205 records** |
| **Total Columns** | **91 features** |
| **Entity Coverage** | Exactly **2,065 commercial seaports** across **180 sovereign nations** |
| **Temporal Extent** | **2019-01-01 to 2025-12-31** (7 contiguous calendar years) |
| **Panel Balance** | **100% Balanced** (Every port contains exactly **2,557 contiguous calendar days**) |
| **Primary Key** | `(portwatch_id, date_utc)` (Verified: 0 duplicate records) |
| **Terminal Boundary Nulls** | Exactly **14,455 records** ($2,065 \times 7$) populated as `NaN` at terminal boundary |

### Spatial Coverage Sparsity of Exogenous Features:
- **ERA5 Offshore Proxies (`has_era5_coverage`):** 5 ports (**0.24%** of rows; 12,785 records).
- **NOAA Storm Exposure (`has_noaa_coverage`):** 56 US ports (**2.71%** of rows; 143,192 records).
- **BTS Capacity Baselines (`has_bts_coverage`):** 36 deepwater gateways (**1.74%** of rows; 92,052 records).
- **EM-DAT Disaster Shocks (`has_emdat_coverage`):** 150 countries (**97.24%** of rows; 5,134,456 records).

> [!WARNING]
> Exogenous shock features must **never** be averaged globally without conditioning on their respective `has_*_coverage` indicator flags. Uncovered non-US ports have structural nulls that must be masked or imputed safely.

---

## 3. Univariate Profile of Target & Observable Features

Descriptive statistics computed directly across all 5.28M records:

| Variable | Valid Count | Mean | Std | Median | IQR | Min | P90 | Max | Skewness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `portcalls` | 5,280,205 | 2.42 | 6.58 | 1.0 | 2.0 | 0.0 | 6.0 | 224.0 | **+9.34** |
| `portcalls_container` | 5,280,205 | 0.62 | 2.49 | 0.0 | 0.0 | 0.0 | 1.0 | 76.0 | **+9.11** |
| `portcalls_tanker` | 5,280,205 | 0.76 | 2.87 | 0.0 | 1.0 | 0.0 | 2.0 | 117.0 | **+13.49** |
| `portcalls_dry_bulk` | 5,280,205 | 0.38 | 1.40 | 0.0 | 0.0 | 0.0 | 1.0 | 88.0 | **+14.36** |
| `total_cargo_tonnage` | 5,280,205 | 22305.07 | 100073.32 | 0.0 | 4045.0 | 0.0 | 46650.0 | 4140242.0 | **+10.59** |
| `pw_calls_rolling_7d_mean` | 5,280,205 | 2.42 | 6.38 | 0.7 | 2.1 | 0.0 | 5.6 | 169.7 | **+9.53** |
| `pw_calls_rolling_14d_mean` | 5,280,205 | 2.42 | 6.36 | 0.7 | 2.1 | 0.0 | 5.5 | 158.4 | **+9.54** |
| `pw_calls_rolling_30d_mean` | 5,280,205 | 2.42 | 6.35 | 0.7 | 2.1 | 0.0 | 5.5 | 156.4 | **+9.55** |
| `target_portcalls_lead_1d` | 5,278,140 | 2.42 | 6.58 | 1.0 | 2.0 | 0.0 | 6.0 | 224.0 | **+9.34** |
| `target_portcalls_lead_3d` | 5,274,010 | 2.42 | 6.58 | 1.0 | 2.0 | 0.0 | 6.0 | 224.0 | **+9.34** |
| `target_portcalls_lead_7d` | 5,265,750 | 2.42 | 6.58 | 1.0 | 2.0 | 0.0 | 6.0 | 224.0 | **+9.34** |
| `target_portcalls_7d_future_sum` | 5,265,750 | 16.95 | 44.66 | 5.0 | 15.0 | 0.0 | 39.0 | 1188.0 | **+9.53** |
| `target_container_calls_7d_future_sum` | 5,265,750 | 4.34 | 16.76 | 0.0 | 2.0 | 0.0 | 10.0 | 356.0 | **+9.26** |
| `target_cargo_tonnage_7d_future_sum` | 5,265,750 | 156149.39 | 653125.44 | 6187.0 | 65989.0 | 0.0 | 309678.0 | 19608542.0 | **+10.34** |

### Target Skewness & Mathematical Transformation Guidance:
- Primary target `target_portcalls_7d_future_sum` exhibits heavy right skewness (**+9.53**) and kurtosis (**+137.68**).
- Applying a logarithmic transformation $\tilde{y} = \log(1 + y)$ successfully reduces skewness to **+0.12**, yielding a symmetric distribution suitable for standard gradient-based optimization in deep neural networks.

---

## 4. 7-Year Global Vessel Trends & Historical Disruption Shocks (2019–2025)

Aggregated annual global vessel arrivals across all 2,065 ports:

| Year | Total Port Calls | Container Calls | Cargo Tonnage (Short Tons) | YoY Vessel Calls % |
| :---: | :---: | :---: | :---: | :---: |
| **2019** | 1,786,836 | 463,206 | 16.09 Billion | BASELINE |
| **2020** | 1,746,560 | 451,403 | 15.97 Billion | -2.25% |
| **2021** | 1,798,765 | 441,129 | 16.76 Billion | +2.99% |
| **2022** | 1,858,163 | 453,261 | 16.85 Billion | +3.30% |
| **2023** | 1,886,743 | 494,631 | 17.03 Billion | +1.54% |
| **2024** | 1,878,439 | 486,302 | 17.56 Billion | -0.44% |
| **2025** | 1,828,397 | 483,439 | 17.50 Billion | -2.66% |

### Analysis of Historical Shocks:
1. **2020 COVID-19 Pandemic Drop:** Global vessel calls contracted by **-2.25%** in 2020 (a decline of 40,276 vessel calls), hitting bottom during the Q2 2020 lockdown period.
2. **2021–2022 Bottleneck & Congestion Crisis:** Rapid consumer goods demand caused global arrivals to surge by **+2.98% in 2021** (1,798,765 calls) and **+3.30% in 2022** (1,858,163 calls), precipitating massive vessel anchorage queues across major gateways.
3. **Day-of-Week Rhythm:** Mondays represent the peak operational arrival day (mean = 2.60 calls/day), whereas weekend days (Saturday and Sunday) drop to 2.28 and 2.23 calls/day (**-14.1% weekend dip**).
4. **Monthly Seasonality:** February represents the annual global minimum (mean = 2.37 calls/day, 969K calls) driven by the shorter calendar month and Asian factory closures during the Lunar New Year holiday. Arrivals surge from March through July (mean = 2.47 calls/day).

---

## 5. Spatial Graph Network Architecture

### 5.1 North American Multimodal Hinterland Graph (`north_american_graph_*.parquet`)
- **Node Inventory (251,646 nodes):** 250,204 freight rail junctions, 1,401 intermodal ramp terminals, and 41 deepwater port complexes.
- **Edge Inventory (303,016 edges):** 302,774 physical rail track segments and 242 port-to-intermodal drayage connectors within 50 km.
- **Class I Railroad Operators:** Canadian National (CN: 42,479 lines), Union Pacific (UP: 42,446 lines), BNSF (35,108 lines), CPKC (29,287 lines), CSX (28,458 lines), and Norfolk Southern (NS: 27,841 lines).
- **Distance Metrics:** Mean rail segment length is 1.41 km (median: 0.38 km, P95: 6.82 km).

### 5.2 Caribbean Regional Maritime Graph (`caribbean_graph_*.parquet`)
- **Node Inventory (528 nodes):** 528 commercial seaports across 38 territories, anchored by 18 canonical PortWatch gateways.
- **Edge Inventory (9,969 directed lanes):** 9,969 empirical AIS shipping lanes, including 1,179 inter-canonical gateway corridors.
- **Voyage Composition:** 44.8% Container voyages, 30.2% Tanker voyages, 13.9% Island Feeder voyages, and 11.1% Dry Bulk / General Cargo.

---

## 6. Native-Resolution Empirical Benchmarks

| Benchmark Dataset | Native Resolution | Entities | Total Records | Temporal Coverage | Key Observed Metrics | Operational Finding |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **ADB Weekly Port Traffic** | Weekly | 19 ports | 3,990 | 2019 - 2022 | Turnaround: 43.9h (std: 28.0h), Waiting: 12.7h, Service: 31.2h | Anchorage waiting time accounts for **28.9%** of total port turnaround time. |
| **Indian Major Ports Dwell (Pooled)** | Monthly | 13 ports | 291 | 2020 | Cargo Dwell: 3.44 days (82.5h, max: 12.83 days) | Landside/customs dwell across 13 major commercial ports in calendar year 2020. |
| **Indian Major Ports Dwell (Import)** | Monthly | 13 ports | 156 | 2020 | Import Dwell: 2.75 days (66.1h, max: 12.83 days) | Rapid evacuation under Direct Port Delivery (DPD) protocols. |
| **Indian Major Ports Dwell (Export)** | Monthly | 12 ports | 135 | 2020 | Export Dwell: 4.23 days (101.4h, max: 10.08 days) | Substantially longer dwell (+35.3h / 1.53x) due to export buffer staging. |

> [!NOTE]
> **DGQI Data Governance & Integrity:** All 363 observations in the source dataset belong strictly to calendar year **2020**. Regional summary rows (72 observations across 3 regions) were excluded from port-level aggregate calculations to prevent double-counting. Export container dwell exhibits a **1.53x** ratio relative to import dwell, reflecting operational buffer times before vessel loading.

---

## 7. Sequence Partitioning & Chronological Stability ($L=30, H=7$)

Sequence sliding windows from `sequence_metadata.parquet` enforce leak-free chronological boundaries:

| Partition | Sequence Count | Share % | Reference Date Span | Target Window Span | Target Mean | PSI vs Train |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | 3,696,350 | 71.34% | 2019-01-30 to 2023-12-24 | Ends 2023-12-31 | 16.87 | **0.0000** |
| **VAL** | 743,400 | 14.35% | 2023-12-31 to 2024-12-24 | Ends 2024-12-31 | 17.43 | **0.0008** |
| **TEST** | 741,335 | 14.31% | 2024-12-31 to 2025-12-24 | Ends 2025-12-31 | 17.02 | **0.0008** |

> [!NOTE]
> **Strict Leakage Quarantine:** Exactly **24,780 candidate boundary-crossing sequences** were pruned by Data Engineering. Target distribution stability is confirmed by near-zero PSI values ($\text{PSI}_{\text{Val}} = 0.0008$, $\text{PSI}_{\text{Test}} = 0.0008$), verifying that the holdout test partition reflects production operational conditions without artificial domain shift.

---

## 8. Concrete Handoff Guidelines for Deep Learning & GNN Engineers

1. **Sequence Tensor Dimension:** Configure PyTorch `Dataset` to output `[BatchSize, 30, C]` feature tensors for lookback $L=30$, predicting multi-step lead vectors `[BatchSize, 7]` and cumulative sum scalar `[BatchSize, 1]`.
2. **Leak-Free Normalization:** Fit all `StandardScaler` / `RobustScaler` instances strictly on the **Train partition ($t \le \text{2023-12-24}$)**. Never compute global statistics across the entire 7-year panel.
3. **Target Transformation & Loss Selection:** Use $\log(1 + y)$ target scaling or **Smooth L1 Loss (Huber Loss, $\beta=1.0$)** to prevent gradient destabilization from megahub throughput spikes (>500 calls/week).
4. **Graph Neural Networks (ST-GNN / GAT):** Ingest `north_american_graph_edges.parquet` and `caribbean_graph_edges.parquet` as sparse adjacency matrices with `distance_km` edge weights, linking port nodes to intermodal hinterlands and maritime corridors.
5. **Baselines to Beat:**
   - *Persistence Baseline:* $\hat{Y}_{7d}(t) = \sum_{k=0}^6 y(t-k)$ (Trailing 7-day backward sum).
   - *Lagged Gradient Boosted Trees (LightGBM/CatBoost):* Consuming rolling 7d, 14d, 30d means and calendar features.
   - *Target Sequence Forecasters:* LSTM / GRU / Temporal Fusion Transformers (TFT).
