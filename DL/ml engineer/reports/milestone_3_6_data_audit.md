# Milestone 3.6 — Data and Implementation Audit: Supplementary Dwell-Time & Direct Throughput Panel

**Author:** Deep Learning Engineer  
**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Phase:** 2 — Deep Learning Engineering  
**Workspace:** `DL/ml engineer/`  
**Date:** 2026-11-02  

---

## 1. Executive Summary & Purpose

Milestone 3.6 addresses two critical capability gaps identified against the original Stage 02 specification:
1. **Dwell-Time / Waiting-Time Forecasting:** Transitioning from proxy vessel turnaround times to genuine temporal sequence forecasting using empirical waiting-time and berth service data.
2. **Direct Throughput Forecasting:** Expanding the deep learning forecasting scope from aggregate port-call counts to direct throughput metrics—specifically container vessel calls and total cargo tonnage.

Before implementing models, this audit rigorously inspects all relevant supplementary and core datasets across the repository to verify file paths, schemas, row counts, temporal granularities, missingness, measurement semantics, and methodological legitimacy.

---

## 2. Inventory of Candidate Datasets

| Dataset Identifier | Canonical File Path | Temporal Frequency | Spatial Scope | Observation Count | Usable for Deep Learning? |
|---|---|---|---|---|---|
| **ADB AIS Weekly Dwell** | `DL/data/staging/supplementary_acquisition/raw/AISData.csv` | Weekly | 19 Global Mega-Ports | 3,990 clean (3,991 raw) | **Yes** — Sufficient history (210 contiguous weeks/port) for autoregressive sequence modeling. |
| **Indian DGQI Monthly Dwell** | `DL/data/staging/supplementary_acquisition/processed/indian_port_dwell_monthly.parquet` | Monthly | 13 Major Indian Ports + 3 Regional Averages | 363 records | **No for DL** — Only 12 monthly points (2020); strictly descriptive / cross-sectional benchmark. |
| **US AIS Monthly Dwell** | `DL/data/raw/ais_dwell/monthly_dwell_2015_2025.csv` | Monthly | 5 US Port Complexes | 660 records | **No for DL** — Only 5 ports across 132 months; insufficient spatial-temporal density for deep sequences. |
| **PortWatch Unified Panel** | `DL/data/staging/supplementary_acquisition/processed/unified_temporal/port_daily_unified.parquet` | Daily | 2,065 Global Commercial Ports | 5,280,205 rows | **Yes** — Primary panel containing precomputed 7-day throughput targets and operational core. |
| **Sequence Metadata** | `DL/data/staging/supplementary_acquisition/processed/temporal_sequences/sequence_metadata.parquet` | Daily Reference Date | 2,065 Global Commercial Ports | 5,181,085 sequences | **Yes** — Pre-indexed split tags (Train: 3.69M, Val: 743.4k, Test: 741.3k) and throughput targets. |

---

## 3. In-Depth Dataset Forensic Audits

### 3.1. ADB AIS Weekly Dwell Dataset (`AISData.csv`)

- **Exact Path:** `/Users/abdul/Desktop/DS/DL/data/staging/supplementary_acquisition/raw/AISData.csv`
- **Raw Dimensions:** 3,991 rows $\times$ 70 columns.
  - *Data Cleansing Finding:* Row index 3,990 is entirely empty / NaN across key identifier columns (`Port`, `Year`, `Week`). Dropping this single boundary artifact leaves exactly 3,990 clean, fully populated port-week records.
- **Spatial Coverage:** Exactly 19 major global container ports:
  - Antwerp, Busan, Dubai, Guangzhou, Hamburg, Hong Kong, Kaohsiung, LALB (Los Angeles / Long Beach), NYNJ (New York / New Jersey), Ningbo, Port Klang, Qingdao, Rotterdam, Shanghai, Shenzhen, Singapore, Tanjung Pelepas, Tianjin, Xiamen.
- **Temporal Coverage:** 2019 Week 1 through 2022 Week 52.
  - 2019: 52 weeks (Weeks 1–52) $\times$ 19 = 988 records
  - 2020: 53 weeks (Weeks 1–53) $\times$ 19 = 1,007 records
  - 2021: 53 weeks (Weeks 1–53) $\times$ 19 = 1,007 records
  - 2022: 52 weeks (Weeks 1–52) $\times$ 19 = 988 records
  - Total per port: Exactly 210 contiguous, monotonic weekly observations per port. Zero gaps, zero missing weeks.
- **Key Target Columns & Empirical Distributions:**
  - `avgwaiting_time` (Anchorage Waiting Time):
    - *Definition:* Weekly mean time (in hours) that arriving container vessels spend waiting at outer anchorage before obtaining a designated berth.
    - *Mean:* 12.69 hours; *Std:* 13.32 hours; *Median:* 9.0 hours; *IQR:* [6.0, 15.0] hours; *Min:* 0.0 hours; *Max:* 157.0 hours (~6.5 days, observed at LALB during the peak 2021 congestion crisis).
  - `avgservice_time` (Berth Service Time):
    - *Definition:* Weekly mean time (in hours) spent actively moored at the berth during loading and discharge operations.
    - *Mean:* 31.25 hours; *Std:* 20.07 hours; *Median:* 25.0 hours; *IQR:* [20.0, 36.0] hours; *Min:* 2.0 hours; *Max:* 170.0 hours.
  - *Vessel Breakdown Granularity:* Data provides stratified waiting and service times across three vessel size classes:
    - Feeder: `avgwaiting_time_feed`, `avgservice_time_feed`, `numofcall_feeder`, `count_entrance_feeder`
    - Intermediate: ` avgwaiting_time_intermediate `, ` avgservice_time_intermediate `, `numofcall_intermediate` (Note: whitespace in column header requires string stripping during ingest)
    - Neo-Panamax: `avgwaiting_time_neopanamax`, `avgservice_time_neopanamax`, `numofcall_neopanamax`, `share_neopan`
- **Methodological Target Semantics:**
  - *Anchorage Waiting Time vs. Container Dwell Time:* `avgwaiting_time` represents vessel-level anchorage waiting duration derived from AIS trajectory geofencing. It does **NOT** measure yard-level container dwell time (the duration an individual container sits stacked in a terminal between discharge and gate-out truck drayage).
  - *Mandated Task Terminology:* To maintain absolute scientific integrity, Phase B is formally named: **Weekly Anchorage Waiting-Time Forecasting**.

---

### 3.2. Indian DGQI Dwell Dataset (`indian_port_dwell_monthly.parquet`)

- **Exact Path:** `/Users/abdul/Desktop/DS/DL/data/staging/supplementary_acquisition/processed/indian_port_dwell_monthly.parquet`
- **Dimensions:** 363 rows $\times$ 20 columns.
- **Spatial Coverage:** 13 major Indian commercial ports (e.g., Jawaharlal Nehru Port / Nhava Sheva, Mundra, Haldia, Chennai, Cochin, Visakhapatnam) plus 3 aggregated regional summary entities (`Eastern Region Average`, `Southern Region Average`, `Western Region Average`).
  - *Duplicate/Summary Finding:* Regional averages represent non-port aggregations and must be filtered out of port-level analyses.
- **Temporal Coverage:** Exactly 12 months: January 2020 through December 2020 (`2020-01` to `2020-12`).
- **Core Metrics:**
  - `container_dwell_hours`: True container dwell time (time from vessel discharge to terminal gate-out for imports, and gate-in to vessel loading for exports).
  - `trade_direction`: Stratified into `Import` (192 records, mean 84.1h) and `Export` (171 records, mean 76.1h).
- **Usability Determination:**
  - With only 12 monthly data points per port, this dataset cannot support deep recurrent or transformer sequence architectures (an 8-step lookback would leave only 4 training/evaluation points per port).
  - *Role in Milestone 3.6:* Serves as a vital domain grounding anchor to quantitatively demonstrate the distinction between vessel waiting time (~12.7 hours) and container terminal dwell time (~80.3 hours / ~3.3 days).

---

### 3.3. US AIS Monthly Dwell Dataset (`monthly_dwell_2015_2025.csv`)

- **Exact Path:** `/Users/abdul/Desktop/DS/DL/data/raw/ais_dwell/monthly_dwell_2015_2025.csv`
- **Dimensions:** 660 rows $\times$ 13 columns.
- **Spatial Coverage:** Exactly 5 US container port complexes: Houston, LA_Long_Beach, NY_NJ, Savannah, Seattle.
- **Temporal Coverage:** 132 contiguous months from January 2015 to December 2025 (`2015-01` to `2025-12`).
- **Core Metrics:**
  - `MeanDwellDays`: Mean total vessel port turnaround time in days (mean: 3.98 days, std: 1.29 days, min: 1.19 days, max: 7.73 days).
  - `MedianDwellDays`, `StdDwellDays`, `UniqueVessels`, `TotalObservations`.
- **Usability Determination:**
  - Highly aggregated monthly resolution across only 5 spatial entities ($N=660$). While temporal depth is solid (11 years), the spatial breadth is inadequate for training deep neural networks.
  - Furthermore, mixing monthly US port-turnaround data with weekly global ADB waiting time is methodologically invalid (different frequencies, different target definitions).

---

### 3.4. PortWatch Unified Panel & Direct Throughput Targets

- **Primary Panel Path:** `DL/data/staging/supplementary_acquisition/processed/unified_temporal/port_daily_unified.parquet` (5,280,205 rows, 2,065 ports, 2019–2025).
- **Sequence Metadata Path:** `DL/data/staging/supplementary_acquisition/processed/temporal_sequences/sequence_metadata.parquet` (5,181,085 sequences).
- **Target Semantics & Empirical Characteristics on Validation Set ($N=743,400$):**

| Target Field | Meaning & Operational Scope | Missing Count | Zero % | Mean | Std | 50% (Median) | 75% | Max |
|---|---|---|---|---|---|---|---|---|
| `target_portcalls_7d_future_sum` | Cumulative vessel calls (all vessel types) over $[t+1, t+7]$ | 0 | 18.7% | 16.95 | 44.66 | 5.0 | 16.0 | 1,188.0 |
| `target_container_calls_7d_future_sum` | Cumulative container-carrying vessel calls over $[t+1, t+7]$ | 0 | 61.0% | 4.51 | 17.26 | 0.0 | 2.0 | 295.0 |
| `target_cargo_tonnage_7d_future_sum` | Cumulative metric cargo tonnage loaded/discharged over $[t+1, t+7]$ | 0 | 31.8% | 162,708 t | 707,067 t | 6,538 t | 67,780 t | 19,608,542 t |

- **Cross-Target Empirical Correlation Matrix (Validation Partition):**
  - $\text{Corr}(\text{PortCalls}, \text{ContainerCalls}) = 0.8311$
  - $\text{Corr}(\text{PortCalls}, \text{CargoTonnage}) = 0.6976$
  - $\text{Corr}(\text{ContainerCalls}, \text{CargoTonnage}) = 0.6730$
- **Target Selection & Modeling Decision:**
  - Total port calls include tugboats, service vessels, passenger ferries, and fishing boats that do not reflect logistics throughput.
  - `target_container_calls_7d_future_sum` provides a focused, high-precision metric of global container shipping activity.
  - `target_cargo_tonnage_7d_future_sum` provides the true physical volume metric (in metric tons).
  - Both targets are already precomputed, leak-free, aligned with the $T-2$ operational input convention, and ready for multi-head or independent residual forecasting.

---

## 4. Methodological Decisions for Milestones 3.6B and 3.6C

1. **Phase B (Dwell Time):**
   - Use the **ADB AIS Weekly Dataset** (`AISData.csv`).
   - Forecast target: `avgwaiting_time` (Weekly Anchorage Waiting Time, hours).
   - Lookback: $L=8$ weeks.
   - **Chronological Split Reconciliation (Audit Update):**
     - *Initial Working Proposal:* Train 2019-W01 to 2021-W26; Val 2021-W27 to 2022-W52 (2-way split).
     - *Actual Final Implementation (Methodologically Approved):*
       - **Training Partition:** 2019-W01 to 2020-W53 (105 weeks $\times$ 19 ports = 1,995 observations; 1,843 sequences with $L=8$).
       - **Validation Partition:** 2021-W01 to 2021-W53 (53 weeks $\times$ 19 ports = 1,007 sequences; captures peak global supply chain crisis surge).
       - **Held-Out Test Partition:** 2022-W01 to 2022-W52 (52 weeks $\times$ 19 ports = 931 sequences; captures post-crisis stabilization and true out-of-sample generalization).
     - *Rationale for Reconciliation:* Prompt Section B4 strictly mandated a three-way chronological evaluation sequence where "Training observations must precede validation observations. Validation observations must precede the final held-out evaluation period." A 2-way split leaves no independent held-out evaluation period. Adopting the 3-way annual boundary partition ensures clean, leak-free hyperparameter selection on the 2021 shock year and final frozen confirmation on the 2022 held-out test year.
   - Model: Lightweight Autoregressive GRU Forecaster with port entity embeddings and residual connection to weekly persistence.

2. **Phase C (Direct Throughput):**
   - Leverage the frozen M3.4/M3.5 Transformer encoder ($h_{\text{mean}} \in \mathbb{R}^{64}$) as a frozen feature extractor.
   - Train dedicated, isolated non-negative residual throughput projection heads:
     $$\hat{y} = \operatorname{ReLU}(y_{\text{naive}} + \mathbf{w}^T h + b)$$
     for both Container Vessel Calls and Cargo Tonnage.
   - Strictly maintain frozen checkpoint integrity; zero modifications to `artifacts/transformer_mean_pooling/best_model.pt`.

---

## 5. Audit Conclusion

The data landscape for Milestone 3.6 is verified:
- `AISData.csv` is pristine, continuous, and ideal for weekly anchorage waiting-time forecasting across 19 global container ports.
- `port_daily_unified.parquet` and `sequence_metadata.parquet` contain robust, leak-free direct throughput targets covering container calls and cargo tonnage across the global port network.
- The 2025 blind test split remains completely unread, pristine, and guarded.
- Phase B and Phase C implementation can proceed on sound, verified empirical foundations.

