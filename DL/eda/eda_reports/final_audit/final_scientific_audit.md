# Stage 02 Deep Learning EDA — Final Independent Scientific Audit Report

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Phase:** Phase 2 (Deep Learning) — Stage 02: Exploratory Data Analysis & Spatial/Temporal Feature Verification  
**Auditor:** Senior Independent Technical & Scientific Auditor  
**Repository:** `AbdulRahman1807/DS2` | **Branch:** `main`  
**Date of Audit:** 2026-11-01  
**Audit Directory:** `DL/eda/eda_reports/final_audit/`  
**Deliberate Verdict:** **READY WITH QUALIFICATIONS**  

---

## 1. Executive Verdict

The Stage 02 Deep Learning Exploratory Data Analysis (EDA) is certified as **READY WITH QUALIFICATIONS**.

### Basis for the Verdict:
1. **Core Data Integrity & Pipeline Scale (PASS):** The foundational data assets—specifically the 5,280,205-row primary panel ([`port_daily_unified.parquet`](file:///Users/abdul/Desktop/DS/DL/data/staging/supplementary_acquisition/processed/unified_temporal/port_daily_unified.parquet)), the 5,181,085-sequence metadata table ([`sequence_metadata.parquet`](file:///Users/abdul/Desktop/DS/DL/data/staging/supplementary_acquisition/processed/temporal_sequences/sequence_metadata.parquet)), the 251,646-node North American multimodal graph, and the 528-node Caribbean maritime graph—are authentic, structurally sound, and reproducible. The master pipeline ([`02_eda_engineer.py`](file:///Users/abdul/Desktop/DS/DL/eda%20engineer/02_eda_engineer.py)) executes end-to-end on the real data in **112.63 seconds**.
2. **Identification of Material Discrepancies & Qualifications (QUALIFIED):** The audit uncovered specific instances where narrative prose, hardcoded figure annotations, and unverified assumptions contradicted the underlying generated tables. Specifically:
   - **Autocorrelation Mass Discrepancy:** The narrative claim that a 30-day lookback captures "85.4% of cumulative 60-day autocorrelation mass" is demonstrably incorrect; the actual empirical calculation in Table 12 is **54.22%**.
   - **Zenodo vs ADB Correlation Contradiction:** Narrative prose and Figure 16 reported correlations of $+0.428$ (LA/LB) and $+0.457$ (NY/NJ), whereas Table 22 and the underlying data recomputation prove the true correlations are **$+0.8373$ (LA/LB)** and **$-0.1294$ (NY/NJ)**.
   - **Sequence Partition Narrative Drift:** Narrative prose claimed an approximate 70/10/20 split ending in September 2024 (3.62M/518k/1.04M), whereas the actual generated partition in Table 08 and `sequence_metadata.parquet` is strictly annual: **Train 3,696,350 (71.34%, 2019–2023)**, **Val 743,400 (14.35%, 2024)**, and **Test 741,335 (14.31%, 2025)**.
   - **Graph Degree Arithmetic:** The reported mean degree of 2.30 for North American rail reflects the collapsed simple graph (289,343 unique edges), whereas the multigraph in Parquet (303,016 edge rows) has a mean degree of 2.41. Furthermore, 121 Caribbean ports are inactive (0 voyages), and port-to-intermodal connectors are inferred spatial buffers (<30 km), not confirmed logistics links.
   - **Predictor Feature Count ($C$):** Downstream engineers must not set $C = 91$. The actual pool of eligible continuous numerical predictors is **$C = 66$** (or $C = 72$ with binary flags; $C = 20$ for the global operational core), strictly excluding 10 future targets and 8 metadata identifiers.

Because these issues are confined to documentation prose and operational assumptions rather than fatal dataset corruption, and because the underlying tables and parquet assets are sound, downstream engineering can proceed safely under the explicit qualifications documented herein.

---

## 2. Scope and Audit Methodology

This audit conducted an independent, line-by-line verification across:
1. **Source Data & Columnar Files:** Inspected Parquet metadata, row groups, schemas, and value distributions for `port_daily_unified.parquet`, `sequence_metadata.parquet`, `north_american_graph_*.parquet`, `caribbean_graph_*.parquet`, `adb_port_traffic_weekly.parquet`, `indian_port_dwell_monthly.parquet`, and `mendeley_event_log.parquet`.
2. **Analysis Scripts:** Audited logic, statistical solvers, NetworkX algorithms, and pandas dataflows in `02_eda_engineer.py`, `03_temporal_dependency_analysis.py`, `04_spatial_graph_topology.py`, `05_feature_quality_and_correlation.py`, `06_supplementary_cross_source.py`, and `07_generate_final_reports.py`.
3. **Generated Tables:** Independently recomputed numerical metrics across all 25 CSV tables in `DL/eda/eda_reports/tables/`.
4. **Visual Figures:** Inspected all 16 PNG figures in `DL/eda/eda_reports/figures/` for visual fidelity, scaling, and annotation correctness.
5. **Narrative Findings:** Compared prose assertions in `findings.md` and `eda_report.txt` against reproducible data outputs.

Every material metric was traced back through the chain:
$$\text{Source Data} \longrightarrow \text{Analysis Script} \longrightarrow \text{Calculated Output} \longrightarrow \text{Table/Figure} \longrightarrow \text{Written Report}$$

---

## 3. Verified Findings

The following headline findings have been independently reproduced and verified as mathematically and scientifically sound:

1. **Global Panel Scale & Completeness:** Primary panel contains exactly **5,280,205 rows**, 91 columns, covering 2,065 canonical ports over 2,557 days (2019-01-01 to 2025-12-31). It forms a complete Cartesian product with zero missing dates or orphan ports.
2. **Quarantined Sequence Generation ($L=30, H=7$):** The sequence index in `sequence_metadata.parquet` contains **5,181,085 valid sliding sequences**. Exactly **24,780 boundary-crossing sequences** were pruned by Data Engineering to prevent sequence-target overlap between partitions.
3. **Forecasting Target Profile:** `target_portcalls_7d_future_sum` across 5,265,750 valid instances has: Mean $= 16.95$, Median $= 5.0$, Std $= 44.66$, P90 $= 39.0$, P99 $= 185.0$, Max $= 1,188.0$, Skewness $= 9.53$, Kurtosis $= 137.68$, and Zero-inflation $= 18.74\%$. The 7-day terminal truncation ($7 \times 2,065 = 14,455$ unforecastable terminal rows) is mathematically exact.
4. **Port Autocorrelation Harmonics:** Singapore (`port1201`) displays sharp weekly harmonics with $\text{ACF}(7) = 0.4219$, $\text{ACF}(14) = 0.3994$, $\text{ACF}(21) = 0.4151$, and $\text{ACF}(28) = 0.4095$. Rotterdam (`port1114`) exhibits $\text{ACF}(7) = 0.3772$.
5. **Scale Confounding in Correlations:** Pooled Pearson correlation between `portcalls` and the 7-day future target is $0.9594$, but within-port mean correlation across the Top 100 ports collapses to **$0.2703$**, proving that pooled correlation is dominated by cross-sectional port size.
6. **Trailing Feature Predictive Signal:** Within-port correlation peaks at the trailing 7-day rolling mean (`pw_calls_rolling_7d_mean`, mean within-port $r = 0.4479$).
7. **Rolling Feature Multicollinearity:** `pw_calls_rolling_7d_mean` and `14d_mean` have $r = 0.9973$; `7d_mean` and `30d_mean` have $r = 0.9955$.
8. **ADB 2021 Supply Chain Congestion:** Global Neopanamax anchorage waiting time surged by **+171.2%** (from 6.6h in 2019 to 17.9h in 2021) across 19 major ports, exceeding 100h at LA/Long Beach.
9. **Mendeley Customs Clearance Bottleneck:** At TPS Surabaya (95,648 containers), Red Channel physical inspection increases container dwell from 3.18 days to 5.85 days (+84%), with delays concentrated in the pre-customs inspection phase (4.47 days vs 0.26 days median for Green Channel).
10. **Exogenous Feature Localized Sparsity:** Confirmed coverage: EM-DAT covers 2,008 ports (97.24%), NOAA covers 56 ports (2.71%), BTS covers 36 ports (1.74%), and ERA5 covers 5 pilot ports (0.24%).

---

## 4. Findings Requiring Qualification

The following findings are numerically valid but require strict narrowing and qualification to prevent downstream misinterpretation:

1. **Port Hedland Seasonality:** Port Hedland exhibits low autocorrelation at lag 7 ($\text{ACF}(7) = 0.0468$). This reflects the operational reality of iron ore terminals (continuous conveyor loading governed by tidal draft windows rather than scheduled weekly liner loops), but it should not be described as proof that no cyclicality exists (e.g. cyclonic or annual monsoon seasonality).
2. **Lookback Window Heuristic ($L=30$):** $L=30$ is an **operational trade-off heuristic**, capturing 4 full weekly harmonics, a complete calendar month, and 54.22% of cumulative 60-day autocorrelation mass. It is not an mathematically proven optimal parameter; downstream sequence length ablations ($L \in \{14, 21, 30, 45\}$) are required.
3. **Loss Function Recommendation:** High skewness (9.53) and kurtosis (137.68) motivate robustness against megahub outliers, but Smooth L1 / Huber loss is an **architectural hypothesis**. Downstream engineers must benchmark it against $\log(1+y)$ MSE, Tweedie loss, and count-based Negative Binomial objectives.
4. **Spatial Proximity Connectors:** The 242 transfer edges linking 41 seaports to intermodal facilities (mean distance 10.07 km) are **inferred spatial proximity linkages (<30 km buffer)** from NTAD data. They are not confirmed physical rail track or dedicated drayage transport links.
5. **Indian Port Dwell Asymmetry:** The 3.08x export-to-import dwell ratio applies specifically to the **Western Region average** (94.60h vs 30.76h), while the national commercial port average across 13 ports is **1.53x** (101.41h vs 66.08h). Furthermore, all data is strictly confined to calendar year 2020.
6. **GNN Architecture Recommendations:** Proposing GraphSAGE for rail and ST-GATv2 for maritime are **modeling hypotheses** based on graph density (planar vs small-world), not established empirical facts.

---

## 5. Incorrect or Inconsistent Findings

The following items in the previous EDA reports and visualizations were found to be demonstrably incorrect or internally contradictory:

1. **Autocorrelation Mass Claim (85.4% vs 54.22%):**
   - *Reported in Prose:* Narrative text in `findings.md`, `eda_report.txt`, and `03_temporal_dependency_analysis.py` asserted that lookback $L=30$ captures "85.4% of 60-day autocorrelation mass".
   - *Actual Calculated Value:* Column `cumulative_autocorr_mass_pct` in Table 12 is **54.22%**.
   - *Cause:* The script author hardcoded "85.4%" into the text string justification variable without referencing the calculated column.
2. **Zenodo AIS vs ADB Correlation Claim ($+0.428/+0.457$ vs $+0.8373/-0.1294$):**
   - *Reported in Prose & Figure 16 Legend:* Text and legend asserted $r = +0.428$ for LA/LB and $r = +0.457$ for NY/NJ.
   - *Actual Calculated Value:* Table 22 (`22_ais_vessel_dwell_vs_portwatch_comparison.csv`) and `SUPPLEMENTARY_DATASETS_INTEGRATION_REPORT.md` report **$+0.8373$ for LA/LB** and **$-0.1294$ for NY/NJ**.
   - *Cause:* Disconnected copy-paste in Figure 16 plotting code; contradicted Table 22.
3. **Sequence Partition Counts Narrative Drift:**
   - *Reported in Prose:* Text asserted Train $= 3,618,348$ (69.84%), Val $= 517,908$ (10.00%), Test $= 1,044,829$ (20.17%), with Val ending in September 2024.
   - *Actual Stored Parquet & Table 08:* Train $= 3,696,350$ (71.34%), Val $= 743,400$ (14.35%), Test $= 741,335$ (14.31%), representing annual calendar splits (Train: 2019–2023, Val: 2024, Test: 2025).
   - *Cause:* Stale documentation from an obsolete partition trial.
4. **Total Schema Width as Predictor Feature Count:**
   - *Reported in Prose:* Narrative text repeatedly referred to "all 91 features" in modeling contexts.
   - *Actual Eligible Predictors:* Ingesting 91 columns into feature tensor $X$ introduces 10 future targets (fatal leakage) and 8 string/calendar keys. The actual continuous numerical feature pool is **$C = 66$** (or $C = 72$ with binary flags).

---

## 6. Unverified Claims

1. **Causal Attribution of Indian Trade Asymmetry:** Asserting that Direct Port Delivery (DPD) and customs cutoffs caused the 3.08x Western Region export/import dwell ratio is an unverified operational hypothesis; the DGQI dataset contains no milestone event timestamps to isolate customs vs trucking duration.
2. **Equivalence of AIS Dwell and Port Bottlenecks:** Claiming that Zenodo AIS geofence dwell represents port congestion is unverified and contradicted by data; at NY/NJ, Zenodo multi-cargo dwell had a negative correlation ($-0.1294$) with container ship turnaround.
3. **GNN Superiority over Sequence Baselines:** Claims that GraphSAGE or GATv2 will prevent over-smoothing and outperform temporal sequence models are unverified hypotheses requiring Stage 03/04 experimental validation.

---

## 7. Spatial Graph Arithmetic and Topology Audit

### Detailed North American Hinterland Network Audit:
- **Node Count:** 251,646 nodes confirmed.
  - Rail Junctions: 250,204 (99.43%)
  - Intermodal Facilities: 1,401 (0.56%)
  - Deepwater Seaports: 41 (0.01%)
- **Edge Count & Multi-Edge Audit:**
  - Parquet rows: **303,016 edges** (302,774 rail lines, 242 transfer connectors).
  - Unordered unique node pairs: **289,343 edges**.
  - Multi-edges (parallel tracks between same junctions): **13,449 pairs** accounting for **13,673 excess rows**.
  - Self-loops: **71 edges**.
- **Mean Degree Reconciliation:**
  - On collapsed simple graph `nx.Graph()`: $\text{Mean Degree} = \frac{2 \times 289,343}{251,646} = \mathbf{2.2996} \approx \mathbf{2.30}$.
  - On raw multigraph: $\text{Mean Degree} = \frac{2 \times 303,016}{251,646} = \mathbf{2.4083} \approx \mathbf{2.41}$.
  - *Audit Resolution:* Both 2.30 and 2.41 are mathematically correct depending on whether the graph is treated as a simple topological graph or a physical track multigraph. Table 14 used simple graph; Table 05 used multigraph.
- **Connectivity & Isolates:**
  - Giant Component: **249,070 nodes (98.98%)**, consisting of rail junctions.
  - Isolated Nodes (Degree 0): **1,159 nodes**, consisting of inland intermodal facilities not linked to seaports.
  - Port-to-Intermodal Connectors: All 41 ports link to 242 intermodal facilities (mean distance 10.07 km, min 0.52 km, max 29.86 km).

### Detailed Caribbean Maritime Network Audit:
- **Node Count:** 528 cataloged entities.
- **Active Node vs Isolated Node Audit:**
  - Nodes appearing in edges: **407 nodes (77.08%)**.
  - Inactive / isolated anchorages (Degree 0): **121 nodes (22.92%)**.
- **Edge Count & Density:**
  - Directed edges: **9,969 edges** (6,530 unique unordered pairs; 3,439 bidirectional route pairs).
  - Density over full catalog: $\frac{9,969}{528 \times 527} = \mathbf{0.035827} \approx \mathbf{3.58\%}$.
  - Density over active nodes: $\frac{9,969}{407 \times 406} = \mathbf{0.060329} \approx \mathbf{6.03\%}$.
- **Centrality & Small-World Metrics:**
  - Giant Strongly Connected Component: **394 nodes** (74.6% of catalog; 96.8% of active network).
  - Average Path Length: **2.498 hops**.
  - Diameter: **6 hops**.
  - Top Hub by Degree: Port Everglades (`carib_294`, total degree 280; in=140, out=140, betweenness centrality $= 0.0432$).

---

## 8. Temporal Dependency and Target Audit

- **ACF/PACF Calculations:** Confirmed using statsmodels FFT and Yule-Walker solvers across 2,557 daily intervals.
- **Weekly Cycles:** Liner megahubs exhibit sharp weekly harmonics ($p < 0.001$), confirming scheduled vessel rotations.
- **Lookback Window Autocorrelation Mass:**
  $$\text{Cumulative Mass}(L) = \frac{\sum_{k=1}^L \overline{\text{ACF}}(k)}{\sum_{k=1}^{60} \overline{\text{ACF}}(k)}$$
  - $L=7$: $14.08\%$
  - $L=14$: $26.92\%$
  - $L=21$: $39.21\%$
  - $L=30$: **$54.22\%$** *(Retracts erroneous narrative claim of 85.4%)*
  - $L=45$: $78.10\%$
  - $L=60$: $100.00\%$
- **Target Distribution & Zero-Inflation:**
  - $18.74\%$ exact zeros in `target_portcalls_7d_future_sum` (986,607 out of 5,265,750 sequences).
  - High kurtosis (137.68) is driven by megahubs (Singapore, Ningbo, Shanghai, Rotterdam) with calls exceeding 500/week, while smaller island ports record 0 to 2 calls.

---

## 9. Feature Quality, Correlation, and Multicollinearity Audit

- **Schema Profiling:** All 91 columns profiled. Confirmed 0 nulls across PortWatch operational columns, calendar features, and rolling features. Exogenous columns carry documented sparsity flags.
- **Scale Confounding Proof:**
  - `portcalls` vs `target_portcalls_7d_future_sum`: Pooled $r = 0.9594$; Within-port mean $r = 0.2703$.
  - `pw_calls_rolling_7d_mean` vs target: Pooled $r = 0.9891$; Within-port mean $r = 0.4479$.
  - *Audit Finding:* Over 70% of pooled correlation variance is attributable to cross-sectional port scale. Downstream feature selection must evaluate within-port time-series correlations.
- **Multicollinearity Assessment:**
  - High correlation ($r > 0.995$) between 7d, 14d, and 30d rolling means induces rank deficiency in unregularized linear layers.
  - Recommended remediation: Ingest 7d mean as baseline autoregressive predictor; encode wider windows as residuals (`roll7 - roll30`).

---

## 10. Supplementary Cross-Source Dataset Audit

1. **ADB Port Traffic Database:** 19 ports, 3,990 weekly observations (2019–2022). Verified Neopanamax waiting time explosion in 2021 (+171.2% globally; from 2.6h to 69.9h at LA/LB).
2. **Zenodo AIS Monthly Dwell Calibration:** 48 monthly intervals (2019–2022) in `lalb_nynj_overlap_comparison.csv`.
   - LA/Long Beach: $r = +0.8373$ (strong positive co-movement during San Pedro Bay container gridlock).
   - NY/New Jersey: $r = -0.1294$ (no positive association; multi-cargo harbor residence time did not track container terminal operations).
   - *Audit Finding:* Confirms that AIS geofence dwell cannot be treated as a universal ground truth for container ship operations.
3. **Mendeley Event Log (TPS Surabaya):** 952,069 event records across 95,648 containers. Verified that customs Red Channel physical inspection doubles dwell (3.18d to 5.85d), with delays isolated to pre-customs clearance.
4. **DGQI Indian Port Dwell:** 363 rows, 13 commercial ports, 2020 only. Western Region export/import ratio $= 3.08x$; National commercial average $= 1.53x$. Regional summaries properly segregated.

---

## 11. Historical Feature Availability & Evidential Categorization

Feature availability constraints are classified by their evidential status (see detailed audit in [`feature_availability_audit.csv`](file:///Users/abdul/Desktop/DS/DL/eda/eda_reports/final_audit/feature_availability_audit.csv)):
1. **Source-Documented Publication Latency:**
   - ERA5T near-real-time reanalysis has a documented 5-day publication latency; consolidated ERA5 has 2–3 month latency. Forward predictions require NWP forecasts (IFS/GFS), not reanalysis actuals.
   - NOAA NCEI Storm Events are published monthly with a 60–90 day latency; monetary damage figures undergo retrospective insurance/disaster audits.
   - CRED EM-DAT macro disaster impact figures are retrospectively audited and revised over 3–6 months.
   - US BTS Annual Port Performance reports are published 12–18 months after calendar year-end.
2. **Empirically Observed Delay:**
   - IMF PortWatch operational daily vessel calls update on the public platform with an observed 24–48 hour refresh cadence post-AIS ingestion.
3. **Conservative Engineering Assumptions (Simulation Firewalls):**
   - PortWatch $T-2$ day lag rule imposed to prevent look-ahead bias during historical simulation.
   - BTS $\ge 1$-year backward lag rule ($t_{\text{base}} \le t_{\text{eval}} - 1\text{ year}$) ensures annual capacity baselines are treated as static structural features.
   - Active warning binary flags (`noaa_is_storm_active_50km`, `emdat_macro_disaster_active`) are permitted as contemporaneous proxies, while audited dollar damages are strictly quarantined.
4. **Unverified Availability Assumptions:**
   - Supplementary research datasets (ADB, Zenodo, Mendeley, DGQI) are offline calibration assets with no live operational feeds.

---

## 12. Downstream Handoff Assessment

| Component | Status | Required Engineering Action |
| :--- | :---: | :--- |
| **Forecasting Output Contract** | **QUALIFIED** | Downstream leads must formally select and document Contract 1 ($\hat{y} \in \mathbb{R}^1$, cumulative sum), Contract 2 ($\mathbf{\hat{y}} \in \mathbb{R}^3$, multi-task cumulative), or Contract 3 ($\mathbf{\hat{y}}_{1:7} \in \mathbb{R}^7$, daily lead vector). Do not hardcode a single universal tensor shape. |
| **Sequence Tensor Input** | **READY** | Configure shape `[BatchSize, 30, C]` with $L=30$. Fit scalers strictly on `TRAIN` ($t \le \text{2023-12-24}$). Transforming validation lookback windows drawing from late-2023 input context uses frozen training scalers. |
| **Input Channel Dimension ($C$)** | **QUALIFIED** | Set $C = 20$ (universal core), $C = 30$ (global macro), or $C = 72$ (full numerical/binary candidate pool). Categorical `era5_wave_proxy_type` requires one-hot encoding or exclusion. **Never hardcode $C = 72$ without documenting feature selection; never set $C = 91$.** |
| **Sequence Partitions** | **QUALIFIED** | Partitions enforce target-window separation (Train 2019–2023, Val 2024, Test 2025; 24,780 boundary-crossing sequences quarantined). Validation legitimately uses late-2023 input lookback, and Test uses late-2024 input lookback. |
| **Loss Function** | **QUALIFIED** | Benchmark Smooth L1 Loss on $\log(1+y)$ against standard MSE, Tweedie loss, and count models. |
| **Spatial Graph Backbones** | **QUALIFIED** | Implement GraphSAGE (rail) and ST-GATv2 (maritime) as modeling hypotheses; benchmark against non-graph sequence baselines. |
| **Reference Baselines** | **CONSOLIDATED** | Trailing 7-day sum persistence and Day-of-Week seasonal naive sum are mathematically identical for cumulative targets; evaluate both flat persistence and cyclic seasonal naive for daily vector targets. |
| **Leakage Quarantine** | **ENFORCED** | Verify that future targets and retrospective damage USD are excluded from feature tensor $X$. |

---

## 13. Prioritised Correction List

1. **[High Priority] Narrative Synchronization:** Update narrative prose in `findings.md` and `eda_report.txt` to replace the hardcoded "85.4% autocorrelation mass" claim with the true Table 12 value of **54.22%**.
2. **[High Priority] Correlation Value Synchronization:** Correct prose and Figure 16 legend to reflect true Table 22 values: **$+0.8373$ for LA/LB** and **$-0.1294$ for NY/NJ** (retracting $+0.428$ and $+0.457$).
3. **[High Priority] Sequence Partition Documentation:** Correct narrative split descriptions to match the true annual calendar partitions in Table 08: Train $= 3,696,350$ (71.34%), Val $= 743,400$ (14.35%), Test $= 741,335$ (14.31%).
4. **[Medium Priority] Graph Definition Transparency:** Note in documentation that 2.30 is simple graph degree (289,343 edges) and 2.41 is multigraph degree (303,016 edges), and that 121 Caribbean ports have 0 recorded voyages.
5. **[Medium Priority] Proximity Connector Labeling:** Explicitly label the 242 seaport-to-intermodal edges as inferred spatial proximity linkages (<30 km), not confirmed logistics transport routes.

---

## 14. Final EDA Readiness Decision

### **VERDICT: READY WITH QUALIFICATIONS**

The completed Stage 02 EDA provides a massive, authentic, and methodologically defensible quantitative foundation for Phase 2 deep learning. The 5.28M-row panel, sequence splits, spatial graphs, and feature inventories are real, verified, and modeling-ready. Downstream engineers can immediately proceed to Stage 03 implementation by observing the qualifications and constraints established in this audit.

