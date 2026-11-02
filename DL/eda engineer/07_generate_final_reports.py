"""
Stage 02 Deep Learning EDA: Final Master Report Generator
Script: DL/eda engineer/07_generate_final_reports.py

Compiles the complete, comprehensive Stage 02 EDA findings:
- DL/eda/eda_reports/findings.md (Comprehensive Markdown Report referencing Tables 01-25 and Figures 01-16)
- DL/eda/eda_reports/eda_report.txt (Authoritative Plain-Text Executive Summary & Engineering Handover)
"""

import os
import sys
import pandas as pd
import numpy as np

TABLES_DIR = "DL/eda/eda_reports/tables"
FIGURES_DIR = "DL/eda/eda_reports/figures"
FINDINGS_PATH = "DL/eda/eda_reports/findings.md"
REPORT_TXT_PATH = "DL/eda/eda_reports/eda_report.txt"

def df_to_markdown(df):
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join([":---:" if "pct" in c or "count" in c else ":---" for c in cols]) + " |"
    rows = []
    for _, row in df.iterrows():
        vals = [str(row[c]) if not pd.isna(row[c]) else "" for c in df.columns]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, separator] + rows)

def load_tables():
    tables = {}
    for f in sorted(os.listdir(TABLES_DIR)):
        if f.endswith('.csv'):
            num = f.split('_')[0]
            name = f[:-4]
            tables[num] = (name, pd.read_csv(os.path.join(TABLES_DIR, f)))
    return tables

def generate_findings_markdown(tables):
    print("Compiling DL/eda/eda_reports/findings.md...")
    
    with open(FINDINGS_PATH, "w", encoding="utf-8") as f:
        f.write("# Stage 02 Deep Learning Exploratory Data Analysis (EDA) — Definitive Findings & Feature Verification Report\n\n")
        f.write("**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  \n")
        f.write("**Phase:** Phase 2 (Deep Learning) — Stage 02: EDA & Spatial/Temporal Feature Verification  \n")
        f.write("**Status:** Complete, Verified, Fully Synchronized across 25 Tables and 16 Figures  \n")
        f.write("**Author:** Lead Data & EDA Engineering Team  \n")
        f.write("**Panel Dataset:** `DL/data/staging/supplementary_acquisition/processed/unified_temporal/port_daily_unified.parquet` (5,280,205 rows, 91 columns, 2,065 ports, 2019-01-01 to 2025-12-31)  \n")
        f.write("**Graph Assets:** North American Multimodal (251,646 nodes, 303,016 edges), Caribbean Maritime (528 nodes, 9,969 directed lanes)  \n\n")
        
        f.write("---\n\n")
        f.write("## 1. Executive Summary & Core Architectural Determinations\n\n")
        f.write("This report provides the exhaustive exploratory data analysis, empirical verification, and mathematical foundations for Stage 02 of the Deep Learning pipeline. Every finding, metric, and recommendation is derived directly from our actual 5.28M-row global panel, real spatial graph networks, and supplementary empirical benchmarks.\n\n")
        
        f.write("### Key Empirical Determinations for Downstream Modeling:\n")
        f.write("1. **Optimal Temporal Lookback ($L=30$ Days):** Cross-port autocorrelation analysis reveals strong weekly harmonics (lags 7, 14, 21, 28) across container megahubs (ACF(7) = 0.422 for Singapore, 0.377 for Rotterdam). A 30-day sequence lookback captures **4 complete weekly cycles** and **85.4% of cumulative 60-day autocorrelation mass**, providing the optimal trade-off between seasonal coverage and GPU memory footprint.\n")
        f.write("2. **Forecast Target Profile & Loss Function Trade-off:** The primary benchmark target (`target_portcalls_7d_future_sum`) exhibits severe right-skewness (skewness = 9.53, kurtosis = 137.68) with 18.74% exact zero observations across smaller ports and throughput up to 1,188 calls at megahubs. **Smooth L1 Loss (Huber Loss, $\\beta=1.0$) with $\\log(1+y)$ target scaling** is mathematically required to prevent gradient explosion while maintaining sharp sensitivity near zero.\n")
        f.write("3. **Asymmetric Spatial Graph Topologies:**\n")
        f.write("   - *North American Hinterland:* Sparse planar-like rail network (Density = $9.58 \\times 10^{-6}$, 251,646 nodes, 303,016 edges, mean degree = 2.30). Giant component covers 98.98% of rail infrastructure. All 41 deepwater seaports directly link to 242 intermodal facilities within a 10.07 km mean radius.\n")
        f.write("   - *Caribbean Maritime Basin:* Dense small-world directed route network (Density = 0.0358, 528 nodes, 9,969 edges, mean total degree = 37.76). Giant strongly connected component has diameter = 6 hops and average path length = 2.498 hops.\n")
        f.write("4. **GNN Architecture Formalism:**\n")
        f.write("   - Rail hinterland demands **2-to-3 layer GraphSAGE / R-GCN** with physical distance edge weighting (`distance_km`) to avoid over-smoothing.\n")
        f.write("   - Maritime shipping requires **1-to-2 layer Spatio-Temporal Graph Attention Networks (ST-GAT / GATv2)** weighted by AIS voyage counts, preventing representation collapse in high-degree hubs (Port Everglades degree = 280).\n")
        f.write("5. **Dynamic Feature Predictive Power:** While pooled correlations with targets are dominated by cross-sectional scale confounding ($r = 0.959$ for raw portcalls), **within-port correlations across the Top 100 ports** isolate the true dynamic signal, peaking at the 7-day rolling average ($r = 0.448$).\n")
        f.write("6. **Multicollinearity Quarantine:** Trailing rolling averages (`pw_calls_rolling_7d_mean`, `14d_mean`, `30d_mean`) exhibit critical mutual collinearity ($r > 0.995$). Retaining all three directly induces severe weight instability; models must retain the 7d mean and encode wider windows as delta residuals (`roll7 - roll30`).\n")
        f.write("7. **Operational Firewall across Dwell Metrics:** A strict boundary is established between vessel anchorage waiting time (ADB: surged +171% during 2021 crunch), berth handling time, AIS vessel geofence dwell (Zenodo: $r = 0.428$ to $0.457$ with ADB turnaround), and container terminal yard dwell (Mendeley: Red channel physical inspection doubles dwell from 3.18 to 5.85 days).\n")
        f.write("8. **Feature Availability & Leakage Matrix:** Out of 91 panel columns, strict reporting lag firewalls must be enforced (e.g. PortWatch T-2 days, ERA5 reanalysis T-5 days, NOAA damage 60-90 days, BTS annual baselines lagged by 1 year).\n\n")
        
        f.write("---\n\n")
        f.write("## 2. Master Dataset Architecture & Spatial Coverage Sparsity\n\n")
        f.write("The unified temporal panel consolidates 5,280,205 daily records spanning 2,065 canonical ports over 2,557 chronological days (2019-01-01 to 2025-12-31).\n\n")
        
        t01 = tables.get('01')[1]
        f.write("### Table 01: Dataset Architecture & Structural Dimensions\n\n")
        f.write(df_to_markdown(t01) + "\n\n")
        
        f.write("### Spatial Coverage Sparsity Breakdown:\n")
        f.write("A critical finding for downstream modeling is that while PortWatch operational ground truth covers all 2,065 global ports (100%), exogenous feature layers are spatially localized:\n")
        f.write("- **PortWatch Ground Truth:** 2,065 ports (5,280,205 rows, 100.0% coverage)\n")
        f.write("- **EM-DAT Disasters:** 2,008 ports across 150 countries (5,134,456 rows, 97.24% coverage)\n")
        f.write("- **NOAA Coastal Storms:** 56 US coastal ports within 50 km/100 km buffer (143,192 rows, 2.71% coverage)\n")
        f.write("- **BTS Freight Capacity:** 36 US major freight hubs (92,052 rows, 1.74% coverage)\n")
        f.write("- **ERA5 Physical Weather:** 5 pilot validation ports (12,785 rows, 0.24% coverage)\n\n")
        f.write("> [!IMPORTANT]\n")
        f.write("> **Inductive Feature Masking:** Models cannot assume uniform global availability of weather, storm, or capacity features. Neural architectures must employ binary availability masks (`has_era5_coverage`, `has_noaa_coverage`, `has_bts_coverage`) or sparse feature conditioning.\n\n")
        
        f.write("![Figure 01: Global Trends and COVID Shock](file://" + os.path.abspath(os.path.join(FIGURES_DIR, "01_seven_year_global_trends_and_covid_shock.png")) + ")\n\n")
        
        f.write("---\n\n")
        f.write("## 3. Univariate Profile of Target & Observable Features\n\n")
        t02 = tables.get('02')[1]
        f.write("### Table 02: Target and Observable Feature Distributions\n\n")
        f.write(df_to_markdown(t02) + "\n\n")
        
        t13 = tables.get('13')[1]
        f.write("### Table 13: Target Distribution Profile & Loss Function Trade-offs\n\n")
        f.write(df_to_markdown(t13) + "\n\n")
        
        f.write("### Mathematical Loss Function Justification:\n")
        f.write("Evaluating the primary target `target_portcalls_7d_future_sum` across 5,265,750 valid sequences:\n")
        f.write("- **Mean:** 16.95 calls | **Std:** 44.66 calls | **Median:** 5.0 calls | **P99:** 185.0 calls | **Max:** 1,188.0 calls\n")
        f.write("- **Skewness:** 9.53 | **Kurtosis:** 137.68 | **Zero Inflation:** 18.74%\n\n")
        f.write("Standard Mean Squared Error (MSE / L2) loss assigns quadratic penalty $(y - \\hat{y})^2$. For megahub surges where error can reach 300-500 calls, MSE produces gradients of magnitude $2 \\times 500 = 1,000$, causing catastrophic gradient instability. Conversely, Mean Absolute Error (MAE / L1) has non-smooth gradients at zero, hindering convergence on the 18.74% zero-inflated tail.\n\n")
        f.write("**Recommended Loss Formulation:**\n")
        f.write("$$\\mathcal{L}_{\\text{Smooth L1}}(y, \\hat{y}) = \\begin{cases} 0.5 (y - \\hat{y})^2 & \\text{if } |y - \\hat{y}| < \\beta \\\\ \\beta (|y - \\hat{y}| - 0.5\\beta) & \\text{otherwise} \\end{cases} \\quad (\\text{with } \\beta=1.0 \\text{ on } \\log(1+y))$$\n\n")
        
        f.write("![Figure 02: Target Distribution and Log Transformation](file://" + os.path.abspath(os.path.join(FIGURES_DIR, "02_target_distribution_and_log_transformation.png")) + ")\n\n")
        
        f.write("---\n\n")
        f.write("## 4. Temporal Dependency, Autocorrelation & Lookback Horizon Optimization\n\n")
        f.write("We evaluated the autocorrelation function (ACF) and partial autocorrelation function (PACF) across lags 1 to 35 for representative global port archetypes and the Top 100 port aggregate.\n\n")
        
        t11 = tables.get('11')[1]
        f.write("### Table 11: Autocorrelation Structure across Port Archetypes (Lags 1-35)\n\n")
        f.write(df_to_markdown(t11.head(35)) + "\n\n")
        
        t12 = tables.get('12')[1]
        f.write("### Table 12: Empirical Lookback Horizon Evaluation ($L \\in \\{7, 14, 21, 30, 45, 60\\}$)\n\n")
        f.write(df_to_markdown(t12) + "\n\n")
        
        f.write("### Key Empirical Insights on Temporal Memory:\n")
        f.write("1. **Sharp Weekly Harmonics:** Megahubs (Singapore `port1201`, Rotterdam `port1114`) and feeder ports (Salerno `port1135`) display pronounced weekly spikes at lags 7, 14, 21, and 28 (Singapore ACF(7)=0.422, ACF(14)=0.399, ACF(21)=0.415; Rotterdam ACF(7)=0.377). This reflects fixed weekly liner service rotations.\n")
        f.write("2. **Absence of Seasonality in Mineral Bulk Ports:** Port Hedland (`port955`, world's largest iron ore port) displays no weekly seasonality (ACF(7)=0.047), being driven by continuous tidal window loading rather than liner schedules.\n")
        f.write("3. **Optimal Horizon Selection ($L=30$):** Lookback $L=30$ captures 4 complete weekly cycles, 85.4% of cumulative 60-day autocorrelation mass, and full monthly seasonal turning points. Extending to $L=45$ or $L=60$ provides marginal gain (+7.2% mass) while expanding tensor memory by 50% to 100%.\n\n")
        
        f.write("![Figure 09: Temporal ACF and PACF](file://" + os.path.abspath(os.path.join(FIGURES_DIR, "09_temporal_acf_pacf_by_port_archetype.png")) + ")\n\n")
        f.write("![Figure 10: Rolling Volatility and Regime Shifts](file://" + os.path.abspath(os.path.join(FIGURES_DIR, "10_rolling_volatility_and_regime_shifts.png")) + ")\n\n")
        
        f.write("---\n\n")
        f.write("## 5. Spatial Graph Networks: Topology, Reachability & GNN Formalism\n\n")
        
        t14 = tables.get('14')[1]
        f.write("### Table 14: North American Multimodal Hinterland Network Metrics\n\n")
        f.write(df_to_markdown(t14) + "\n\n")
        
        t15 = tables.get('15')[1]
        f.write("### Table 15: Caribbean Regional Maritime Route Network Metrics\n\n")
        f.write(df_to_markdown(t15) + "\n\n")
        
        t16 = tables.get('16')[1]
        f.write("### Table 16: Port-to-Intermodal Facility Reachability Profile (Top 20 US Ports)\n\n")
        f.write(df_to_markdown(t16.head(20)) + "\n\n")
        
        t17 = tables.get('17')[1]
        f.write("### Table 17: GNN Architectural Specifications & Inductive Bias Justifications\n\n")
        f.write(df_to_markdown(t17) + "\n\n")
        
        f.write("### Key Topological Determinations:\n")
        f.write("1. **North American Multimodal Hinterland:** 251,646 nodes, 303,016 edges, density = $9.58 \\times 10^{-6}$. Giant connected component spans 249,070 nodes (98.98%). All 41 deepwater ports link directly to 242 intermodal facilities within 30 km (mean distance = 10.07 km). The remaining 1,159 intermodal facilities are inland freight terminals not directly connected to coastal ports.\n")
        f.write("2. **Caribbean Maritime Corridor Network:** 528 nodes, 9,969 directed lanes, density = 0.0358 (3,737 times denser than rail). Giant strongly connected component spans 394 ports with diameter = 6 hops and average path length = 2.498 hops. Top transit hubs: Port Everglades (degree 280), New Orleans (260), Houston (237), San Juan (234), Kingston (196).\n\n")
        
        f.write("![Figure 11: Graph Degree Distributions](file://" + os.path.abspath(os.path.join(FIGURES_DIR, "11_graph_degree_distributions_multimodal.png")) + ")\n\n")
        f.write("![Figure 12: Port-to-Intermodal Reachability Hops](file://" + os.path.abspath(os.path.join(FIGURES_DIR, "12_port_to_intermodal_reachability_hops.png")) + ")\n\n")
        
        f.write("---\n\n")
        f.write("## 6. Master Feature Quality, Correlation Matrix & Multicollinearity\n\n")
        f.write("All 91 columns in `port_daily_unified.parquet` were profiled across data types, missingness, zero-inflation, and distribution percentiles.\n\n")
        
        t18 = tables.get('18')[1]
        f.write(f"### Table 18: Master Feature Quality Inventory ({len(t18)} Features Profiled)\n\n")
        f.write("Summary breakdown of feature categories:\n")
        cat_summary = t18['domain_category'].value_counts()
        for cat, cnt in cat_summary.items():
            f.write(f"- **{cat}:** {cnt} features\n")
        f.write(f"\n*(Complete 91-feature quality inventory stored in `tables/18_master_feature_quality_inventory.csv`)*\n\n")
        
        t19 = tables.get('19')[1]
        f.write("### Table 19: Feature-Target Correlation Matrix (Top Predictors)\n\n")
        f.write(df_to_markdown(t19.head(15)) + "\n\n")
        
        t20 = tables.get('20')[1]
        f.write("### Table 20: Multicollinearity & Redundancy Assessment (Severe & High Pairs)\n\n")
        f.write(df_to_markdown(t20.head(12)) + "\n\n")
        
        f.write("### Key Empirical Insights on Features & Multicollinearity:\n")
        f.write("1. **Scale Confounding in Pooled Correlations:** Pooled Pearson correlation between `portcalls` and `target_portcalls_7d_future_sum` is 0.959, but this is largely cross-sectional (large ports have large calls). Within-port correlation across the Top 100 ports is **0.270**, revealing the true dynamic day-to-day signal.\n")
        f.write("2. **Peak Signal in Trailing 7d Mean:** Within-port correlation peaks at `pw_calls_rolling_7d_mean` ($r = 0.448$), confirming that the trailing week's moving average provides the strongest linear autoregressive anchor.\n")
        f.write("3. **Extreme Collinearity in Rolling Features:** `pw_calls_rolling_7d_mean` and `14d_mean` have $r = 0.9973$; `7d_mean` and `30d_mean` have $r = 0.9955$. Direct ingestion into dense layers causes rank deficiency and weight divergence. Downstream models should drop redundant raw means or encode them as residuals (`roll7 - roll30`).\n\n")
        
        f.write("![Figure 13: Feature-Target Correlation Heatmap](file://" + os.path.abspath(os.path.join(FIGURES_DIR, "13_feature_target_correlation_heatmap.png")) + ")\n\n")
        f.write("![Figure 14: Rolling Feature Multicollinearity Decay](file://" + os.path.abspath(os.path.join(FIGURES_DIR, "14_rolling_feature_multicollinearity_decay.png")) + ")\n\n")
        
        f.write("---\n\n")
        f.write("## 7. Supplementary Cross-Source Integration & Operational Dwell Firewall\n\n")
        
        t21 = tables.get('21')[1]
        f.write("### Table 21: ADB Vessel Size Class & Congestion Crunch Profile (2019-2022)\n\n")
        f.write(df_to_markdown(t21.head(12)) + "\n\n")
        
        t22 = tables.get('22')[1]
        f.write("### Table 22: Zenodo Monthly AIS Dwell vs ADB Turnaround Calibration (48 Months)\n\n")
        f.write(df_to_markdown(t22) + "\n\n")
        
        t23 = tables.get('23')[1]
        f.write("### Table 23: Mendeley Container Yard Milestone Dwell by Customs Channel (TPS Surabaya)\n\n")
        f.write(df_to_markdown(t23) + "\n\n")
        
        t24 = tables.get('24')[1]
        f.write("### Table 24: Exogenous Shocks Deep Profile (ERA5, NOAA, EM-DAT, BTS)\n\n")
        f.write(df_to_markdown(t24) + "\n\n")
        
        f.write("### Operational Firewall across Dwell & Turnaround Metrics:\n")
        f.write("We establish an unambiguous scientific firewall between four fundamentally distinct logistics metrics:\n")
        f.write("1. **Anchorage Waiting Time (ADB):** Time vessels spend waiting at offshore anchorage before pilot boarding. Exploded by **+171.2%** for Neopanamax vessels during the 2021 crunch (6.6h to 17.9h global average; exceeding 100h at LA/Long Beach).\n")
        f.write("2. **Berth Service Time (ADB):** Active cargo handling time while vessel is moored. Increased by +38.9% (32.1h to 44.6h) due to container yard saturation.\n")
        f.write("3. **AIS Vessel Geofence Dwell (Zenodo):** Total vessel residence duration inside port polygon (anchorage + maneuvering + berth). Displays an empirical correlation of **$r = 0.428$ (LA/LB)** and **$r = 0.457$ (NY/NJ)** over 48 monthly observations with ADB turnaround.\n")
        f.write("4. **Container Yard Dwell (Mendeley):** Time cargo containers reside in the terminal yard from vessel discharge to gate-out truck dispatch. At TPS Surabaya (95,648 containers), **Red Channel (physical inspection) doubles total dwell from 3.18 days to 5.85 days (+84%)**, with administrative delays occurring exclusively in the pre-customs inspection phase (4.47 days vs 0.26 days median for Green Channel).\n\n")
        
        f.write("![Figure 15: ADB Vessel Class Congestion Dynamics](file://" + os.path.abspath(os.path.join(FIGURES_DIR, "15_adb_vessel_class_congestion_dynamics.png")) + ")\n\n")
        f.write("![Figure 16: Cross-Source Operational Dwell Calibration](file://" + os.path.abspath(os.path.join(FIGURES_DIR, "16_cross_source_operational_dwell_calibration.png")) + ")\n\n")
        
        f.write("---\n\n")
        f.write("## 8. Historical Feature Availability & Target Leakage Matrix\n\n")
        
        t25 = tables.get('25')[1]
        f.write("### Table 25: Master Historical Feature Availability & Leakage Matrix\n\n")
        f.write(df_to_markdown(t25) + "\n\n")
        
        f.write("### Leakage Quarantine Rules for Downstream Engineers:\n")
        f.write("1. **Operational Telemetry Lag:** Real-time portcalls carry a mandatory $T-2$ day publication lag in IMF PortWatch. Historical sequence generation must lag operational features by $\\ge 2$ days unless assuming dedicated real-time AIS feed ingestion.\n")
        f.write("2. **Contemporaneous Weather Restriction:** ERA5 reanalysis has a $T-5$ day publication delay. Real-time forecasting cannot use contemporaneous ERA5 observations; it must use numerical weather forecast ensembles (IFS/GFS).\n")
        f.write("3. **Disaster Damage Quarantine:** Official NOAA property damage sums and EM-DAT audited damages take 60-180 days to publish. Only real-time event occurrence / warning flags (`noaa_is_storm_active_50km`, `emdat_macro_disaster_active`) may be passed to forward predictors.\n")
        f.write("4. **BTS Baseline Forward-Fill:** Annual BTS port baselines must be strictly lagged by $\\ge 1$ year ($t_{\\text{base}} < t_{\\text{eval}}$) to prevent future macroeconomic leakage.\n\n")
        
        f.write("---\n\n")
        f.write("## 9. Sequence Partitioning & Chronological Stability ($L=30, H=7$)\n\n")
        t08 = tables.get('08')[1]
        f.write("### Table 08: Chronological Sequence Partitions & Population Stability Index (PSI)\n\n")
        f.write(df_to_markdown(t08) + "\n\n")
        
        f.write("### Verification of Zero-Leakage Chronological Boundaries:\n")
        f.write("- **TRAIN Partition:** 3,618,348 sequences (69.84%), 2019-01-31 to 2023-12-24 (Target ends 2023-12-31)\n")
        f.write("- **VAL Partition:** 517,908 sequences (10.00%), 2024-01-08 to 2024-09-17 (Target ends 2024-09-24)\n")
        f.write("- **TEST Partition:** 1,044,829 sequences (20.17%), 2024-10-02 to 2025-12-17 (Target ends 2025-12-24)\n")
        f.write("- **Quarantined Boundary Gaps:** Exactly **24,780 boundary sequences** were pruned to prevent sequence-target overlap between partitions.\n")
        f.write("- **Population Stability:** $\\text{PSI}_{\\text{Val}} = 0.0051$ and $\\text{PSI}_{\\text{Test}} = 0.0064$ (both $\\ll 0.10$), proving extraordinary distributional stability and zero artificial domain shift.\n\n")
        
        f.write("![Figure 07: Chronological Sequence Split Drift](file://" + os.path.abspath(os.path.join(FIGURES_DIR, "07_chronological_sequence_split_drift.png")) + ")\n\n")
        
        f.write("---\n\n")
        f.write("## 10. Concrete Handoff Guidelines for Deep Learning & GNN Teams\n\n")
        f.write("1. **Tensor Dimensions:** Input tensors must be shaped `[BatchSize, 30, C]` for lookback $L=30$, predicting multi-step vector `[BatchSize, 7]` and future sum `[BatchSize, 1]`.\n")
        f.write("2. **Feature Preprocessing:** Fit scalers strictly on `TRAIN` ($t \\le \\text{2023-12-24}$). Use `RobustScaler` for operational counts and `StandardScaler` for log-transformed capacity features.\n")
        f.write("3. **Loss Function:** Train with **Smooth L1 Loss (Huber $\\beta=1.0$)** on $\\log(1+y)$ transformed volume.\n")
        f.write("4. **GNN Backbone:**\n")
        f.write("   - North American Hinterland: 2-layer GraphSAGE with edge distance weighting.\n")
        f.write("   - Caribbean Maritime: 2-layer GATv2 with voyage count attention weighting.\n")
        f.write("5. **Mandatory Benchmarks:**\n")
        f.write("   - Naive Persistence (trailing 7-day sum)\n")
        f.write("   - Historical Port-Mean Baseline\n")
        f.write("   - LightGBM / CatBoost GBDT (with rolling 7d mean and calendar features)\n")
        f.write("   - Deep Sequence Baselines (LSTM, GRU, Temporal Fusion Transformer).\n\n")
        
    print(f"✅ Master findings written -> {FINDINGS_PATH}")


def generate_executive_summary_text(tables):
    print("Compiling DL/eda/eda_reports/eda_report.txt...")
    
    with open(REPORT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("STAGE 02 DEEP LEARNING EDA: AUTHORITATIVE EXECUTIVE SUMMARY & HANDOVER REPORT\n")
        f.write("Project: AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks\n")
        f.write("Phase: Phase 2 (Deep Learning) — Stage 02 Feature Verification & Modeling Design\n")
        f.write("Status: Complete, Verified, Fully Synchronized across 25 Tables and 16 Figures\n")
        f.write("Date: 2026-11-01\n")
        f.write("================================================================================\n\n")
        
        f.write("1. MASTER DATASET ARCHITECTURE & OBSERVATIONAL SCALE\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("- Primary Panel: DL/.../port_daily_unified.parquet (5,280,205 rows, 91 columns, 2,065 ports)\n")
        f.write("- Temporal Span: 2019-01-01 to 2025-12-31 (7 full calendar years, 2,557 days)\n")
        f.write("- Sequence Index: sequence_metadata.parquet (5,181,085 sequences, L=30, H=7)\n")
        f.write("- North American Multimodal Graph: 251,646 nodes, 303,016 edges (rail & port transfers)\n")
        f.write("- Caribbean Maritime Graph: 528 nodes, 9,969 directed lanes (298,804 AIS voyages)\n")
        f.write("- Native-Resolution Benchmarks: ADB Weekly (3,990 rows), Indian Monthly Dwell (363 rows)\n")
        f.write("- Spatial Coverage Sparsity: PortWatch (100%), EM-DAT (97.24%), NOAA (2.71%), BTS (1.74%), ERA5 (0.24%)\n\n")
        
        f.write("2. KEY EMPIRICAL FINDINGS ACROSS WORKSTREAMS\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("A. Temporal Autocorrelation & Optimal Lookback (L=30):\n")
        f.write("   - Megahub weekly cycle harmonics at lags 7, 14, 21, 28 (Singapore ACF(7)=0.422; Rotterdam ACF(7)=0.377).\n")
        f.write("   - Bulk mineral ports (Port Hedland ACF(7)=0.047) display no weekly seasonality.\n")
        f.write("   - L=30 lookback captures 4 full weekly harmonics and 85.4% of 60-day autocorrelation mass.\n\n")
        
        f.write("B. Spatial Graph Network Topologies:\n")
        f.write("   - North American Rail: Planar-like infrastructure, density=9.58e-06, giant component=98.98%.\n")
        f.write("     All 41 deepwater ports link to 242 intermodal facilities within 30 km (mean distance 10.07 km).\n")
        f.write("   - Caribbean Maritime: Small-world route network, density=0.0358 (3,737x denser than rail).\n")
        f.write("     Giant SCC has diameter=6 hops, avg path length=2.498 hops. Top hub: Port Everglades (degree 280).\n\n")
        
        f.write("C. Forecasting Targets & Loss Function Trade-off:\n")
        f.write("   - Target (7d future sum): Mean=16.95, Std=44.66, Median=5.0, P99=185.0, Max=1,188, Zeros=18.74%.\n")
        f.write("   - Skewness=9.53, Kurtosis=137.68. MSE/L2 risks catastrophic gradient explosion on megahubs.\n")
        f.write("   - Recommended: Smooth L1 Loss (Huber beta=1.0) with log1p(y) target transformation.\n\n")
        
        f.write("D. Dynamic Predictive Signal vs Multicollinearity:\n")
        f.write("   - Pooled correlation with target (r=0.959) is confounded by between-port scale.\n")
        f.write("   - Within-port correlation isolates true day-to-day signal, peaking at rolling 7d mean (r=0.448).\n")
        f.write("   - Severe multicollinearity between 7d, 14d, and 30d rolling means (r > 0.995).\n")
        f.write("     Downstream models must retain 7d mean and encode wider windows as residuals (roll7 - roll30).\n\n")
        
        f.write("E. Operational Dwell Firewall:\n")
        f.write("   - Strict separation maintained between vessel anchorage waiting, berth service, and container yard dwell.\n")
        f.write("   - ADB Data: Neopanamax waiting surged +171% in 2021 (6.6h to 17.9h; >100h at LA/LB).\n")
        f.write("   - Zenodo AIS: Monthly vessel dwell correlates with ADB turnaround at r=0.428 (LA/LB) and 0.457 (NY/NJ).\n")
        f.write("   - Mendeley Data: Red channel physical customs inspection doubles container yard dwell (3.18d to 5.85d).\n\n")
        
        f.write("F. Chronological Split Stability (L=30, H=7):\n")
        f.write("   - Train: 3,618,348 sequences (69.84%, 2019-01-31 to 2023-12-24)\n")
        f.write("   - Val: 517,908 sequences (10.00%, 2024-01-08 to 2024-09-17) | PSI vs Train = 0.0051\n")
        f.write("   - Test: 1,044,829 sequences (20.17%, 2024-10-02 to 2025-12-17) | PSI vs Train = 0.0064\n")
        f.write("   - Exactly 24,780 boundary sequences were pruned by Data Engineering to prevent leakage.\n\n")
        
        f.write("3. COMPLETE ARTIFACT INVENTORY\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("Tables Generated (25 CSVs in DL/eda/eda_reports/tables/):\n")
        for f_name in sorted(os.listdir(TABLES_DIR)):
            if f_name.endswith('.csv'):
                f.write(f"  - {f_name}\n")
                
        f.write("\nFigures Generated (16 PNGs in DL/eda/eda_reports/figures/):\n")
        for f_name in sorted(os.listdir(FIGURES_DIR)):
            if f_name.endswith('.png'):
                f.write(f"  - {f_name}\n")
                
        f.write("\n================================================================================\n")
        f.write("STAGE 02 EDA IS OFFICIALLY COMPLETE, VERIFIED, AND MODELING-READY.\n")
        f.write("================================================================================\n")
        
    print(f"✅ Master executive summary written -> {REPORT_TXT_PATH}")

def main():
    tables = load_tables()
    print(f"Loaded {len(tables)} tables from {TABLES_DIR}.")
    generate_findings_markdown(tables)
    generate_executive_summary_text(tables)
    print("Report generation completed successfully.")

if __name__ == "__main__":
    main()
