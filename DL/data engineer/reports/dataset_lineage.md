# End-to-End Dataset Lineage and Transformation Architecture

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Phase:** Phase 2 — Final Data Engineering Stage  
**Artifact Path:** `DL/data/staging/supplementary_acquisition/final_handoff/dataset_lineage.md`  
**Date:** 2026-11-01  
**Status:** Audited & Fully Traced  

---

## 1. High-Level Data Lineage Architecture

The diagram below documents the complete transformation lineage from original raw source files through intermediate enrichment pipelines to the final model-ready datasets delivered for downstream machine learning and deep learning engineers.

```mermaid
flowchart TD
    subgraph S1["Raw Supplementary & Primary Sources"]
        R_PW["PortWatch Daily Panel<br/>Daily_Ports_Data.csv<br/>(5.7M rows, 2019-2026)"]
        R_CANON["Canonical Port Registry<br/>canonical_port_registry.csv<br/>(v1.3 final semantics)"]
        R_GOV["Spatial Governance<br/>spatial_feature_governance.csv<br/>(v1.3 final semantics)"]
        R_ERA5["ERA5 Daily v2<br/>era5_port_daily_v2.parquet<br/>(12.7K rows, 5 hubs)"]
        R_NOAA["NOAA Storm Events<br/>Details CSVs (2019-2025)<br/>(478K rows)"]
        R_EMDAT["EM-DAT Disasters<br/>emdat_public_2026.xlsx<br/>(3.5K records)"]
        R_BTS["BTS Port Catalog<br/>Port_Data_20261101.csv<br/>(8.4K records)"]
        R_CARIB["Caribbean AIS Dataset<br/>14 CSV Tables<br/>(Port calls, berths, trajectories)"]
        R_RAIL["NTAD Rail Network<br/>North_American_Rail_Lines.csv<br/>(302K edges)"]
        R_IM["NTAD Intermodal Facilities<br/>Intermodal_Freight_Facilities.csv<br/>(1.4K terminals)"]
    end

    subgraph S2["Intermediate Supplementary Pipelines"]
        P_NOAA["Workstream 1: NOAA Pipeline<br/>Geodesic 50km/100km radii<br/>noaa_port_daily_shocks.parquet"]
        P_EMDAT["Workstream 3: EM-DAT Pipeline<br/>Daily active disaster panel<br/>emdat_country_daily_shocks.parquet"]
        P_BTS["Workstream 4: BTS Pipeline<br/>Annual capacity baselines & benchmark<br/>bts_port_capacity_features.parquet"]
        P_CARIB["Workstream 2B: Caribbean Pipeline<br/>Directed regional lanes & empirical duration<br/>caribbean_maritime_edges.parquet"]
        P_HINT["Workstream 2A: Hinterland Pipeline<br/>Rail topology + 50km drayage connectors<br/>rail_edges & port_intermodal_candidates"]
    end

    subgraph S3["Final Model-Ready Deliverables"]
        U_TEMP["Workstream A & B:<br/>port_daily_unified.parquet<br/>(5.28M rows, 91 cols, 2019-2025)"]
        U_SEQ["Workstream C:<br/>sequence_metadata.parquet<br/>(5.18M sequences, L=30, H=7)"]
        U_SPLIT["Workstream C:<br/>temporal_split_manifest.csv<br/>(Train / Val / Test Quarantine)"]
        U_NAG["Workstream D:<br/>north_american_graph_nodes.parquet<br/>north_american_graph_edges.parquet<br/>(251.6K nodes, 303.0K edges)"]
        U_CRG["Workstream D:<br/>caribbean_graph_nodes.parquet<br/>caribbean_graph_edges.parquet<br/>(528 nodes, 9.9K directed lanes)"]
    end

    %% Source to Intermediate
    R_NOAA --> P_NOAA
    R_CANON --> P_NOAA
    R_GOV --> P_NOAA
    
    R_EMDAT --> P_EMDAT
    
    R_BTS --> P_BTS
    R_CANON --> P_BTS
    R_PW --> P_BTS
    
    R_CARIB --> P_CARIB
    R_CANON --> P_CARIB
    
    R_RAIL --> P_HINT
    R_IM --> P_HINT
    R_CANON --> P_HINT

    %% Intermediate to Unified Temporal
    R_PW --> U_TEMP
    R_ERA5 --> U_TEMP
    P_NOAA --> U_TEMP
    P_EMDAT --> U_TEMP
    P_BTS --> U_TEMP

    %% Unified Temporal to Sequences
    U_TEMP --> U_SEQ
    U_SEQ --> U_SPLIT

    %% Intermediate to Graph Ready
    P_HINT --> U_NAG
    P_CARIB --> U_CRG
```

---

## 2. Granular Stage-by-Stage Transformation Logic

### Stage 1: Primary Temporal Backbone Integration
* **Input:** PortWatch `Daily_Ports_Data.csv` (filtered to 2019-01-01 through 2025-12-31).
* **Sorting Operation:** Forced strict ascending sort by `(portwatch_id, date_utc)` to correct raw un-indexed ordering.
* **Balanced Grid Verification:** Exactly $2,065 \text{ ports} \times 2,557 \text{ days} = 5,280,205 \text{ rows}$.
* **Calendar Encoding:** Generated `day_of_week` (0=Mon, 6=Sun), `is_weekend` ({0,1}), and `day_of_year` (1–366).
* **Trailing Rolling Windows:** Vectorized transform over $[t-k+1, t]$ for $k \in \{7, 14, 30\}$ for total calls and container calls with `min_periods=1`.

### Stage 2: Supervised Future Target Formulation (Workstream B)
* **Lead Vector Construction:** Computed forward shifts $\text{shift}(-k)$ for $k \in \{1, \dots, 7\}$ per port.
* **Cumulative Horizon Aggregation:** 
  $$\text{target\_portcalls\_7d\_future\_sum} = \sum_{k=1}^7 \text{shift}(-k)$$
* **Leakage Quarantine:** Terminal 7 days of 2025 populated strictly as `NaN`.

### Stage 3: Supplementary Feature Merging & Governance Flags
1. **ERA5 Reanalysis v2 (`has_era5_coverage`):**
   - Left merged on `(portwatch_id, date_utc)`.
   - Populated for 5 primary US gateways (`port1170`, `port1175`, `port481`, `port664`, `port815`).
   - Wave distance classified into `NEARBY_COASTAL_PROXY` ($\le 40\text{ km}$) and `CONDITIONAL_OFFSHORE_PROXY` ($>40\text{ km}$).
   - All other 2,060 ports carry `has_era5_coverage = 0` with null environmental fields.
2. **NOAA Weather Shocks (`has_noaa_coverage`):**
   - Left merged on `(portwatch_id, date_utc)`.
   - Spatially matched severe storms within 50 km and 100 km radii for 56 US canonical ports.
   - Calm days for covered ports carry explicit integer `0` counts and flags; non-covered ports carry `NaN`.
3. **EM-DAT Macro Disaster Shocks (`has_emdat_coverage`):**
   - Left merged on `(country_iso3, date_utc)` across 150 matching sovereign nations.
   - Calm days carry 0 active events; unreported damages/deaths preserved strictly as `NaN`.
4. **BTS Structural Baselines & Capacity Pressure Ratio (`has_bts_coverage`):**
   - Left merged on `(portwatch_id, year)` for 39 US deepwater ports.
   - 2023 structural baseline carried forward across 2024–2025 per port (`is_bts_forward_filled = 1`).
   - Computed dynamic congestion metric:
     $$\text{bts\_capacity\_pressure\_ratio\_14d} = \frac{\text{pw\_calls\_rolling\_14d\_mean}}{\text{bts\_daily\_expected\_vessel\_calls}}$$

### Stage 4: Sliding-Window Indexing & Partition Quarantine (Workstream C)
* **Window Size:** $L = 30$ days lookback, $H = 7$ days forecast horizon (37 days total).
* **Partition Split Rules:**
  - **Train:** Target ends $\le 2023\text{-}12\text{-}31$ (3,696,350 sequences, 71.34%).
  - **Validation:** Target strictly in $2024\text{-}01\text{-}01$ to $2024\text{-}12\text{-}31$ (743,400 sequences, 14.35%).
  - **Test:** Target strictly in $2025\text{-}01\text{-}01$ to $2025\text{-}12\text{-}31$ (741,335 sequences, 14.31%).
* **Boundary Transitions:** 24,780 boundary crossing sequences strictly excluded.

### Stage 5: Graph Topological Packaging (Workstream D)
* **North American Hinterland Network:**
  - 250,204 rail network junctions + 1,401 freight intermodal transfer terminals + 41 canonical US seaport complexes.
  - 302,774 physical rail lines + 242 geodesic intermodal transfer connectors ($\le 50\text{ km}$).
* **Caribbean Regional Maritime Shipping Graph:**
  - 528 commercial seaports spanning 38 Caribbean and Central American territories (including 18 canonical PortWatch gateways).
  - 9,969 empirical directed maritime shipping lanes with observed voyage volumes across vessel classes.

