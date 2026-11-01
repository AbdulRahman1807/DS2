# Phase 2 — Parallel Implementation Report: Supplementary Data Engineering Pipelines

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Phase:** Phase 2 — Supplementary Data Consolidation, Profiling & Integration Readiness  
**Stage:** Parallel Pipeline Implementation & Model-Ready Asset Generation  
**Date:** 2026-11-01  
**Root Staging Directory:** `DL/data/staging/supplementary_acquisition/`

---

## 1. Executive Summary & Workstream Status Matrix

Four independent supplementary data engineering pipelines have been designed, executed, and validated in parallel. All generated tables are written in Apache Parquet format to dedicated subdirectories within the supplementary staging directory. 

Existing Phase 1 and Phase 2 canonical registries, spatial governance files, raw tables, and processed datasets remain completely untouched.

| Workstream | Operational Domain | Output Directory | Primary Output Artifacts | Record Count | Pipeline Status |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **WS 1** | NOAA Weather Disruption Features | `processed/noaa/` | `noaa_port_daily_shocks.parquet`<br>`noaa_event_port_matches.parquet` | 143,192<br>80,244 | **COMPLETED & VERIFIED** |
| **WS 2A**| North American Rail & Intermodal Graph | `processed/graphs/north_american/` | `rail_nodes.parquet`<br>`rail_edges.parquet`<br>`intermodal_nodes.parquet`<br>`port_intermodal_candidates.parquet` | 250,204<br>302,774<br>1,401<br>242 | **COMPLETED & VERIFIED** |
| **WS 2B**| Caribbean Observed Maritime Graph | `processed/graphs/caribbean/` | `caribbean_maritime_nodes.parquet`<br>`caribbean_maritime_edges.parquet` | 528<br>9,969 | **COMPLETED & VERIFIED** |
| **WS 3** | EM-DAT Country Disaster Disruption | `processed/emdat/` | `emdat_country_daily_shocks.parquet`<br>`emdat_event_normalized.parquet` | 488,387<br>3,521 | **COMPLETED & VERIFIED** |
| **WS 4** | BTS vs. PortWatch Benchmark & Capacity | `processed/bts/` | `bts_portwatch_annual_comparison.parquet`<br>`bts_port_capacity_features.parquet` | 124<br>309 | **COMPLETED & VERIFIED** |

---

## 2. Workstream 1: NOAA Port-Level Disruption Features

### 2.1 Inputs Used
* Staged NOAA Storm Events: `raw/noaa_storm_events/StormEvents_details-ftp_v1.0_d*.csv.gz` (7 annual files, 2019–2025 Details, 478,175 total rows).
* Canonical Port Registry (`canonical_port_registry.csv`) and Spatial Governance (`spatial_feature_governance.csv`): 56 US canonical ports with verified accepted coordinates.

### 2.2 Output Artifacts
* `processed/noaa/noaa_port_daily_shocks.parquet` (143,192 rows, 14 columns, 279 KB)
* `processed/noaa/noaa_event_port_matches.parquet` (80,244 rows, 28 columns, 2.85 MB)
* `processed/noaa/noaa_feature_metadata.md`

### 2.3 Feature Definitions & Methodological Standards
* `storm_event_count_50km` / `storm_event_count_100km`: Frequency of severe convective, flood, wind, or storm events within 50 km and 100 km radius buffers.
* `is_storm_active_50km` / `is_storm_active_100km`: Daily binary exposure flags.
* `marine_wind_event_count`: Dedicated count of `Marine Thunderstorm Wind`, `Marine High Wind`, and `Marine Strong Wind` events within 50 km.
* `hurricane_event_flag`: Binary indicator for Hurricane, Tropical Storm, or Tropical Depression within 100 km.
* `flood_event_flag`: Binary indicator for Coastal Flood, Flash Flood, or Storm Surge within 50 km.
* `tornado_event_flag`: Binary indicator for Tornado, Funnel Cloud, or Waterspout within 50 km.
* `storm_damage_property_sum`: Parsed numerical sum of direct property loss in USD ($).
* `rolling_7d_storm_event_count`: Cumulative 7-day storm event frequency per port.
* `rolling_7d_storm_damage`: Cumulative 7-day estimated property damage per port.

### 2.4 Timezone and Spatial Handling
* **Timezone Offset Parsing:** `BEGIN_DATE_TIME` was parsed with `CZ_TIMEZONE` offsets (e.g. `CST-6` $\rightarrow +6\text{h}$, `EST-5` $\rightarrow +5\text{h}$) to establish true `begin_time_utc` and `date_utc`.
* **Geodesic Haversine Distance:** Vectorized spherical Haversine formula on EPSG:4326.
* **Coverage:** 100% daily continuity (2019-01-01 to 2025-12-31, 2,557 days per port) across 56 exposed canonical US ports. 0 duplicate `(portwatch_id, date_utc)` records.

---

## 3. Workstream 2: Spatial Graph & Hinterland Network Construction

### 3.1 Part A: North American Rail and Intermodal Graph
* **Inputs:** `raw/ntad_rail_network/` (302,774 arcs), `raw/ntad_intermodal/` (1,401 terminals), and Canonical Port Registry.
* **Outputs:**
  * `processed/graphs/north_american/rail_nodes.parquet` (250,204 rows, 8 columns)
  * `processed/graphs/north_american/rail_edges.parquet` (302,774 rows, 14 columns)
  * `processed/graphs/north_american/intermodal_nodes.parquet` (1,401 rows, 23 columns)
  * `processed/graphs/north_american/port_intermodal_candidates.parquet` (242 rows, 17 columns)
  * `processed/graphs/north_american/north_american_graph_metadata.md`
* **Topological Rigor:**
  * Built using explicit Federal Railroad Administration endpoint identifiers: `FRFRANODE` $\rightarrow$ `TOFRANODE`.
  * Preserves track mileage (`MILES`), track count (`TRACKS`), strategic corridor flags (`STRACNET`), Class I ownership (`RROWNER1`), and timezone.
  * Undirected physical track representation. Zero node synthesis or geometric endpoint snapping required.
* **Intermodal Connector Governance:**
  * 1,401 freight terminals evaluated for seaport proximity.
  * 242 candidate linkages identified within 30 km:
    * `VERIFIED_ADJACENT_MARINE` (78 links): Distance $\le 5$ km with marine vessel/barge transfer (`WATER == 'Y'`).
    * `STRONG_SPATIAL_CANDIDATE` (62 links): Distance $\le 10$ km with direct rail spur access (`RAIL == 'Y'`).
    * `METROPOLITAN_PROXIMITY_ONLY` (102 links): Proximity candidate only; physical rail link unconfirmed.

### 3.2 Part B: Caribbean Observed Maritime Graph
* **Inputs:** `raw/caribbean_ais/segments_port2port.csv` (9,969 segments), `ports.csv` (528 ports), `portcalls.csv`, and `crosswalks/caribbean_to_canonical_crosswalk.csv`.
* **Outputs:**
  * `processed/graphs/caribbean/caribbean_maritime_nodes.parquet` (528 rows, 16 columns)
  * `processed/graphs/caribbean/caribbean_maritime_edges.parquet` (9,969 rows, 20 columns)
  * `processed/graphs/caribbean/caribbean_graph_metadata.md`
* **Network Structure & Directionality:**
  * Preserves all 9,969 directed voyage corridors observed in 2019 AIS trajectories.
  * **1,179 Directed Edges** connect verified canonical seaports (e.g. Miami, Port of Spain, San Juan, Kingston, Caucedo).
  * 8,790 Directed Edges preserve regional island feeder and bunker/lightering topology.
  * Edge weights carry empirical voyage counts across 7 distinct vessel categories (`voyages_container`, `voyages_tanker`, `voyages_dry_bulk`, `voyages_general_cargo`, `voyages_cruise`, `voyages_service`, `voyages_island_feeder`).

---

## 4. Workstream 3: EM-DAT Country-Level Disruption Features

### 4.1 Inputs Used
* Staged EM-DAT Archive: `raw/emdat/260430_emdat_archive.xlsx` (26,946 global disaster records).
* PortWatch Country ISO3 reference codes.

### 4.2 Output Artifacts
* `processed/emdat/emdat_country_daily_shocks.parquet` (488,387 rows, 12 columns, 588 KB)
* `processed/emdat/emdat_event_normalized.parquet` (3,521 rows, 21 columns, 125 KB)
* `processed/emdat/emdat_feature_metadata.md`

### 4.3 Feature Definitions & Methodological Standards
* `macro_disaster_active`: Binary flag (1 if $\ge 1$ disaster active in country on date).
* `active_disaster_count`: Count of concurrent disasters active within national borders.
* `natural_disaster_count` / `technological_disaster_count`: Disaggregated active event counts.
* `total_deaths` / `total_affected`: Daily sum across active disasters (strictly preserving NaN for unmeasured impacts).
* `log_affected_population`: $\log_{10}(1 + \text{total\_affected})$.
* `total_damage_usd`: Estimated economic damage converted to USD.
* `rolling_7d_disaster_count` / `rolling_30d_disaster_count`: Cumulative 7-day and 30-day disaster exposure counts.

### 4.4 Data Governance & Imputation Transparency
* Filtered to 3,521 disaster events initiating in the 2019–2025 project window across 191 ISO3 countries.
* Start and end dates normalized to UTC calendar dates. Imputation flags (`EXACT`, `MONTH_DEFAULT`, `MONTH_END_DEFAULT`) explicitly recorded in `emdat_event_normalized.parquet`.
* 100% daily panel completeness (191 countries $\times$ 2,557 days = 488,387 panel records).

---

## 5. Workstream 4: BTS vs. PortWatch Benchmarking & Capacity Features

### 5.1 Inputs Used
* Staged BTS Port Data: `raw/bts_port_data/Port_Data_20261101.csv` (8,386 records).
* Crosswalk: `crosswalks/bts_to_canonical_crosswalk.csv` (41 seaport mapping entries).
* PortWatch Daily Panel: `DL/data/raw/portwatch/Daily_Ports_Data.csv` (215,468 US daily observation rows, 2019–2023).

### 5.2 Output Artifacts
* `processed/bts/bts_portwatch_annual_comparison.parquet` (124 rows, 18 columns, 21 KB)
* `processed/bts/bts_port_capacity_features.parquet` (309 rows, 15 columns, 27 KB)
* `processed/bts/bts_benchmark_report.md`

### 5.3 Benchmarking Validation Results (2019–2023)
* **Total Waterborne Tonnage Correlation:** Pearson $r = \mathbf{0.8333}$ across overlapping port-year records, demonstrating exceptional structural agreement between PortWatch cargo estimates and USACE physical manifests.
* **Commercial Vessel Calls Correlation:** Pearson $r = \mathbf{0.5934}$ (MAE = 4,730 calls, RMSE = 10,584 calls).
  * *Discrepancy Analysis:* PortWatch AIS call counts are higher than BTS figures because PortWatch's automated geofence captures all commercial vessel maneuvers (anchoring, shifting, bunkering), whereas USACE records only formal dock arrivals for cargo operations.
* **Composite Port Reconciliation:**
  * *Los Angeles / Long Beach (`port664`):* BTS reports Long Beach (`4110`) and Los Angeles (`4120`) separately. Aggregating `Volume(4110) + Volume(4120)` achieves exact alignment with canonical composite node `port664`.
  * *New York / New Jersey (`port815`):* BTS reports the consolidated bi-state port district (`Port ID: 398`), matching canonical node `port815`.

### 5.4 Slowly Changing Port Capacity Features
* Baseline attributes established for all 38 resolved deepwater ports (2016–2023):
  * `annual_container_teu_baseline`, `annual_vessel_call_baseline`, `annual_cargo_short_tons_baseline`.
  * Derived expected daily baselines: `daily_expected_vessel_calls`, `daily_expected_teu`, `daily_expected_short_tons`.
  * Year-over-year capacity growth rates: `annual_teu_growth_pct`, `annual_tonnage_growth_pct`.
  * `port_national_rank` (US national ranking).

---

## 6. Integration Readiness & Architectural Next-Stage Sequence

```
                                  PHASE 2 MODELING ARCHITECTURE
                                  
   EXOGENOUS WEATHER SHOCKS                MACROECONOMIC SHOCKS             STRUCTURAL BASELINES
   ┌───────────────────────┐             ┌──────────────────────┐         ┌──────────────────────┐
   │ NOAA Daily Shocks     │             │ EM-DAT Daily Shocks  │         │ BTS Capacity Baselines│
   │ (143k port-day rows)  │             │ (488k country-day)   │         │ (309 port-year rows) │
   └───────────┬───────────┘             └──────────┬───────────┘         └──────────┬───────────┘
               │                                    │                                │
               │ Join: (portwatch_id, date)         │ Join: (ISO3, date)             │ Join: (port_id, year)
               ▼                                    ▼                                ▼
   ═══════════════════════════════════════════════════════════════════════════════════════════════
                               PORTWATCH GLOBAL DAILY TEMPORAL PANEL
                                (Dwell Time & Bottleneck Modeling)
   ═══════════════════════════════════════════════════════════════════════════════════════════════
               ▲                                                                     ▲
               │                                                                     │
   ┌───────────┴───────────────────┐                     ┌───────────────────────────┴───────────┐
   │ Caribbean Maritime Graph      │                     │ North American Hinterland Graph       │
   │ • 528 Ports                   │                     │ • 250,204 Rail Nodes                  │
   │ • 9,969 Directed Edges        │                     │ • 302,774 Rail Arcs                   │
   │ • 1,179 Inter-Canonical Links │                     │ • 1,401 Intermodal Transfer Hubs      │
   └───────────────────────────────┘                     └───────────────────────────────────────┘
```

### Concrete Implementation Sequence for Downstream Engineering

1. **Temporal Feature Merging (Panel Construction):**
   * Join `noaa_port_daily_shocks.parquet` to PortWatch US rows on `(portwatch_id, date_utc)`.
   * Join `emdat_country_daily_shocks.parquet` to global PortWatch rows on `(ISO3, date_utc)`.
   * Project `bts_port_capacity_features.parquet` across daily PortWatch rows by matching `(portwatch_id, year)`.
   * Construct the dynamic Congestion Pressure Ratio: $\frac{\text{PortWatch 14d Rolling Calls} / 14}{\text{BTS Daily Expected Calls}}$.

2. **Graph Neural Network (GNN) Ingestion:**
   * Load `rail_nodes.parquet` and `rail_edges.parquet` into PyTorch Geometric `HeteroData` or NetworkX as the continental hinterland layer.
   * Attach Tier-1 candidate linkages (`port_intermodal_candidates.parquet`, 78 verified links) to bind deepwater port vertices to Class I rail corridors.
   * Ingest `caribbean_maritime_edges.parquet` as a regional directed weighted maritime graph for empirical transit modeling.

3. **Model Validation Benchmarking:**
   * Validate long-term model predictions of annual port congestion against `bts_portwatch_annual_comparison.parquet`.

