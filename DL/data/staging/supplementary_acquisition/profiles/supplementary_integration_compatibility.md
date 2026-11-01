# Phase 2 — Supplementary Integration Compatibility Assessment

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Phase:** Phase 2 — Supplementary Data Consolidation, Profiling & Integration Readiness  
**Stage:** 4 Deliverable — Integration Readiness and Pipeline Compatibility  
**Date:** 2026-11-01  
**Isolation Directory:** `DL/data/staging/supplementary_acquisition/`

---

## 1. Executive Summary & Existing Project Architecture

This report assesses the mathematical, architectural, and operational compatibility of the six supplementary datasets against existing Phase 1 and Phase 2 project assets:

1. **Canonical Port Registry:** 2,065 globally resolved port nodes (`canonical_port_registry.csv`).
2. **Spatial Feature Governance:** 2,065 governed spatial profiles with accepted coordinates, modeling eligibility, and proxy classifications (`spatial_feature_governance.csv`).
3. **PortWatch Global Daily Activity:** Micro-level daily time series (`Daily_Ports_Data.csv`) tracking container, tanker, and dry bulk port calls and import/export tonnages per `portid`.
4. **ERA5 Port-Level Environmental Series:** Governed daily atmospheric and oceanographic features (`era5_port_daily_v2.parquet`, 12,785 port-day records across 2019–2025).
5. **Dwell Time Datasets:** Monthly global AIS dwell time (2015–2025) and Mendeley terminal container dwell time (`Dataset of Dwelling Time Process Fix.xlsx`).

None of these existing assets may be altered. All integration bridges, candidate crosswalks, and edge tables reside strictly within `DL/data/staging/supplementary_acquisition/`.

---

## 2. Integration Compatibility Taxonomy (The 9 Classes)

Each acquired supplementary source is formally mapped into one or more of the nine functional integration classes:

```
                    ┌──────────────────────────────────────────────┐
                    │      Acquired Supplementary Datasets         │
                    └──────────────────────┬───────────────────────┘
                                           │
         ┌──────────────────┬──────────────┴─────┬──────────────────┐
         ▼                  ▼                    ▼                  ▼
┌─────────────────┐ ┌────────────────┐ ┌───────────────────┐ ┌───────────────┐
│ Directly        │ │ Identifier     │ │ Spatial           │ │ Temporal      │
│ Joinable        │ │ Reconciliation │ │ Matching          │ │ Aggregation   │
│ • Caribbean AIS │ │ • BTS Port     │ │ • NOAA Storms     │ │ • NOAA Storms │
│   UNLOCODE (20) │ │   Catalog (55) │ │ • NTAD Intermodal │ │ • EM-DAT      │
│ • NTAD Rail     │ │ • Caribbean    │ │ • Caribbean       │ │ • BTS Port    │
│   Topology (Arc)│ │   Ports (528)  │ │   Ports (311)     │ │   Tonnage     │
└─────────────────┘ └────────────────┘ └───────────────────┘ └───────────────┘
         │                  │                    │                  │
         ├──────────────────┴──────────────┬─────┴──────────────────┤
         ▼                                 ▼                        ▼
┌─────────────────┐             ┌────────────────────┐   ┌──────────────────┐
│ Temporal Model  │             │ Spatial Graph      │   │ External Shock / │
│ Features        │             │ Construction       │   │ Disruption Feat. │
│ • NOAA Daily    │             │ • NTAD Rail Graph  │   │ • NOAA Storms    │
│   Shock Flags   │             │   (302k arcs)      │   │   (Local radius) │
│ • EM-DAT        │             │ • Caribbean Edges  │   │ • EM-DAT         │
│   Country Macro │             │   (9.9k segments)  │   │   (Country macro)│
└─────────────────┘             └────────────────────┘   └──────────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
             ┌─────────────────────┐               ┌─────────────────────┐
             │ Benchmarking &      │               │ Incompatible /      │
             │ Validation Only     │               │ Insufficient Scope  │
             │ • BTS Annual TEU    │               │ • Global Rail (NTAD │
             │ • Caribbean UNCTAD  │               │   is N. America)    │
             │   National Stats    │               │ • Global AIS (Carib │
             └─────────────────────┘               │   is Regional 2019) │
                                                   └─────────────────────┘
```

### Class 1: Directly Joinable Sources
* **Caribbean AIS Ports with UNLOCODE:** 20 ports in `ports.csv` have valid UNLOCODEs that join directly with zero ambiguity to `canonical_port_registry.csv` on `locode == unlocode` (e.g. `TTPOS` -> Port of Spain, `USMIA` -> Miami, `BQKRA` -> Kralendijk).
* **NTAD North American Rail Lines:** Directly joinable to itself and to any terminal possessing an explicit `FRFRANODE` or `TOFRANODE` because node topology is 100% explicit in the source data.

### Class 2: Sources Requiring Identifier Reconciliation
* **BTS Port Data Catalog (`Port_Data_20261101.csv`):** Identifiers are USACE port numeric codes and state-suffixed strings (e.g., `'4,820'`, `'700'`, `'Baltimore, MD'`). Reconciled via explicit crosswalk `bts_to_canonical_crosswalk.csv`.
* **Caribbean AIS (`ports.csv` without UNLOCODE):** 508 ports in `ports.csv` lack UNLOCODE. Reconciled via a two-tier mechanism: (a) sub-5km spatial proximity against canonical coordinates, and (b) terminal/country string matching.

### Class 3: Sources Requiring Spatial Matching
* **NTAD Intermodal Freight Facilities:** 1,401 facility locations (WGS84 lat/lon) require spatial distance joins to find nearest canonical port nodes. 119 facilities lie within 10 km of US canonical ports; 192 lie within 25 km.
* **NOAA Storm Events:** 285,805 storm events possess exact `BEGIN_LAT` and `BEGIN_LON` coordinates. Joining to ports requires a spatial geodesic radius buffer (50 km or 100 km) around port coordinates.
* **EM-DAT Coordinates Subset:** Only 10.4% (2,807 events) possess coordinates for spatial radius joins; the remainder require country-level administrative matching.

### Class 4: Sources Requiring Temporal Aggregation / Alignment
* **NOAA Storm Events:** High-resolution local timestamps (`BEGIN_DATE_TIME` + `CZ_TIMEZONE`) must be converted to UTC timestamps and aggregated to daily grain (`YYYY-MM-DD`) to align with daily PortWatch panel observations.
* **EM-DAT Disasters:** Event durations span days to months. Must be projected onto daily time series as active duration flags or aggregated to monthly/annual macro-shock indicators.
* **BTS Port Data:** Annual observations (2016–2023) must be projected either as static annual capacity anchors or slowly-changing annual covariates across the 365 days of each corresponding year.

### Class 5: Sources Suitable for Temporal Model Features
* **NOAA Storm Extreme Weather Indicators:** Daily binary or count flags per port:
  * `is_storm_event_active` (radius < 50 km)
  * `marine_gale_warning_flag` (`EVENT_TYPE == 'Marine Thunderstorm Wind'`)
  * `cum_property_damage_buffer_7d` (rolling storm damage impact)
* **EM-DAT Country-Level Disruption Flags:**
  * `country_macro_disaster_active` (1 if natural disaster active in port's ISO3 country)
  * `disaster_affected_population_log` (measure of societal and infrastructure stress)

### Class 6: Sources Suitable for Spatial Graph Construction
* **NTAD North American Rail Network:** 302,774 edges and 250,204 vertices. Forms a complete, connected hinterland transport graph for North American logistics.
* **NTAD Intermodal Connectors:** Connects port vertices to rail vertices where `distance_to_port <= 10 km` and `WATER == 'Y'`.
* **Caribbean AIS Observed Maritime Edges (`segments_port2port.csv`):** 9,969 directed edges providing empirical maritime corridor weights (`number_all`, `number_cont`, `number_tank`, `number_bulk`).

### Class 7: Sources Suitable for External Disruption Features
* **NOAA Storm Events:** Localized acute physical shocks (hurricanes, convective storms, storm surges, high winds) impacting berth availability, channel navigation, and cargo handling.
* **EM-DAT:** Macroeconomic systemic shocks (major earthquakes, national floods, infrastructure destruction) altering regional trade demand and domestic freight supply.

### Class 8: Sources Suitable Only for Benchmarking or Validation
* **BTS Port Annual Volume / TEU / Calls:** Annual totals serve as ground-truth validation to cross-check whether daily PortWatch daily call sums and import/export volume aggregations accurately match official US DOT/USACE port statistics.
* **Caribbean UNCTAD Country Port Stats (`ungt_*.csv`):** National benchmarks for median turnaround time, average deadweight tonnage, and container TEU throughput. Used to validate regional micro-dwell aggregations.

### Class 9: Sources with Incompatible or Insufficient Coverage
* **NTAD Rail Network for Global Ports:** Confined strictly to the North American continent (US, Canada, Mexico). **Cannot be used as a global rail infrastructure layer.**
* **Caribbean AIS for Global Route Modeling:** Confined strictly to the Caribbean/Gulf basin in calendar year 2019. **Cannot be naively merged into global shipping networks as if it represented global traffic.**
* **EM-DAT for Point-Level Port Berth Flooding:** 89.6% missing coordinates precludes using EM-DAT as a high-precision point-level hazard detector.

---

## 3. Deep-Dive Integration Assessments

### 3.1 Reconciling BTS Port Data against the Canonical Registry

* **The Problem:** BTS data uses USACE port codes (`Port ID`) and administrative port authority district names (e.g. `2031` = `Houston Port Authority, TX`, `5700` = `Virginia, VA, Port of`). In PortWatch and the canonical registry, ports are identified by `portwatch_id` (e.g. `port481`, `port2066`).
* **The Resolution:** All 55 BTS port entities were audited and classified in `bts_to_canonical_crosswalk.csv`:
  * **38 Canonical Coastal Seaports:** 41 BTS entities map directly to 38 canonical seaport IDs (including secondary terminal codes).
  * **14 Inland River Districts Excluded from Ocean Seaport Models:** Waterway statistical districts (e.g., `Cincinnati-Northern KY`, `Huntington-Tristate`, `Pittsburgh`, `St. Louis`, `Mid-America Port Commission`) have no corresponding deepwater ocean container port in the canonical registry. They are formally tagged `INLAND_BENCHMARK_ONLY` to avoid corrupting coastal port models.
* **The LA/LB Multi-Entity Resolution:**
  * In BTS, `4110` (Long Beach) and `4120` (Los Angeles) are reported as two separate records.
  * In PortWatch and canonical registry, they are combined as `port664` (*Los Angeles-Long Beach*).
  * **Operational Rule:** When using BTS annual TEU or tonnage as a benchmark for `port664`, downstream pipelines must sum `Volume(4110) + Volume(4120)`.

### 3.2 Caribbean AIS Directed Graph Edges (`segments_port2port.csv`)

* **Edge Construction:** `segments_port2port.csv` contains 9,969 observed directed pairs. Every `port_origin` and `port_destination` was verified to be a valid foreign key to `ports.csv` `id`.
* **Mapping to Global Canonical Nodes:** Using `caribbean_to_canonical_crosswalk.csv`, origin and destination port IDs were joined to canonical `portwatch_id`:
  * **1,179 Directed Edges** directly connect verified canonical port nodes (e.g., Miami `port736` <-> Port of Spain `port989`, Nassau `port106` <-> San Juan `port1143`, Kingston `port568` <-> Caucedo `port222`).
  * **8,790 Directed Edges** connect regional island berths, oil transfer jetties, and anchorages.
* **Weight Structure:** Each edge possesses empirical voyage counts across 7 distinct vessel classes (`number_cont`, `number_tank`, `number_bulk`, `number_gcar`, `number_crui`, `number_iisl`, `number_serv`).
* **Critical Boundary Rule:** Downstream graph neural networks (GNNs) must treat this edge set as a **Regional Observed Subgraph (Caribbean/Gulf Basin, 2019)**. It must never be imputed globally or assumed to reflect post-2019 route structures without explicit domain caveat.

### 3.3 Connecting NTAD Rail and Intermodal Infrastructure to Port Nodes

* **Multimodal Stacking:** The NTAD rail network (`raw/ntad_rail_network/`) contains 302,774 arcs with 100% explicit topological connectivity across 250,204 nodes (`FRFRANODE` -> `TOFRANODE`).
* **Intermodal Terminal Bridge:** The 1,401 facilities in `raw/ntad_intermodal/` provide the physical bridge between deepwater ports and rail corridors:
  * **Tier 1 (Port Adjacent, < 5 km):** 78 facilities (e.g., Fort Lauderdale petroleum rack at Port Everglades, San Diego marine terminal rack).
  * **Tier 2 (Port Metropolitan, 5–15 km):** 62 facilities.
  * **Tier 3 (Hinterland Gateway, 15–30 km):** 52 facilities.
* **Tri-Modal Terminals:** 143 terminals possess both `RAIL == 'Y'` and `WATER == 'Y'`. These represent verified intermodal transshipment points where container or liquid freight transitions between ocean vessels and inland rail arcs.
* **Graph Integration Path:** 
  1. Seaport Node (`portwatch_id`) 
  2. $\xrightarrow{\text{geodesic connector (<5km)}}$ Intermodal Facility (`TERM_ID`) 
  3. $\xrightarrow{\text{rail spur}}$ Rail Node (`FRFRANODE`) 
  4. $\xrightarrow{\text{Class I Mainline}}$ Continental Hinterland Network.

### 3.4 Aligning NOAA and EM-DAT Disruptions with Daily PortWatch Observations

* **Temporal Alignment:**
  * NOAA Storm Events: Local timestamp (`BEGIN_DATE_TIME`) is parsed using `CZ_TIMEZONE` and normalized to `date_utc`.
  * PortWatch: Daily records are keyed on `(portid, date)` where `date` is UTC.
* **Spatial Alignment Buffer:**
  * A 50 km geodesic buffer around each US port's accepted coordinates (`spatial_feature_governance.csv`) captures direct terminal impacts (high winds, storm surges, marine thunderstorm events).
  * A 100 km buffer captures approach-channel and hinterland trucking disruptions.
* **EM-DAT Alignment:**
  * For the 2019–2025 project window, EM-DAT events are mapped to ports by matching `ISO` country code during the active interval `[Start Date, End Date]`.
  * Generates an exogenous macro-disruption feature matrix tracking country-level disaster intensity (fatalities, affected population, estimated damages).

### 3.5 Annual BTS Features: Static vs. Slowly Changing Dynamics

* **Finding:** BTS data has annual granularity (one observation per port per year).
* **Feature Role:** BTS volume, ranking, and commodity distributions cannot predict day-to-day container dwell spikes directly.
* **Recommended Architecture:**
  * **Slowly Changing Structural Feature:** Projected across daily rows for the corresponding year (`port_annual_teu_capacity`, `port_annual_vessel_calls`).
  * **Congestion Pressure Ratio:** Ratio of rolling 30-day PortWatch vessel calls to BTS annual average daily calls ($\frac{\text{PortWatch 30d Calls} / 30}{\text{BTS Annual Calls} / 365}$). When this ratio significantly exceeds 1.0, the port is operating above design throughput, signaling structural bottleneck risk.

