# Phase 2 — Supplementary Dataset Profiles

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Phase:** Phase 2 — Supplementary Data Consolidation, Profiling & Integration Readiness  
**Stage:** 3 Deliverable — Comprehensive Empirical Data Profiling  
**Date:** 2026-11-01  
**Isolation Directory:** `DL/data/staging/supplementary_acquisition/`

---

## 1. Overview and Profiling Methodology

This report presents the empirical profiling of all six newly acquired supplementary datasets. Every schema, column, data type, missing-value rate, duplicate rate, coordinate distribution, and key relationship documented here was verified directly against the physical files.

None of the schemas or characteristics described below are assumed from documentation; all metrics reflect actual loaded file data.

---

## 2. EM-DAT: Global Disaster Events Database

### 2.1 File Inventory and Structural Characteristics
* **Physical File:** `raw/emdat/260430_emdat_archive.xlsx` (Sheet: `EM-DAT Data`)
* **Metadata Schema File:** `raw/emdat/260430_emdat_columns.csv`
* **File Format:** Microsoft Excel (.xlsx), OpenXML
* **Total Records:** 26,946 rows
* **Total Variables:** 47 columns
* **Primary Key:** `DisNo.` (26,946 unique, 0 nulls, 100% uniqueness)
* **Dataset Archetype:** Observational event time series (discrete socio-natural shocks)

### 2.2 Temporal and Geographic Coverage
* **Historical Temporal Range:** 1900 to 2025 (Start Year: 1900–2024, End Year: 1900–2025)
* **Project Window (2019–2025):** 3,521 events across 191 unique countries.
* **Geographic Extent:** Global (231 ISO country codes, 17 subregions, 5 macro regions).
* **Country Identifiers:** `ISO` (3-letter alpha code, 100% complete, 0 nulls) and `Country` (100% complete).

### 2.3 Disaster Classifications & Distributions
* **Disaster Groups:**
  * `Natural`: 17,367 events (64.5%)
  * `Technological`: 9,579 events (35.5%)
* **Disaster Subgroups:**
  * Hydrological: 6,896 | Transport: 6,352 | Meteorological: 5,627 | Geophysical: 1,943 | Miscellaneous: 1,638 | Biological: 1,611 | Industrial: 1,589 | Climatological: 1,289
* **Top Disaster Types (2019–2025 Project Window):**
  * Flood: 1,102 (31.3%)
  * Storm (Tropical cyclones, convective storms): 742 (21.1%)
  * Road Transport Accidents: 289 (8.2%)
  * Maritime/Water Transport Accidents: 236 (6.7%)
  * Extreme Temperature: 164 (4.7%)
  * Earthquake: 152 (4.3%)
  * Landslide / Mass Movement: 129 (3.7%)

### 2.4 Impact, Loss & Magnitude Metrics
* `Total Deaths`: 21,518 non-null (80.0% coverage; range: 1 to 300,000)
* `Total Affected`: 18,126 non-null (67.3% coverage; range: 1 to 330,000,000)
* `No. Injured`: 8,842 non-null (32.8% coverage)
* `No. Homeless`: 2,708 non-null (10.1% coverage)
* `Total Damage, Adjusted ('000 US$)`: 5,913 non-null (21.9% coverage; range: \$14 to \$254,000,000)
* `Magnitude` & `Magnitude Scale`: 5,265 non-null (Richter, Kph, Km², °C, etc.)

### 2.5 Coordinate Availability & Spatial Resolution Bottlenecks
* **Coordinate Null Rate:** 89.6% of records (24,139 rows) have null `Latitude` and `Longitude`.
* **Spatial Matching Implication:** Direct point-radius spatial matching against port coordinates is **not viable** for ~90% of global events.
* **Country-Level Temporal Mapping:** 100% of events possess a clean 3-letter `ISO` country code. Events can be reliably projected to port nodes via **Country-Temporal Shocks** (mapping disaster onset and duration to all port nodes belonging to that country's ISO3 code).

---

## 3. BTS Port Data Catalog

### 3.1 File Inventory and Structural Characteristics
* **Physical File:** `raw/bts_port_data/Port_Data_20261101.csv`
* **File Format:** Comma-Separated Values (CSV), ASCII/UTF-8
* **Total Records:** 8,386 rows
* **Total Variables:** 11 columns
* **Reporting Years:** 2016 through 2023 (Annual frequency)
* **Dataset Archetype:** Annual reference and ground-truth benchmark metrics

### 3.2 Schema & Inferred Data Types
| Column Name | Inferred Storage Type | Cleaned Type | Null Count | Description |
| :--- | :---: | :---: | :---: | :--- |
| `Cargo Type` | String | String Categorical | 0 | Cargo classification (Container, Dry Bulk, Total Tonnage, Vessel Calls, Commodities) |
| `Port ID` | String (with commas) | Integer (USACE Code) | 0 | USACE Waterborne Commerce Port Code (e.g. `'4,820'`, `'700'`) |
| `Port_Name` | String | String | 0 | Port name with state abbreviation (e.g. `'Baltimore, MD'`) |
| `Region` | String | String Categorical | 93 | Coastal waterbody region (Gulf, Atlantic, Pacific, Great Lakes) |
| `Reporting Year` | Integer | Integer | 0 | Calendar reporting year (2016–2023) |
| `State` | String | String | 0 | US State or multi-state jurisdiction |
| `Trade Type` | String | String Categorical | 0 | Trade direction/flow (Total, Imports, Exports, Domestic, Foreign, Empty) |
| `Units` | String | String Categorical | 0 | Measurement unit (`Short Tons`, `Container TEUs`, `Vessel Calls`) |
| `Port Ranking` | Float | Float | 3,193 | National rank for top-tier ports (unranked ports are null) |
| `Percent Change` | String (float literal) | Float | 0 | Year-over-year volume growth rate |
| `Volume` | String (with commas) | Float | 145 | Physical throughput volume in specified `Units` |

### 3.3 Entity Coverage & Natural Keys
* **Unique Port Entities:** 55 unique `Port ID` values across 54 distinct port names.
* **Natural Composite Primary Key:** `['Port ID', 'Reporting Year', 'Cargo Type', 'Trade Type', 'Units']` with **0 duplicates** across all 8,386 rows.
* **Throughput Metrics by Unit:**
  * `Short Tons`: 6,068 records (Min: 0, Median: 3,228,900, Max: 309,531,236 tons)
  * `Container TEUs`: 1,348 records (Min: 0, Median: 96,269, Max: 7,037,986 TEUs)
  * `Vessel Calls`: 970 records (Min: 0, Median: 237.75, Max: 39,735 calls)

### 3.4 Identifier Reconciliation & Data Quality Observations
* **Number Formatting:** `Port ID`, `Volume`, and `Percent Change` contain embedded commas (e.g. `'58,059.2'`, `'4,820'`) and must be stripped before casting to numeric types.
* **Coverage Scope:** Covers the top 55 US port entities. 41 entities are deepwater coastal/marine seaports; 14 entities represent inland river basins (Ohio River, Mississippi River barge districts).
* **Port Aggregation Discrepancies:**
  * BTS reports *Los Angeles, CA* (`Port ID: 4120`) and *Long Beach, CA* (`Port ID: 4110`) separately, whereas PortWatch aggregates them into composite node `port664` (*Los Angeles-Long Beach*).
  * BTS aggregates *New York and New Jersey* (`Port ID: 398`), matching PortWatch composite node `port815`.

---

## 4. Caribbean Port Calls and Vessel Trajectory Dataset (14 Tables)

### 4.1 Global Characteristics of the Caribbean Suite
* **Staged Directory:** `raw/caribbean_ais/`
* **Delimiter:** Pipe (`|`) across all tables.
* **Temporal Window:** Calendar Year 2019 (January 1, 2019 to December 31, 2019).
* **Geographic Extent:** Caribbean Basin, Gulf of Mexico, Florida Straits, Northern South America.
* **Coordinate System:** WGS 84 (EPSG:4326) with Well-Known Binary (WKB/EWKB) geometries.

### 4.2 Individual Table Profiles

#### 1. `ports.csv` (Port Master Registry)
* **Records:** 528 rows | **Columns:** 9
* **Schema:** `id` (PK, int), `portname` (str), `countryname` (str), `countrycode` (ISO3, str), `quay` (float), `wharf` (float), `offshore` (float), `geom` (hex EWKB MultiLineString/Polygon), `locode` (str)
* **Missingness:** `locode`: 453 nulls (75 populated); `quay`: 249 nulls; `offshore`: 524 nulls.
* **Role:** Node reference dictionary for all Caribbean AIS operations.

#### 2. `portcalls.csv` (Micro-Level Vessel Port Calls)
* **Records:** 1,488,781 rows | **Columns:** 13
* **Schema:** `id` (PK, int), `mmsi` (int), `port` (FK to `ports.id`, int), `ts_start` (timestamp), `ts_end` (timestamp), `nb_mes` (int), `time_bef` (int), `time_aft` (int), `duration` (dwell duration in seconds, int), `conf` (float), `latitude` (float), `longitude` (float), `geom` (hex EWKB Point)
* **Missingness:** `latitude`/`longitude`: 103 nulls (99.993% coordinate completeness).
* **Role:** High-frequency dwell time and turnaround ground-truth observations.

#### 3. `berths.csv` (Berth Polygons & Infrastructure)
* **Records:** 1,902 rows | **Columns:** 8
* **Schema:** `id` (PK, int), `portname` (str), `portid` (FK to `ports.id`, int), `countrycode` (str), `berth` (str), `number` (int), `inport` (float share), `geom` (hex EWKB MultiPolygon)
* **Missingness:** 0 nulls across all columns.
* **Role:** Intra-port spatial infrastructure boundaries.

#### 4. `areapresence.csv` (Daily Vessel Presence Matrix)
* **Records:** 35,652 rows | **Columns:** 367
* **Schema:** `id` (int), `mmsi` (int), `d_01_01` through `d_12_31` (binary presence indicator per calendar day for all 365 days of 2019)
* **Role:** Vessel regional persistence and availability indicator.

#### 5. `vesseltype_original.csv` (Vessel MMSI Type Registry)
* **Records:** 12,476 rows | **Columns:** 3
* **Schema:** `id` (int), `mmsi` (int), `vesseltype` (str: `bulk`, `cont`, `tank`, `gcar`, `crui`, `iisl`, `serv`)
* **Role:** Categorical vessel type mapping per MMSI.

#### 6. `vesseltype_enriched.csv` (Enriched Multi-Class Vessel Types)
* **Records:** 2,643 rows | **Columns:** 4
* **Schema:** `id` (int), `mmsi` (int), `vesseltype` (str), `share` (float probability / activity weight)
* **Role:** Refined vessel classification where vessel operational role spans multiple types.

#### 7. `trajectories.csv` (Sequential Vessel Trajectory Legs)
* **Records:** 1,473,752 rows | **Columns:** 7
* **Schema:** `id` (int), `idtraj` (int), `rankintraj` (int), `mmsi` (int), `port_origin` (FK, int), `port_destination` (FK, int), `ts` (timestamp)
* **Role:** Chronological waypoint sequence for path reconstruction.

#### 8. `segments_port2port.csv` (Observed Directed Graph Edges)
* **Records:** 9,969 rows | **Columns:** 12
* **Schema:** `id` (PK, int), `port_origin` (FK to `ports.id`, int), `port_destination` (FK to `ports.id`, int), `number_all` (total voyages, int), `number_bulk` (int), `number_cont` (int), `number_crui` (int), `number_gcar` (int), `number_iisl` (int), `number_serv` (int), `number_tank` (int), `geom` (hex EWKB LineString)
* **Topology:** 399 unique origin ports, 403 unique destination ports. 0 self-loops.
* **Traffic Weights:** `number_all` ranges from 1 to 14,812 voyages per directed pair.
* **Role:** Empirical directed graph edges with type-specific volume weights.

#### 9. `segments_port2territory.csv` (Port to Territory Corridors)
* **Records:** 7,425 rows | **Columns:** 13
* **Schema:** `id` (int), `port` (FK, int), `country` (ISO3, str), `direction` (int: 1=outbound, 2=inbound), traffic volumes by vessel type, `geom` (EWKB LineString)
* **Role:** International corridor traffic aggregations.

#### 10. `docking.csv` (Dock Type Spatial Footprint)
* **Records:** 3 rows | **Columns:** 3
* **Schema:** `id` (int), `docktype` (`quay`, `jetty`, `wharf`), `geom` (hex EWKB MultiPolygon)
* **Role:** Structural dock typology reference.

#### 11. `port_traffic.csv` (Port Traffic Annual Summary)
* **Records:** 528 rows | **Columns:** 43
* **Schema:** `id` (FK to `ports.id`), `portname`, `country`, port calls (`pc_*`), unique vessels (`uv_*`), vessel kilometers (`vk_*`), total turnaround time (`tt_*`), mean waiting time (`ma_*`) across all 7 vessel types.
* **Role:** Static port capacity and annual activity features.

#### 12. `ungt_country_portcalls.csv` (UNCTAD Country Port Calls)
* **Records:** 38 rows | **Columns:** 11
* **Schema:** `country`, `abbr` (ISO3), `allships`, `liq_bulk`, `liq_petroleum_gas`, `liq_natural_gas`, `dry_bulk`, `dry_breakbulk`, `roro`, `container`, `passenger`
* **Role:** National benchmark calls by UNCTAD commodity type.

#### 13. `ungt_country_portstats.csv` (UNCTAD Vessel Operating Stats)
* **Records:** 333 rows | **Columns:** 11
* **Schema:** `country`, `abbr`, `vesseltype`, `med_time` (median port days), `avg_age`, `avg_gt`, `max_gt`, `avg_dwt`, `max_dwt`, `avg_teu`, `max_teu`
* **Role:** Country-level fleet characteristic benchmarks.

#### 14. `ungt_cont_throughput.csv` (UNCTAD Container TEU Totals)
* **Records:** 25 rows | **Columns:** 3
* **Schema:** `country`, `abbr`, `teu` (annual container volume)
* **Role:** National macroeconomic container volume ground truth.

---

## 5. NOAA Storm Events (2019–2025 Details)

### 5.1 File Inventory and Structural Characteristics
* **Physical Files:** 7 files (`StormEvents_details-ftp_v1.0_d2019` to `d2025_*.csv.gz`)
* **File Format:** Gzip-compressed CSV (CSV.GZ)
* **Total Records:** 478,175 rows across 2019–2025
* **Annual Breakdown:**
  * 2019: 67,864 | 2020: 61,281 | 2021: 61,389 | 2022: 69,887 | 2023: 75,593 | 2024: 69,801 | 2025: 72,360
* **Total Variables:** 51 columns per year
* **Dataset Archetype:** High-impact discrete environmental shock events

### 5.2 Schema Highlights & Temporal/Spatial Variables
* **Event Identifiers:** `EVENT_ID` (unique storm event incident), `EPISODE_ID` (broader weather system)
* **Event Classifications (`EVENT_TYPE`):** Top categories include Thunderstorm Wind (134,047), Hail (60,028), High Wind (29,335), Flash Flood (29,226), Drought (26,778), Winter Storm (19,575), Flood (19,224), Marine Thunderstorm Wind (17,613), Tornado (11,169), Tropical Storm / Hurricane (~5,500).
* **Timestamps & Timezones:**
  * `BEGIN_DATE_TIME` & `END_DATE_TIME`: Formatted as `DD-MON-YY HH:MM:SS` (e.g. `12-JAN-19 14:30:00`).
  * `CZ_TIMEZONE`: Explicit local timezone indicator. **12 distinct timezones present:** `CST-6` (222,248), `EST-5` (161,051), `MST-7` (59,710), `PST-8` (26,704), `HST-10` (3,249), `AST-4` (2,456), `AKST-9` (1,973), `SST-11`, `GST10`, `PDT-7`, `CDT-5`, `EDT-4`.
  * **Critical Rule:** Source timestamps are strictly in local standard time; they must be parsed with `CZ_TIMEZONE` offsets before aligning to UTC-based PortWatch daily dates.

### 5.3 Coordinate Availability & Impact Metrics
* **Point Coordinates (`BEGIN_LAT`, `BEGIN_LON`, `END_LAT`, `END_LON`):**
  * Populated in 285,805 records (59.77% coordinate completeness).
  * Missing in 192,370 records (40.23% nulls, corresponding to county-wide phenomena like winter storms, excessive heat, and drought).
* **Spatial Matching Capability:** 285,805 point events can be dynamically queried via haversine distance buffers (e.g., 50 km or 100 km) around US coastal port coordinates.
* **Loss & Damage Metrics:** `DAMAGE_PROPERTY`, `DAMAGE_CROPS` (with magnitude suffixes `K`, `M`, `B`), `DEATHS_DIRECT`, `DEATHS_INDIRECT`, `INJURIES_DIRECT`, `INJURIES_INDIRECT`.

---

## 6. NTAD Intermodal Freight Facilities

### 6.1 File Inventory and Structural Characteristics
* **Physical File:** `raw/ntad_intermodal/NTAD_Intermodal_Freight_Facilities_Pipeline_Terminals_-3233202154821634717.csv`
* **Total Records:** 1,401 rows
* **Total Variables:** 35 columns
* **Primary Key:** `TERM_ID` (1,401 unique, 0 nulls)
* **Dataset Archetype:** Static infrastructure nodes (multimodal freight transfer hubs)

### 6.2 Spatial Coordinates & Geographic Distribution
* **Coordinates:** `LATITUDE`, `LONGITUDE` (along with projected `x`, `y` fields).
* **Coordinate Completeness:** 100% complete (0 nulls). Range: Lat 19.726°N to 68.843°N; Lon -176.644°W to -68.780°W.
* **Geographic Extent:** United States (all 50 states and territories).

### 6.3 Multimodal Connectivity Indicators
* `TRUCK`: 1,318 facilities ('Y' = 94.1%)
* `WATER`: 403 facilities ('Y' = 28.8%) — represents coastal, riverine, and deepwater marine barge/tanker terminals.
* `RAIL`: 331 facilities ('Y' = 23.6%) — represents direct rail-served pipeline and liquid bulk racks.
* **Tri-Modal Facilities (Water + Rail + Truck):** 143 hubs possess simultaneous marine, rail, and highway transfer infrastructure.

### 6.4 Commodity Capabilities
Indicators for handling bulk liquids and energy commodities: `CRUDE_OIL`, `REFINED`, `PET_CHEM`, `NGL`, `GASOLINE`, `DISTILLATE`, `BIODIESEL`, `JETFUEL`, `ETHANOL`.

---

## 7. NTAD North American Rail Network Lines

### 7.1 File Inventory and Structural Characteristics
* **Physical File:** `raw/ntad_rail_network/NTAD_North_American_Rail_Network_Lines_-4617128124973485019.csv`
* **Total Records:** 302,774 line segments (arcs)
* **Total Variables:** 34 columns
* **Edge Primary Key:** `FRAARCID` (302,774 unique, 0 nulls, 100% unique edge identifiers)
* **Dataset Archetype:** Static topological infrastructure network (intermodal rail graph)

### 7.2 Explicit Graph Topology
* **From Node Identifier:** `FRFRANODE` (202,358 unique nodes, 0 nulls)
* **To Node Identifier:** `TOFRANODE` (226,156 unique nodes, 0 nulls)
* **Total Unique Topology Nodes:** 250,204 distinct topological vertices.
* **Topology Completeness:** 100% explicit graph topology. **No geometric endpoint snapping or spatial clustering is required to assemble the rail network graph.**

### 7.3 Network Scope, Classification & Mileage
* **Total Physical Length:** 264,650.9 miles (425,915.2 km).
* **International Scope:**
  * United States (`US`): 235,703 arcs (77.8%)
  * Canada (`CA`): 64,539 arcs (21.3%)
  * Mexico (`MX`): 2,532 arcs (0.8%)
* **Top Class I Railroad Owners (`RROWNER1`):**
  * Canadian National (`CN`): 42,479 | Union Pacific (`UP`): 42,446 | BNSF Railway (`BNSF`): 35,108 | Canadian Pacific Kansas City (`CPKC`): 29,287 | CSX Transportation (`CSXT`): 28,458 | Norfolk Southern (`NS`): 27,841
* **Network Role Classification (`NET`):**
  * Mainline (`M`): 95,932 arcs (31.7%)
  * Other Track (`O`): 88,559 arcs (29.2%)
  * Yard Track (`Y`): 79,028 arcs (26.1%)
  * Industrial Spur (`I`): 16,130 arcs (5.3%)
  * Passing Siding (`S`): 10,193 arcs (3.4%)
  * Abandoned / Out of Service (`X`): 6,554 arcs (2.2%)
* **Strategic Rail Network (`STRACNET`):** 19,190 arcs designated Strategic Corridor (`S`); 3,928 designated Connector (`C`).

