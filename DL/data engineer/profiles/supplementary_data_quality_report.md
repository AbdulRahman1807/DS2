# Phase 2 — Supplementary Data Quality Report

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Phase:** Phase 2 — Supplementary Data Consolidation, Profiling & Integration Readiness  
**Stage:** Deliverable — Comprehensive Empirical Data Quality Audit  
**Date:** 2026-11-01  
**Isolation Directory:** `DL/data/staging/supplementary_acquisition/`

---

## 1. Quality Audit Scope and Objectives

This quality report provides a rigorous empirical evaluation of the six staged supplementary datasets. The audit tests:
1. **Completeness & Missingness:** Missing value rates across all attributes.
2. **Key Integrity & Uniqueness:** Primary key uniqueness, foreign key validity, and duplicate record detection.
3. **Spatial Validity:** Coordinate presence, range validity, bounding box adherence, and spatial CRS consistency.
4. **Temporal Precision & Alignment:** Timestamp formats, local timezone offsets, date ranges, and frequency mismatches.
5. **Data Representation Anomalies:** Formatting inconsistencies (commas in numeric fields, mixed types, trailing whitespace).
6. **Downstream Modeling Risks & Mitigations:** Explicit failure modes and required guardrails for feature engineering.

---

## 2. Dataset-Specific Quality Audits

### 2.1 EM-DAT Global Disaster Database

#### Completeness & Missingness
* **Total Records:** 26,946
* **Fully Populated Attributes (0% Missing):** `DisNo.`, `Historic`, `Classification Key`, `Disaster Group`, `Disaster Subgroup`, `Disaster Type`, `ISO`, `Country`, `Subregion`, `Region`, `Start Year`, `End Year`, `CPI`.
* **High-Missingness Attributes:**
  * Exact Coordinates (`Latitude`, `Longitude`): **89.6% missing** (24,139 null rows).
  * Administrative Units (`Admin Units`, `GADM Admin Units`): **68.8% to 70.9% missing**.
  * Economic Damages (`Total Damage ('000 US$)`): **78.1% missing** (21,033 null rows).
  * Direct Casualties (`No. Injured`, `No. Homeless`): **67.2% to 89.9% missing**.
  * Exact Day of Onset (`Start Day`): **14.5% missing** (3,912 null rows).

#### Uniqueness & Referential Integrity
* `DisNo.` is 100% unique across all 26,946 records (0 duplicates).
* ISO country codes strictly follow ISO 3166-1 alpha-3 standards across 231 distinct entities.

#### Quality Anomalies & Mitigation
* **Reporting Threshold Bias:** EM-DAT enters events meeting at least one criterion: 10+ deaths, 100+ affected, state of emergency declaration, or call for international assistance. Smaller, localized port disruptions are omitted.
* **Temporal Padding:** For events missing `Start Day` or `End Day`, imputation must not assume day 1 or day 28/31 arbitrarily; downstream pipelines should default to month-level aggregation intervals.

---

### 2.2 BTS Port Data Catalog (`Port_Data_20261101.csv`)

#### Completeness & Missingness
* **Total Records:** 8,386
* **Zero-Null Columns (100% Complete):** `Cargo Type`, `Port ID`, `Port_Name`, `Reporting Year`, `State`, `Trade Type`, `Units`, `Percent Change`.
* **Incomplete Columns:**
  * `Port Ranking`: **38.1% missing** (3,193 nulls). This is by design: only top-ranked ports receive national rankings.
  * `Region`: **1.1% missing** (93 nulls). Occurs primarily in early years (2016–2018) for inland river port authorities.
  * `Volume`: **1.7% missing** (145 nulls). Occurs where specific commodity trade flows were suppressed or zero.

#### Data Type & Representation Anomalies
* **Numeric Fields Stored as Formatted Strings:**
  * `Port ID` contains thousands commas (e.g., `'4,820'`, `'2,393'`). Direct string joins without removing commas will fail.
  * `Volume` contains embedded commas (e.g., `'58,059.2'`). Direct numeric casting requires `str.replace(',', '').astype(float)`.
  * `Percent Change` contains string-encoded floating point numbers and null representations.

#### Key Uniqueness & Entity Consistency
* The composite natural key `['Port ID', 'Reporting Year', 'Cargo Type', 'Trade Type', 'Units']` has **zero duplicates**.
* `Port_Name` strings have state qualifiers appended (e.g. `'Kalama, WA Port of'`, `'Savannah, GA Port of'`). Normalization is required to match canonical names.

---

### 2.3 Caribbean AIS & Port Trajectory Dataset (14 Tables)

#### Delimiter & Format Verification
* **Critical Finding:** All 14 tables use the pipe (`|`) character as delimiter, not comma. Any pipeline reading these files with default CSV parsing settings fails immediately.

#### Individual Table Quality Breakdown
| Table Name | Total Rows | Null Rate Key Columns | Coordinate Completeness | PK / FK Integrity | Quality Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `ports.csv` | 528 | `locode`: 85.8% null | 100% EWKB geom | PK `id` unique | High (Geometry intact; UNLOCODE sparse) |
| `portcalls.csv` | 1,488,781 | `lat/lon`: 0.007% null | 99.993% valid lat/lon | `port` FK valid to `ports.id` | Exceptional (Micro-level event integrity) |
| `berths.csv` | 1,902 | 0.0% null | 100% EWKB geom | `portid` FK valid | High (Clean polygon footprints) |
| `areapresence.csv` | 35,652 | 0.0% null | Matrix (no coords) | `mmsi` valid | High (Complete 365-day presence) |
| `vesseltype_original.csv`| 12,476 | 0.0% null | N/A | `mmsi` valid | High (Single classification per MMSI) |
| `vesseltype_enriched.csv`| 2,643 | 0.0% null | N/A | `mmsi` valid | High (Probabilistic weights sum to 1.0) |
| `trajectories.csv` | 1,473,752 | 0.0% null | Chronological waypoints | `port_origin`, `dest` FKs valid | High (Clean sequential legs) |
| `segments_port2port.csv` | 9,969 | 0.0% null | 100% LineString geom | 100% valid origin/dest FKs | Exceptional (Zero self-loops; clean edge weights) |
| `segments_port2territory`| 7,425 | 0.0% null | 100% LineString geom | Valid port FKs | High (Directional corridor flows) |
| `docking.csv` | 3 | 0.0% null | 100% MultiPolygon geom| 3 distinct dock types | Reference standard |
| `port_traffic.csv` | 528 | `ma_*`: 32.6% to 82.4% null| N/A | `id` FK valid | High (Calls complete; waiting time sparse for minor types) |
| `ungt_country_portcalls`| 38 | Bulk types: 23% to 84% null| Country aggregate | ISO3 valid | Moderate (Commodity gaps in small island states) |
| `ungt_country_portstats`| 333 | `avg_teu`: 81.4% null | Country aggregate | ISO3 valid | Moderate (TEU reported only for container nations) |
| `ungt_cont_throughput` | 25 | 0.0% null | Country aggregate | ISO3 valid | High (Complete national macro TEUs) |

#### Geometric Integrity
* All geometries are encoded in PostGIS EWKB hex format (`0105000020E6100000...`).
* The embedded spatial reference system is EPSG:4326 (WGS 84 ellipsoid), verified by SRID flag `20E61000` (hex 0x10E6 = 4326).

---

### 2.4 NOAA Storm Events (2019–2025 Details)

#### Temporal Quality & Timezone Discrepancies
* **Total Records:** 478,175 rows across 7 annual files.
* **Timestamp Syntax:** `DD-MON-YY HH:MM:SS` (e.g. `12-JAN-19 14:30:00`).
* **Timezone Offset Audit:**
  * Source data does **not** use UTC. Timestamps reflect local standard time defined by `CZ_TIMEZONE`.
  * Major timezones: `CST-6` (46.5%), `EST-5` (33.7%), `MST-7` (12.5%), `PST-8` (5.6%), `HST-10` (0.7%), `AST-4` (0.5%), `AKST-9` (0.4%).
  * **Critical Integration Requirement:** Timestamps must be adjusted by their respective timezone offset before aggregating to UTC-based daily port activity panels. Failure to do so introduces a 4-to-10 hour temporal misalignment.

#### Spatial Completeness
* **Point-Level Events (Valid `BEGIN_LAT` / `BEGIN_LON`):** 285,805 records (59.77%).
  * Bounding Box: Latitude 13.3°N to 71.3°N; Longitude -178.2°W to -64.6°W.
  * Captures localized, high-impact events (tornados, marine thunderstorm winds, flash floods, convective storms).
* **Zone/County-Level Events (Missing `BEGIN_LAT`):** 192,370 records (40.23%).
  * These represent large-scale weather episodes (winter storms, excessive heat, blizzard, regional drought) reported at the county (`CZ_NAME`) or NWS forecast zone level.
  * Spatial joins for these events require county FIPS or state-level matching.

---

### 2.5 NTAD Intermodal Freight Facilities

#### Completeness & Quality
* **Total Facilities:** 1,401
* **Coordinate Completeness:** **100% complete** (0 nulls in `LATITUDE`, `LONGITUDE`, `x`, `y`).
* **Facility Identification:** `TERM_ID` is 100% unique and complete.
* **Mode Flag Completeness:**
  * `TRUCK`: 100% populated ('Y' or 'N').
  * `RAIL`: 100% populated ('Y' or 'N').
  * `WATER`: 100% populated ('Y' or 'N').
* **Commodity Indicators:** Refined products (99.7% complete), crude oil (98.5% complete), petrochemicals (98.8% complete). Biofuel/ethanol fields have higher missingness (43–46% nulls).
* **Spatial Range:** Covers mainland US, Alaska, Hawaii, and Puerto Rico. Coordinates are valid WGS84 coordinates.

---

### 2.6 NTAD North American Rail Network Lines

#### Topological Completeness & Integrity
* **Total Line Segments (Edges):** 302,774
* **Edge Primary Key (`FRAARCID`):** 100% unique (0 duplicates, 0 nulls).
* **Node Identifiers (`FRFRANODE`, `TOFRANODE`):**
  * `FRFRANODE`: 0 nulls across 302,774 rows.
  * `TOFRANODE`: 0 nulls across 302,774 rows.
  * **Topological Soundness:** 100% complete topological connectivity across 250,204 unique vertices.
* **Network Continuity:** Every arc has designated start and end nodes. Graph traversal algorithms (NetworkX, PyTorch Geometric, cuGraph) can ingest the network directly from `(FRFRANODE, TOFRANODE, MILES, RROWNER1)`.

#### Attribute Quality
* `MILES` and `KM`: 100% complete and strictly positive (Min: 0.0001 miles, Max: 41.2 miles per segment, Total: 264,650.9 miles).
* `NET` (Network Classification): 100% complete across 10 functional classes.
* `COUNTRY`: 100% complete (`US`, `CA`, `MX`).
* `RROWNER1`: 98.9% populated (3,434 nulls on minor industrial sidings).

---

## 3. Summary of Identified Data Anomalies & Mitigation Protocol

| Dataset | Anomaly / Limitation | Severity | Required Pipeline Mitigation |
| :--- | :--- | :---: | :--- |
| **Caribbean AIS** | Pipe delimiter (`\|`) across all tables | Critical | Pipelines must explicitly pass `sep='\|'` |
| **NOAA Storms** | Local timezones (`CZ_TIMEZONE`) | Critical | Parse with timezone offsets before daily UTC aggregation |
| **BTS Port Data** | Embedded commas in numeric columns | Medium | Clean strings with `str.replace(',', '')` before numeric conversion |
| **EM-DAT** | 89.6% missing coordinates | High | Restrict EM-DAT to country-temporal macro features; do not spatial-join |
| **BTS Port Data** | Disaggregated LA/LB; Aggregated NY/NJ | Medium | Use composite handling in crosswalk (`Volume(4110) + Volume(4120) = port664`) |
| **BTS Port Data** | 14 inland river districts present | Low | Exclude inland statistical areas from deepwater ocean models |
| **Caribbean AIS** | Regional scope (2019 Caribbean basin) | High | Treat graph edges as regional observed subgraph; do not impute globally |
| **NTAD Rail** | North America continental scope | High | Confine rail network graph to US, Canada, Mexico hinterlands |
| **NOAA Storms** | 40.2% non-point county storm events | Medium | Spatial radius matching applies only to 285k point coordinate records |

