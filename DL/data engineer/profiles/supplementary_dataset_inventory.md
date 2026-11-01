# Phase 2 — Supplementary Dataset Inventory

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Phase:** Phase 2 — Supplementary Data Consolidation, Profiling & Integration Readiness  
**Stage:** 1 & 2 Deliverable — Inventory, Consolidation & Integrity Verification  
**Date:** 2026-11-01  
**Isolation Directory:** `DL/data/staging/supplementary_acquisition/`

---

## 1. Executive Summary

This inventory establishes the formal catalog of all newly acquired supplementary datasets staged for Phase 2 integration readiness. All files have been located, identified from actual physical contents, staged into isolated staging directories under `DL/data/staging/supplementary_acquisition/raw/`, verified for cryptographic integrity (SHA-256), and validated for programmatic readability.

No existing Phase 1 or Phase 2 scripts, models, canonical registries, raw datasets, or processed outputs have been modified or overwritten. Original source archives and files have been strictly preserved.

---

## 2. Dataset Classification & Physical Storage Layout

The supplementary datasets are isolated within the designated directory structure:

```text
DL/data/staging/supplementary_acquisition/
├── raw/
│   ├── emdat/
│   ├── bts_port_data/
│   ├── caribbean_ais/
│   ├── noaa_storm_events/
│   ├── ntad_intermodal/
│   └── ntad_rail_network/
├── profiles/
├── manifests/
├── crosswalks/
├── edges/
└── spatial/
```

### Acquired Supplementary Datasets vs. Existing Project Data

| Category | Dataset Name | Staged Location | Status & Role |
| :--- | :--- | :--- | :--- |
| **Supplementary 1** | EM-DAT Global Disasters | `raw/emdat/` | 8 files (archive, columns metadata, PDF documentation) |
| **Supplementary 2** | BTS Port Data Catalog | `raw/bts_port_data/` | 1 CSV table (`Port_Data_20261101.csv`, 8,386 records) |
| **Supplementary 3** | Caribbean AIS Dataset | `raw/caribbean_ais/` | 14 pipe-delimited CSV tables (ports, portcalls, berths, segments, etc.) |
| **Supplementary 4** | NOAA Storm Events | `raw/noaa_storm_events/` | 7 annual gzip CSV files (2019–2025 Details, 478,175 records) |
| **Supplementary 5** | NTAD Intermodal Facilities | `raw/ntad_intermodal/` | 1 CSV table (Pipeline/marine/rail terminals, 1,401 records) |
| **Supplementary 6** | NTAD Rail Network Lines | `raw/ntad_rail_network/` | 1 CSV table (North American rail arcs, 302,774 lines) |
| *Existing Phase 1* | Mendeley Dwell Time | `DL/data/raw/mendeley/` | Preserved untouched (`Dataset of Dwelling Time Process Fix.xlsx`) |
| *Existing Phase 1* | PortWatch Daily Data | `DL/data/raw/portwatch/` | Preserved untouched (`Daily_Ports_Data.csv`) |
| *Existing Phase 1* | Monthly AIS Dwell | `DL/data/raw/ais_dwell/` | Preserved untouched (`monthly_dwell_2015_2025.csv`) |
| *Existing Phase 1* | Upply & UNLOCODE Reference | `DL/data/reference/` | Preserved untouched (`UPPLY-SEAPORTS.csv`, `UNLOCODE`) |
| *Existing Phase 2* | Canonical Port Registry | `DL/reports/port_identity_resolution/` | Preserved untouched (2,065 canonical ports) |
| *Existing Phase 2* | Spatial Feature Governance | `DL/reports/port_identity_resolution/` | Preserved untouched (2,065 governed spatial profiles) |
| *Existing Phase 2* | ERA5 Port Daily Features | `DL/data/processed/era5/` | Preserved untouched (`era5_port_daily_v2.parquet`) |
| *Existing Phase 2* | ADB Port Traffic & India Dwell | `DL/data/staging/supplementary_acquisition/processed/` | Preserved untouched (Prior weekly/monthly supplementary aggregations) |

---

## 3. Comprehensive Supplementary File Inventory

The table below catalogs all 32 staged files comprising the six supplementary datasets, including physical size, SHA-256 cryptographic checksum, and record counts.

### 3.1 EM-DAT (Centre for Research on the Epidemiology of Disasters — CRED)
*Original Source Location:* `/Users/abdul/Downloads/doi-10.14428-dvn-i0ltph`  
*Staged Directory:* `DL/data/staging/supplementary_acquisition/raw/emdat/`

| Filename | Format | File Size (Bytes) | SHA-256 Checksum | Record Count | Description |
| :--- | :---: | :---: | :--- | :---: | :--- |
| `260430_emdat_archive.xlsx` | XLSX | 8,137,119 | `21c210e479623144d41fa16db322cb202611f71dfaa1aa0d39e6a0d249f056d3` | 26,946 | Complete global disaster database (1900–2025) |
| `260430_emdat_columns.csv` | CSV | 5,499 | `fd42225c139023507fb72a81387d8a9854747b0a79ec0bcf53dc1a1e0fbef9a6` | 47 | Formal schema definition and column descriptions |
| `01_data_structure_and_content.pdf` | PDF | 11,931,572 | `45b47f46193c0e6ce638bc283253cb281855e96f13b6326177b9fc284f1eb8ad` | N/A | Documentation: Structure, classifications, variables |
| `02_data_collection_protocols.pdf` | PDF | 1,297,274 | `c1c01f8b0d5e16a9c1c4161a49c636f33cf23a7fae42b26090e5015b63795b6c` | N/A | Documentation: Sourcing criteria and inclusion rules |
| `03_known_issues_and_limitations.pdf` | PDF | 2,433,269 | `7b3d98b46b64df12a2082269c530dc2dd755c3258525ea243f7a4773cbb7b884` | N/A | Documentation: Reporting biases, missingness protocols |
| `04_additional_resources_and_tutorials.pdf` | PDF | 6,237,563 | `2b54fabe5439210e255263158c894fa8ec1f99c0daaeeb0da94c3ce7d3e9c5f8` | N/A | Documentation: Methodological guides and API notes |
| `05_about_emdat.pdf` | PDF | 1,015,723 | `21d9c10fdda43b78cf61e93895318db5033fcb7ea14f5fe91f63ecdf73f443b7` | N/A | Documentation: Institutional overview and citation |
| `MANIFEST.TXT` | TXT | 494 | `20c159d4644dad007328bf143a9483321689650caad33c94f0e74f1df91a6e50` | 8 | Dataverse distribution manifest |

### 3.2 BTS Port Data Catalog (Bureau of Transportation Statistics)
*Original Source Location:* `/Users/abdul/Downloads/Port_Data_20261101.csv`  
*Staged Directory:* `DL/data/staging/supplementary_acquisition/raw/bts_port_data/`

| Filename | Format | File Size (Bytes) | SHA-256 Checksum | Record Count | Description |
| :--- | :---: | :---: | :--- | :---: | :--- |
| `Port_Data_20261101.csv` | CSV | 1,169,458 | `51e39a3f2db788c1c4df2e21b8fbf7934bb79743477161b9e2ae2d325fa1fa58` | 8,386 | Annual US port throughput, tonnage, TEUs, calls (2016–2023) |

### 3.3 Caribbean AIS & Port Trajectory Dataset (14 Tables)
*Original Source Location:* `/Users/abdul/Downloads/CSV/`  
*Staged Directory:* `DL/data/staging/supplementary_acquisition/raw/caribbean_ais/`

| Filename | Format | File Size (Bytes) | SHA-256 Checksum | Record Count | Description |
| :--- | :---: | :---: | :--- | :---: | :--- |
| `ports.csv` | CSV (`\|`) | 469,239 | `e1f82f254e0c46644f6918ff8855e71c45f44da2b2170311f92e0be30648c66e` | 528 | Port nodes, country names, quays, wharves, EWKB geom, locode |
| `portcalls.csv` | CSV (`\|`) | 207,163,959 | `1ca1fcfdb77cb63a992bb3f9cfb40090886a5b6f0eec8be0db10b23f81e3a6c9` | 1,488,781 | Individual vessel port calls, duration, start/end ts, lat/lon |
| `berths.csv` | CSV (`\|`) | 528,045 | `a2ddad93f640cbe3efb49673998782a6fcf74092473489e248fb3cf3b63ccb5f` | 1,902 | Port berth boundaries, numbers, in-port shares, polygon geom |
| `areapresence.csv` | CSV (`\|`) | 16,849,445 | `2f9f8fa2a6fdf9cf9fcbfca4f76269ea085fb03478832a89df881f1fc9715a51` | 35,652 | Daily presence indicator per vessel MMSI across 365 calendar days |
| `vesseltype_original.csv` | CSV (`\|`) | 263,386 | `b0ca67923485f838dd66120eef000e008d669389ea2bf4b5db4f3f4e24eb2913` | 12,476 | Raw vessel MMSI type classifications |
| `vesseltype_enriched.csv` | CSV (`\|`) | 71,648 | `714f3c7e7cae2be2feea36dfef674a9c84918e7b91bf575d314841961e9389ee` | 2,643 | Multi-category vessel MMSI classifications with probability share |
| `trajectories.csv` | CSV (`\|`) | 76,086,131 | `7d73feef0fa83c3e80e1ba1c32021665a3baad88086055d72f91dbf80bc9b51a` | 1,473,752 | Chronological trajectory points linking origin to destination |
| `segments_port2port.csv` | CSV (`\|`) | 1,203,870 | `c6ebcf7c99279148d44c4fa83f9827011d8da9ff265da498a44ec2f84b6dbe4b` | 9,969 | Directed inter-port edges with traffic weights by vessel type |
| `segments_port2territory.csv` | CSV (`\|`) | 915,947 | `e74f2ca5e6085a2197609eead906a21350a45da26b0ea262c5896ff0a1871a36` | 7,425 | Port-to-country maritime corridors by traffic type and direction |
| `docking.csv` | CSV (`\|`) | 791,300 | `f2868bfd046c8a7cbb0d1e57c6b412953d1000df837e2ab70a92543ee7d4d707` | 3 | Dock infrastructure boundary geometries (quay, jetty, wharf) |
| `port_traffic.csv` | CSV (`\|`) | 67,543 | `d720b00c26620f4c0ce3015a6b0c2688f8d689626388915494cb30bc064239e2` | 528 | Port-level annual summary: calls, unique vessels, gross tonnage |
| `ungt_country_portcalls.csv` | CSV (`\|`) | 1,870 | `a49aa523829023ef08ff944d5c9dd43c74eb7fc8aa58cf9f2b8743ae847ef2e7` | 38 | UNCTAD country-level port call statistics by vessel commodity class |
| `ungt_country_portstats.csv` | CSV (`\|`) | 18,275 | `f59990b7937b83d16279f187ee09e4693b4a259c9d03cbfb332314a5113d0773` | 333 | Country vessel stats: median dwell, average age, GT, DWT, TEU |
| `ungt_cont_throughput.csv` | CSV (`\|`) | 623 | `2d665f8d2ccf09ca88a531d054238e55e4e892c90c885994fef405c1d68a9b34` | 25 | UNCTAD annual container throughput (TEU) per Caribbean nation |

### 3.4 NOAA Storm Events (National Centers for Environmental Information)
*Original Source Location:* `/Users/abdul/Downloads/StormEvents_details-ftp_v1.0_d*.csv.gz`  
*Staged Directory:* `DL/data/staging/supplementary_acquisition/raw/noaa_storm_events/`

| Filename | Format | File Size (Bytes) | SHA-256 Checksum | Record Count | Temporal Coverage |
| :--- | :---: | :---: | :--- | :---: | :---: |
| `StormEvents_details-ftp_v1.0_d2019_c20260323.csv.gz` | CSV.GZ | 11,602,648 | `710fbba74d81f1fe9b78daef52e37946894c25d8869c6f2a361b9bf5fb078b67` | 67,864 | 2019-01-01 to 2019-12-31 |
| `StormEvents_details-ftp_v1.0_d2020_c20260323.csv.gz` | CSV.GZ | 10,444,606 | `565bf9d8858a706be2cff66cfa455ec353597b20340c242c1626cbe01524e4d5` | 61,281 | 2020-01-01 to 2020-12-31 |
| `StormEvents_details-ftp_v1.0_d2021_c20260323.csv.gz` | CSV.GZ | 10,563,953 | `5f05809756b5a37128cb59a68579ca63297a7395cc7913ea3504353cbcc75b11` | 61,389 | 2021-01-01 to 2021-12-31 |
| `StormEvents_details-ftp_v1.0_d2022_c20260625.csv.gz` | CSV.GZ | 11,817,849 | `f1db5cbcfb00f14d7a8fa538e1b59a60aaeb925565538eec7bf21fbe7e721d0a` | 69,887 | 2022-01-01 to 2022-12-31 |
| `StormEvents_details-ftp_v1.0_d2023_c20260323.csv.gz` | CSV.GZ | 12,888,092 | `a9b2bce8cc5d0824b2ee5f7918a3ea23cf82fae388d184497fa97950c05877c8` | 75,593 | 2023-01-01 to 2023-12-31 |
| `StormEvents_details-ftp_v1.0_d2024_c20260728.csv.gz` | CSV.GZ | 12,693,243 | `4d257a05ebf39df5d233e0700d1767be329705f41cb837136f4fc911f4cc06cb` | 69,801 | 2024-01-01 to 2024-12-31 |
| `StormEvents_details-ftp_v1.0_d2025_c20260819.csv.gz` | CSV.GZ | 12,585,209 | `b6892faae0545f2a2491106e2361b17b6dc1969df0d8e84775210c4bc1be4b06` | 72,360 | 2025-01-01 to 2025-12-31 |
| **NOAA Total** | — | **82,595,600** | — | **478,175** | **2019–2025 (Complete)** |

### 3.5 NTAD Intermodal Freight Facilities (Pipeline & Marine Terminals)
*Original Source Location:* `/Users/abdul/Downloads/NTAD_Intermodal_Freight_Facilities_Pipeline_Terminals_-3233202154821634717.csv`  
*Staged Directory:* `DL/data/staging/supplementary_acquisition/raw/ntad_intermodal/`

| Filename | Format | File Size (Bytes) | SHA-256 Checksum | Record Count | Description |
| :--- | :---: | :---: | :--- | :---: | :--- |
| `NTAD_Intermodal_Freight_Facilities_Pipeline_Terminals_-3233202154821634717.csv` | CSV | 626,554 | `aa49d37bbba0f0f46f393a55eb13a372134cba79d035414d682413e2f5f135b5` | 1,401 | Intermodal pipeline, rail, truck, marine bulk terminals |

### 3.6 NTAD North American Rail Network Lines
*Original Source Location:* `/Users/abdul/Downloads/NTAD_North_American_Rail_Network_Lines_-4617128124973485019.csv`  
*Staged Directory:* `DL/data/staging/supplementary_acquisition/raw/ntad_rail_network/`

| Filename | Format | File Size (Bytes) | SHA-256 Checksum | Record Count | Description |
| :--- | :---: | :---: | :--- | :---: | :--- |
| `NTAD_North_American_Rail_Network_Lines_-4617128124973485019.csv` | CSV | 43,654,251 | `ec63c7882dc857791823eb9798544d852a3a5fec7dbf222dfa6b25102a0a2df3` | 302,774 | Rail arcs with explicit node topology across US, CA, MX |

---

## 4. Verification and File Integrity Statement

1. **Staging Completeness:** All six datasets have been successfully copied from user download locations into their dedicated staging subdirectories.
2. **Readability:** All files were programmatically verified using Python 3.13 (`pandas`, `openpyxl`, `gzip`). Zero I/O errors or parse corruptions were encountered.
3. **Delimiter Audit:** The Caribbean AIS tables use pipe (`|`) delimiters. Standard comma parsers fail on these tables; all pipelines must specify `sep='|'`.
4. **Duplicate Detection:** `Port_Data_20261101.csv` was present in both `Downloads/` and `Downloads/CSV/`. These files were verified to be bitwise identical. It was staged into `raw/bts_port_data/` and excluded from `raw/caribbean_ais/` to prevent schema contamination.
5. **No Existing Assets Modified:** Existing project directories (`DL/data/raw/`, `DL/data/processed/`, `DL/reports/`) remain completely unmodified.

