# Data Engineering Phase — Complete Asset & Workflow Guide

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Phase:** Phase 2 — Final Data Engineering  
**Directory:** `DL/data engineer/`  
**Status:** Completed, Verified, and Archived for Downstream Execution  

---

## 1. Directory Structure

```text
DL/data engineer/
├── scripts/       # All 27 python acquisition, processing, feature engineering, and packaging scripts
├── manifests/     # All crosswalks, split manifests, join readiness matrices, and feature dictionaries
├── profiles/      # Initial profiling, dataset quality, compatibility, and inventory audits
├── reports/       # All architectural, implementation, lineage, schema, and handoff reports
├── registry/      # Canonical Port Registry (v1.3 final semantics) & Spatial Feature Governance
└── processed_data # Symlink to DL/data/staging/supplementary_acquisition/processed/
```

---

## 2. Core Scripts Inventory (`scripts/`)

### Stage 1: Port Identity Resolution & Spatial Governance
* `build_canonical_port_registry.py`: Resolves PortWatch, UN/LOCODE, and Upply seaports.
* `build_spatial_feature_governance.py`: Enforces geodesic distance thresholds and proxy classifications.
* `finalize_spatial_governance_semantics.py` & `remediate_spatial_governance_v1_3.py`: Applies v1.3 semantic standards.

### Stage 2: ERA5 Atmospheric & Oceanographic Feature Pipelines
* `build_global_era5_spatial_manifest.py` & `build_global_era5_spatial_manifest_v2.py`: Ocean-aware grid manifests.
* `extract_era5_port_hourly.py`: ARCO-ERA5 cloud ingestion.
* `aggregate_era5_daily.py` & `freeze_era5_daily_v2.py`: Vectorized daily environmental aggregation.

### Stage 3: Supplementary Acquisition & Intermediate Feature Pipelines
* `process_supplementary_datasets.py`: ADB AIS port traffic weekly and Indian DGQI monthly dwell ingestion.
* `run_workstream1_noaa.py`: Spatial matching of NOAA Storm Events (2019–2025) at 50 km / 100 km radii.
* `run_workstream2_graphs.py`: North American rail topology extraction and Caribbean AIS trajectory lane derivation.
* `run_workstream3_emdat.py`: CRED EM-DAT active disaster daily country panel generation.
* `run_workstream4_bts.py`: BTS annual container/cargo capacity baselines and PortWatch physical benchmarking.

### Stage 4: Unified Integration & Model-Ready Packaging
* `build_unified_temporal.py`: Merges PortWatch, ERA5, NOAA, EM-DAT, and BTS into `port_daily_unified.parquet` (5.28M rows, 91 columns).
* `build_temporal_sequences.py`: Generates 5.18M sliding sequences ($L=30, H=7$) with strict chronological split quarantine.
* `package_graph_ready.py`: Standardizes North American hinterland and Caribbean maritime graphs into production Parquet tables.
* `build_final_feature_dictionary.py`: Compiles the master 166-feature catalog.

---

## 3. Data Storage & Shared Assets

In accordance with repository organization standards, all large physical datasets remain in `DL/data/`:
* Raw datasets: `DL/data/raw/` and `DL/data/staging/supplementary_acquisition/raw/`
* Processed models-ready datasets: `DL/data/staging/supplementary_acquisition/processed/`
* Interim datasets: `DL/data/interim/`

Downstream engineers can access all processed datasets directly via the `DL/data engineer/processed_data/` symlink or directly under `DL/data/staging/supplementary_acquisition/processed/`.

