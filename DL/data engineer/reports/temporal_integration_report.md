# Temporal Feature Integration Report: Unified Global Daily Panel

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Phase:** Phase 2 — Final Data Engineering Stage  
**Pipeline:** Workstream A — Unified Temporal Feature Integration  
**Artifact Path:** `DL/data/staging/supplementary_acquisition/processed/unified_temporal/port_daily_unified.parquet`  
**Date:** 2026-11-01  
**Status:** Completed and Verified  

---

## 1. Executive Summary

Workstream A successfully unified the primary PortWatch global daily maritime panel with four independent supplementary datasets:
1. **ERA5 Reanalysis v2** daily port-level atmospheric and oceanographic observations.
2. **NOAA Storm Events** spatially matched severe weather shocks (2019–2025).
3. **EM-DAT** national macro disaster and economic shock indicators (2019–2025).
4. **BTS Port Data Catalog** structural capacity baselines, national rankings, and derived capacity pressure ratios.

The resulting dataset, `port_daily_unified.parquet`, contains **5,280,205 rows** across **91 columns**, representing an unbroken, balanced daily panel for **2,065 global commercial seaports** across 7 complete calendar years (**2019-01-01 to 2025-12-31**).

---

## 2. Temporal Join Architecture & Source Coverage Matrix

The integration enforces strict primary key semantics, temporal alignment, and explicit spatial governance rules:

```
                            PRIMARY DAILY TIMELINE (2,065 Global Ports)
                         PortWatch Daily Panel (5,280,205 rows, 2019–2025)
                                              │
           ┌──────────────────────────────────┼──────────────────────────────────┐
           │                                  │                                  │
           ▼                                  ▼                                  ▼
    ERA5 Daily v2                      NOAA Storm Events                 EM-DAT Macro Shocks
 5 Key Maritime Gateways              56 US Canonical Ports             150 Host Sovereign States
 (portwatch_id, date_utc)           (portwatch_id, date_utc)             (country_iso3, date_utc)
 [has_era5_coverage = 1]             [has_noaa_coverage = 1]             [has_emdat_coverage = 1]
           │                                  │                                  │
           └──────────────────────────────────┼──────────────────────────────────┘
                                              │
                                              ▼
                                 BTS Port Capacity Features
                                 39 US Deepwater Gateways
                                   (portwatch_id, year)
                                 [has_bts_coverage = 1]
```

### Coverage Summary Table (`temporal_join_coverage.csv`)

| Dataset Source | Join Key | Covered Entities | Total Entities | Entity Coverage (%) | Covered Rows | Total Rows | Row Coverage (%) | Temporal Coverage | Coverage Flag Column |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **PortWatch Daily (Base)** | `(portwatch_id, date_utc)` | 2,065 | 2,065 | 100.00% | 5,280,205 | 5,280,205 | 100.00% | 2019-01-01 to 2025-12-31 | `has_portwatch_coverage` |
| **ERA5 Daily Environmental v2** | `(portwatch_id, date_utc)` | 5 | 2,065 | 0.24% | 12,785 | 5,280,205 | 0.24% | 2019-01-01 to 2025-12-31 | `has_era5_coverage` |
| **NOAA Storm Events Shocks** | `(portwatch_id, date_utc)` | 56 | 2,065 | 2.71% | 143,192 | 5,280,205 | 2.71% | 2019-01-01 to 2025-12-31 | `has_noaa_coverage` |
| **EM-DAT Macro Disaster Shocks** | `(country_iso3, date_utc)` | 150 | 180 | 83.33% | 5,134,456 | 5,280,205 | 97.24% | 2019-01-01 to 2025-12-31 | `has_emdat_coverage` |
| **BTS Port Capacity Features** | `(portwatch_id, year)` | 39 | 2,065 | 1.89% | 92,052 | 5,280,205 | 1.74% | 2019 to 2025 | `has_bts_coverage` |

---

## 3. Explicit Missingness Governance & Imputation Principles

Downstream ML models require rigorous separation between **genuine physical zeros** (calm days) and **unmeasured missingness** (spatial or institutional non-coverage):

1. **NOAA Storm Events:**
   - For ports where `has_noaa_coverage == 1`, calm weather days with no storms are represented as explicit numeric `0` for count/flag features (`noaa_storm_event_count_50km = 0`, `noaa_flood_event_flag = 0`, `noaa_storm_damage_property_sum = 0.0`).
   - For non-US ports or unmapped composite ports where `has_noaa_coverage == 0`, all NOAA features are preserved strictly as `NaN`. Models must use `has_noaa_coverage` as an attention mask or gating indicator.

2. **EM-DAT Disaster Impacts:**
   - Active disaster event counts are 0 on disaster-free days for covered countries (`has_emdat_coverage == 1`).
   - However, economic damages (`emdat_total_damage_usd`), deaths (`emdat_total_deaths`), and affected populations (`emdat_total_affected`) are left strictly as **`NaN` when unmeasured or unreported**. They are **never imputed as zero**, avoiding severe downward bias in macro impact estimations.

3. **ERA5 Wave Proxies:**
   - Governed wave proxy classifications (`era5_wave_proxy_type`) are preserved:
     - `NEARBY_COASTAL_PROXY`: Wave grid point within 40 km of port coordinate (Seattle: 13.4 km, LA/LB: 18.5 km).
     - `CONDITIONAL_OFFSHORE_PROXY`: Wave grid point $>40\text{ km}$ offshore (Savannah: 57.6 km, Houston: 53.2 km, NY/NJ: 77.8 km).
   - All 2,060 non-ERA5 ports carry `has_era5_coverage = 0` with null meteorological features.

4. **BTS Capacity Features & Carry-Forward:**
   - BTS historical reporting covers 2016 through 2023.
   - For 2024 and 2025, the 2023 structural capacity baseline was forward-filled per port. This allows downstream models evaluated on 2024 (Validation) and 2025 (Test) to use structural capacity with zero forward leakage (since 2023 design capacity is fully known before 2024).
   - Provenance flags `is_bts_forward_filled` (1 for 2024–2025, 0 for 2019–2023) and `bts_baseline_year` track the exact reporting period.

---

## 4. Derived Bottleneck Indicator: Capacity Pressure Ratio

To capture congestion spikes relative to structural throughput limits, we formulated the **14-day Capacity Pressure Ratio**:

$$\text{PressureRatio}_{14d}(i, t) = \frac{\text{pw\_calls\_rolling\_14d\_mean}(i, t)}{\text{bts\_daily\_expected\_vessel\_calls}(i, y)}$$

Where:
- $\text{pw\_calls\_rolling\_14d\_mean}(i, t) = \frac{1}{14}\sum_{\tau=0}^{13} \text{portcalls}(i, t-\tau)$ is the trailing 14-day rolling mean vessel arrival rate.
- $\text{bts\_daily\_expected\_vessel\_calls}(i, y) = \frac{\text{annual\_vessel\_call\_baseline}(i, y)}{365}$ is the design arrival rate derived from USACE / BTS annual manifests.

### Operational Interpretation:
- **Ratio $\approx 1.0$:** Equilibrium operations; arrivals match design throughput.
- **Ratio $\in [1.2, 1.5]$:** Elevated congestion pressure; berth occupancy exceeds operational safety buffer.
- **Ratio $> 1.5$:** Acute supply chain bottleneck shock; severe queueing, anchoring, and dwell inflation.
- **Null Values:** Safe division leaves nulls for non-BTS ports or facilities without commercial vessel call baselines.

---

## 5. Dataset Quality & Primary Key Integrity Audit

An exhaustive validation check verified:
1. **Row Count Match:** $2,065 \text{ ports} \times 2,557 \text{ days} = 5,280,205 \text{ rows}$. Zero duplicate rows; zero omitted dates.
2. **Primary Key Uniqueness:** `df.groupby(['portwatch_id', 'date_utc']).size().max() == 1`.
3. **Temporal Ordering:** Monotonically increasing calendar order per port.
4. **Target Integrity:** Target lead vectors ($t+1$ to $t+7$) are non-null for all dates $t \le 2025\text{-}12\text{-}24$. The terminal 7 days ($t \in [2025\text{-}12\text{-}25, 2025\text{-}12\text{-}31]$) are cleanly null, preventing false target leakage.
5. **Physical Bounds:** Vessel calls, cargo tonnages, wind speeds, and wave heights strictly adhere to physical constraints ($x \ge 0$).

