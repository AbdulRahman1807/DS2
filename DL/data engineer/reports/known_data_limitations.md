# Known Data Limitations, Spatial Boundaries & Methodological Caveats

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Phase:** Phase 2 — Final Data Engineering Stage  
**Artifact Path:** `DL/data/staging/supplementary_acquisition/final_handoff/known_data_limitations.md`  
**Date:** 2026-11-01  
**Status:** Documented for Scientific Integrity  

---

## 1. Executive Summary & Epistemic Boundaries

This document provides downstream model engineers, data scientists, and research stakeholders with an unvarnished audit of known dataset limitations, spatial coverage boundaries, resolution discrepancies, and missingness mechanisms. 

Adhering to these caveats is mandatory to prevent spurious model correlations, invalid inductive biases, and out-of-distribution evaluation failures.

---

## 2. Spatial Coverage Boundaries & Geographic Heterogeneity

The unified panel spans 2,065 ports globally, but supplementary features operate across differing spatial footprints:

```
                            SPATIAL COVERAGE HIERARCHY
                            
   Population Domain          Entity Count   Share of Global Ports   Covered Geographies
   ───────────────────────────────────────────────────────────────────────────────────────
   PortWatch Primary          2,065 ports    100.00%                 Global (180 countries)
   EM-DAT Sovereign Shocks    150 countries   83.33% (97.2% rows)    Sovereign host states
   NOAA Storm Events           56 ports        2.71%                 United States coastal
   BTS Capacity Baselines      39 ports        1.89%                 United States deepwater
   ERA5 Reanalysis v2           5 ports        0.24%                 US Tier-1 Gateways
   Caribbean Maritime Graph   528 ports       25.57%                 Caribbean & Central Am
   North American Rail Net    250,204 nodes       —                  USA, Canada, Mexico
```

### 2.1 NOAA Storm Events Spatial Boundary
* **Limitation:** NOAA Storm Events covers only the territory of the United States and adjacent US territorial waters.
* **Caveat:** For the 2,009 non-US ports, NOAA features are strictly `NaN`. Models evaluated globally must treat NOAA as a regional shock feature.
* **Proximity Metric:** Spatial exposure is measured at 50 km and 100 km geodesic radii from port centroids. A storm within 50 km does not guarantee that terminal crane operations were halted; it indicates elevated meteorological exposure.
* **Damage Under-Reporting:** Property damage values (`noaa_storm_damage_property_sum`) represent localized estimates from National Weather Service storm surveys. Unsurveyed rural events or early-report figures frequently record $0 damage despite physical wind disruption.

### 2.2 ERA5 Atmospheric & Wave Proxy Governance
* **Limitation:** ERA5 daily reanalysis features are staged exclusively for 5 primary US container gateways: Houston (`port481`), Los Angeles / Long Beach (`port664`), NY/NJ (`port815`), Savannah (`port1170`), and Seattle (`port1175`).
* **Conditional Offshore Wave Proxies:** Because standard wave reanalysis grids resolve open ocean cells ($0.5^{\circ}$ resolution), wave grid points for semi-enclosed ports (e.g., Savannah at 57.6 km offshore, NY/NJ at 77.8 km offshore) represent **conditional offshore proxies** rather than harbor-basin swell. They capture deepwater ocean swell affecting vessel harbor approach rather than inner berth sloshing.

### 2.3 BTS Capacity Baseline Temporal Carry-Forward
* **Limitation:** BTS publishes port throughput statistics with a 2-year lag. Contemporaneous BTS data is available for 2016 through 2023.
* **Methodological Carry-Forward:** For 2024 and 2025, the latest known structural baseline (2023) is forward-filled. While terminal physical land area and channel depths change slowly over multi-year horizons, new berth expansions or crane installations commissioned in 2024–2025 are not reflected in the static baseline.
* **Provenance Tracking:** Downstream users can identify forward-filled rows via `is_bts_forward_filled = 1`.

---

## 3. Dwell Ground-Truth Reality: The Temporal Resolution Disparity

### 3.1 The Daily Ground-Truth Absence
* **Empirical Reality:** **There is no continuous, daily port-level dwell time ground truth dataset available across the global port population.**
* **Audited Dwell Resolutions:**
  - AIS Dwell: 5 US ports at **monthly** resolution (660 records).
  - Indian Ports Dwell: 12 ports at **monthly** resolution (363 records).
  - ADB Port Traffic: 30 ports at **weekly** resolution (3,990 records).
  - Mendeley Dwell: 1 Indonesian terminal at event-level (2017–2018).

### 3.2 Formal Prohibition Against Synthetic Interpolation
* **Do Not Interpolate:** Under no circumstances should monthly dwell averages be linearly interpolated, spline-smoothed, or step-forward repeated into synthetic daily series.
* **Methodological Harms:**
  1. *Lookahead Leakage:* Monthly averages incorporate observations from days at the end of the month, leaking future information into early-month days.
  2. *False Autocorrelation:* Step-forward repetition generates artificial autocorrelation ($\rho \approx 0.97$), producing misleadingly high $R^2$ scores in naive autoregressive models.
  3. *Distortion of Bottleneck Dynamics:* Real daily bottleneck shocks (e.g., 48-hour hurricane closures) are averaged away into smooth monthly trends.

### 3.3 Governed Modeling Role for Dwell
* Daily sequence models must forecast **daily vessel throughput** (`target_portcalls_7d_future_sum`).
* Dwell datasets must be reserved strictly for **macro-level calibration, validation, and bivariate regime analysis** (e.g., verifying that predicted 7-day throughput surges correlate with observed monthly dwell expansions).

---

## 4. Missingness Mechanisms (MCAR vs. MAR vs. MNAR)

Downstream modelers must understand the data generation mechanism behind missing values:

1. **Missing Completely at Random (MCAR):**
   - *Does not apply to our datasets.* Missingness is never purely random.
2. **Missing at Random (MAR) — Institutional & Geographic Boundaries:**
   - NOAA features for non-US ports (`has_noaa_coverage = 0`) are MAR conditioned on geographic country code.
   - BTS features (`has_bts_coverage = 0`) are MAR conditioned on US port status and national throughput ranking.
   - ERA5 features (`has_era5_coverage = 0`) are MAR conditioned on Tier-1 gateway governance selection.
3. **Missing Not at Random (MNAR) — Disaster Impact Reporting:**
   - In EM-DAT, `emdat_total_damage_usd`, `emdat_total_deaths`, and `emdat_total_affected` are frequently missing for smaller disasters or in developing nations with weak post-disaster survey infrastructure.
   - **Severe Pitfall:** Replacing missing damage values with 0 creates extreme downward bias and treats un-surveyed catastrophic events as harmless. These fields are preserved strictly as **`NaN`**.

