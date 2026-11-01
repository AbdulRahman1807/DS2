# Global ERA5 Spatial Feature Governance Policy (v1.3 Remediated)

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Milestone:** Phase 2 Data Engineering — Final Feature Governance & Remediation  
**Date:** 2026-11-01 02:53:58 UTC  
**Version:** 1.3-Remediated (Operational Policy for Bulk ERA5 Extraction and Supervised ML)  

---

## 1. Executive Principles

This policy defines mandatory spatial feature governance rules for environmental data extraction and machine learning pipelines. It ensures broad global coverage without compromising feature fidelity or introducing data leakage.

### Core Governance Axioms
1. **Extraction Eligibility ≠ Modeling Eligibility:** Retrieving an ERA5 time series from the ECMWF API does not grant permission to use that series as an unconstrained feature in terminal dwelling time models.
2. **Primary vs. Conditional Eligibility:** Primary features (`ELIGIBLE_PRIMARY`) represent local atmospheric or nearshore forcing. Conditional features (`ELIGIBLE_CONDITIONAL`) represent distant offshore shelf conditions and require explicit indicator flags.
3. **Identity Confidence vs. Coordinate Confidence:** Confirming that a PortWatch record corresponds to a specific UN/LOCODE seaport does not imply that catalog coordinates are accurate or present.
4. **No Claim of Berth Conditions:** ERA5 wave products (0.5° resolution) resolve deep-water swell and wind-sea on open ocean shelves. Wave proxies never represent sheltered harbor basins, docks, or berths.

---

## 2. Spatial Feature Governance Matrix

| Governance Category | Port Count | Surface Extraction | Wave Extraction | Surface Dwell Modeling | Wave Dwell Modeling | Offshore Proxy Flag | Operational Feature Policy |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Direct Ocean Ports** | 509 | ELIGIBLE | ELIGIBLE | `ELIGIBLE_PRIMARY` | `ELIGIBLE_PRIMARY` | `False` | 0 km displacement. Theoretical 0.5° grid vertex is valid ocean. High spatial confidence. |
| **Nearby Coastal Proxies** | 205 | ELIGIBLE | ELIGIBLE | `ELIGIBLE_PRIMARY` | `ELIGIBLE_PRIMARY` | `False` | Displaced < 40 km to outer fairway / approach channel. Marine forcing on harbor approaches. |
| **Regional Marine Proxies** | 226 | ELIGIBLE | ELIGIBLE | `ELIGIBLE_PRIMARY` | `ELIGIBLE_CONDITIONAL` | `True` | Displaced 40–100 km on outer shelf. Mandatory `offshore_proxy_flag = True`; subject to ablation. |
| **Geographically Unrepresentative** | 21 | ELIGIBLE | ELIGIBLE (TRANSIT) | `ELIGIBLE_PRIMARY` | `INELIGIBLE` | `False` | Inland river ports (Quebec, Rosario, Nanjing). Wave features extracted solely for transit; masked for dwell models. |
| **Blocked Inland Ports** | 15 | ELIGIBLE | INELIGIBLE | `ELIGIBLE_PRIMARY` | `INELIGIBLE` | `False` | Enclosed river/lake ports (Amazon basin, Montreal, Santa Fe). Wave coordinates strictly null. |
| **Identity / Coordinate Uncertain** | 4 | INELIGIBLE | INELIGIBLE | `INELIGIBLE` | `INELIGIBLE` | `False` | Gary, Bay City, Monroe, Delfzijl. Erroneous coordinates nullified pending registry errata. |
| **Unmapped Ports** | 1,085 | INELIGIBLE | INELIGIBLE | `INELIGIBLE` | `INELIGIBLE` | `False` | Reference coordinates unavailable. Strictly preserved with null coordinates. |
| **TOTAL UNIVERSE** | **2,065** | **976 Eligible** | **961 Extractable** | **976 Primary** | **714 Primary + 226 Cond.** | **226 True** | **Complete, mutually exclusive partition.** |

---

## 3. Surface Extraction & Modeling Governance

- Exactly **976 ports** are approved for surface meteorological extraction across **936 unique 0.25° grid vertices**.
- Surface variables: `u10` (10m u-wind), `v10` (10m v-wind), `t2m` (2m temperature), `msl` (mean sea level pressure), `tp` (total precipitation).
- Geodesic distance from port coordinate to nearest 0.25° grid vertex is ≤ 18.31 km (mean 9.62 km).
- All 976 approved ports receive `surface_modeling_eligibility_status = 'ELIGIBLE_PRIMARY'`.
- All 1,089 unmapped and exception ports receive `INELIGIBLE` and are strictly excluded from queries to prevent sampling weather at false municipality centroids.

---

## 4. Wave Extraction & Modeling Governance

- Exactly **961 ports** are approved for wave extraction across **774 unique valid 0.50° ocean cells**.
- Wave variables: `swh` (Significant Wave Height), `mwp` (Mean Wave Period), `mwd` (Mean Wave Direction).
- **Modeling Eligibility Breakdown (940 ports total):**
  1. **Primary Wave Features (714 ports):** Comprises 509 direct ocean ports and 205 nearby coastal proxies (<40 km). Wave proxy confidence is `HIGH_DIRECT_OCEAN` and `MEDIUM_COASTAL_PROXY`. Represents approach fairway conditions.
  2. **Conditional Wave Features (226 ports):** Comprises regional shelf proxies (40–100 km displacement). Wave proxy confidence is `LOW_REGIONAL_FORCING`. Requires `offshore_proxy_flag = True` in all downstream models.
  3. **Transit-Only Features (21 ports):** Comprises deep inland river terminals (e.g. Quebec, Rosario, Nanjing, Baton Rouge). Extracted across 16 unique ocean cells exclusively for regional shipping lane analysis. Strictly masked (`wave_modeling_eligibility_status = 'INELIGIBLE'`) for port dwell models.
  4. **Bypassed / Ineligible Ports (1,104 ports):** Comprises 15 blocked inland ports, 4 exception ports, and 1,085 unmapped ports. All wave queries are bypassed; features are populated with nulls.

---

## 5. Downstream Feature Engineering & Missingness Policy

When training supervised models (CatBoost, Gradient Boosting, Deep Learning LSTM/Transformer architectures):
1. **Surface Features:** Fully populated for all 976 validated ports. Zero synthetic imputation allowed.
2. **Wave Features:**
   - **Tier 1 (714 ports):** Use raw `swh`, `mwp`, `mwd` directly.
   - **Tier 2 (226 ports):** Feed `swh`, `mwp`, `mwd` accompanied by binary feature `offshore_proxy_flag = 1.0`. Model must evaluate feature importance with and without proxy interaction.
   - **Tier 3 (1,125 ports):** Wave features must be masked (`NaN` with dedicated `wave_missing_flag = 1.0`). Under no circumstances should open-sea swell be attributed to river terminals (Quebec, Rosario, Nanjing).

---

## 6. Audit & Immutability Verification

All historical Phase 1 assets, frozen ERA5 datasets, canonical registry v1.2, and manifests v1/v2 remain strictly immutable and verified via SHA-256 cryptographic digests.