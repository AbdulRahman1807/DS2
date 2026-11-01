# Global ERA5 Spatial Feature Governance Policy

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Milestone:** Phase 2 Data Engineering — Final Feature Governance  
**Date:** 2026-11-01 02:37:26 UTC  
**Version:** 1.0 (Operational Policy for Bulk ERA5 Extraction)  

---

## 1. Purpose and Operational Principles

This policy establishes rigorous feature-governance rules for global ERA5 environmental extraction and downstream supervised deep learning models. It ensures broad global spatial coverage while preventing erroneous, distant, or fabricated coordinates from polluting port-level dwelling time prediction models.

**Core Rule:** *A port eligible for environmental extraction is NOT automatically eligible for supervised dwell-time modeling.* Extraction eligibility governs whether raw ERA5 time series should be retrieved; modeling eligibility governs whether those time series may be fed as local features into terminal dwell-time neural networks.

---

## 2. Category Breakdown and Governance Matrix

| Governance Category | Port Count | Surface Extraction | Wave Extraction | Surface Dwell Modeling | Wave Dwell Modeling | Governance Description |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Direct Ocean Ports** | 509 | ELIGIBLE | ELIGIBLE | YES | YES | Theoretical 0.5° grid vertex is in open water (0 km displacement). High spatial fidelity. |
| **Nearby Coastal Proxies** | 205 | ELIGIBLE | ELIGIBLE | YES | YES | Displaced < 40 km to adjacent outer fairway. Represents harbor approach marine forcing. |
| **Regional Marine Proxies** | 226 | ELIGIBLE | ELIGIBLE | YES | YES (FLAGGED) | Displaced 40–100 km on coastal shelf. Mandatory `offshore_proxy_flag=True`; subject to ablation. |
| **Distant / Disconnected River Proxies** | 21 | ELIGIBLE | TRANSIT ONLY | YES | **NO (MASKED)** | River terminals (Quebec, Rosario, Nanjing, Baton Rouge). Open-sea swell is masked for local berth models. |
| **Blocked Inland Ports** | 15 | ELIGIBLE | **INELIGIBLE (NULL)** | YES | **NO (NULL)** | Enclosed river/lake ports (Amazon basin, Montreal, Santa Fe). Wave coordinates strictly null. |
| **Identity / Coordinate Uncertain** | 4 | **INELIGIBLE (NULL)** | **INELIGIBLE (NULL)** | **NO** | **NO** | Gary, Bay City, Monroe, Delfzijl. Erroneous coordinates nullified pending registry errata. |
| **Unmapped Ports** | 1,085 | **INELIGIBLE (NULL)** | **INELIGIBLE (NULL)** | **NO** | **NO** | Primary source coordinates missing. Strictly retained with null values. |
| **TOTAL UNIVERSE** | **2,065** | **976 Eligible** | **940 Extractable** | **976 Eligible** | **714 Direct/Near + 226 Reg** | Complete, mutually exclusive partition of the PortWatch universe. |

---

## 3. Surface Extraction Policy

- Exactly **976 ports** are approved for surface meteorological extraction across **936 unique 0.25° grid vertices**.
- Surface extraction covers `u10`, `v10`, `t2m`, `msl`, `tp`.
- All 976 approved ports have confirmed, authoritative coordinates with geodesic distance to nearest surface vertex ≤ 18.31 km (mean 9.62 km).
- The 4 exception ports (Gary, Bay City, Monroe, Delfzijl) and 1,085 unmapped ports are strictly excluded from surface queries to prevent sampling weather at incorrect town centroids.

---

## 4. Wave Extraction Policy

- Exactly **961 ports** are approved for wave extraction across **748 unique valid ocean cells**.
- Wave variables cover `swh` (Significant Wave Height), `mwp` (Mean Wave Period), and `mwd` (Mean Wave Direction).
- **Direct Ocean (509 ports):** Directly sampled from theoretical vertices.
- **Nearby Coastal (205 ports):** Sampled from outer fairways (< 40 km).
- **Regional Marine (226 ports):** Sampled from outer shelf (40–100 km).
- **Transit Only (21 ports):** Displaced > 100 km or inland waterways. Extracted solely for maritime transit features; strictly excluded from port-level dwell modeling.
- **Blocked Inland (15 ports):** Wave queries bypassed; features set to `NaN`.
- **Identity Uncertain (4 ports) & Unmapped (1,085 ports):** Wave queries bypassed.

---

## 5. Dwell-Time Machine Learning Feature Governance

When training supervised models (e.g. CatBoost baseline, Deep Learning Sequence models):
1. **Surface Weather Features:** Eligible for all 976 validated ports.
2. **Wave Features:**
   - **Tier 1 (High Confidence):** 714 ports (509 direct + 205 nearby coastal). Primary wave features.
   - **Tier 2 (Regional Forcing):** 226 ports. Eligible only with `offshore_proxy_flag = 1.0`.
   - **Tier 3 (Masked):** All remaining 1,125 ports must have wave features masked (`NaN` or dedicated missingness indicator). Under no circumstances should open-sea swell be attributed to river terminals (Quebec, Rosario, Nanjing).

---

## 6. Audit Immutability

All underlying reference datasets, raw files, canonical registry v1.2, manifest v1/v2, and Phase 1 assets remain 100% immutable and verified via SHA-256 digests.