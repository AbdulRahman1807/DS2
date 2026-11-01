# Spatial Feature Provenance Completeness Report — Final Registry v1.3

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Milestone:** Phase 2 Data Engineering — Spatial Provenance Audit  
**Date:** 2026-11-01 03:04:41 UTC  

---

## 1. Provenance Completeness Summary

An exhaustive audit of coordinate provenance was conducted across all 2,065 PortWatch ports. For every port with accepted coordinates, source datasets, raw coordinate representations, and applied transformations were verified:

- **Total Port Universe:** 2,065 ports.
- **Ports with Accepted Coordinates:** Exactly 976 ports (100% provenance completeness).
- **Ports without Accepted Coordinates:** Exactly 1,089 ports (1,085 unmapped + 4 exceptions; coordinates strictly null).
- **Fabricated Coordinates:** **ZERO** (0). No synthetic centroids or unverified coordinates were introduced.

---

## 2. Coordinate Provenance Breakdown

| Provenance Source Category | Port Count | Source Dataset Name | Raw Representation | Applied Transformation | Coordinate Validation Status |
|:---|:---:|:---|:---|:---|:---|
| **Upply Seaports Catalog** | 945 | `UPPLY-SEAPORTS.csv` | Semicolon-delimited decimal string (e.g. `35.6459465, 136.0556972`) | `DECIMAL_DIRECT_SEMICOLON_PARSED` | `VALID` |
| **UN/LOCODE Recovered** | 31 | `UNLOCODE CodeListPart*.csv` | 12-char DMS string (e.g. `3913N 00907E`) | `DMS_TO_DECIMAL_DEGREE` ($DD = D + M/60$) | `VALID` |
| **Upply Country Discrepancy** | 1 | `UPPLY-SEAPORTS.csv` | `12.15, -68.2667` | `DECIMAL_DIRECT_SEMICOLON_PARSED` | `VALID_COUNTRY_CONFLICT` (Kralendijk `port598`, BES/BQ/NL) |
| **Primary Missing Coordinates** | 2 | `UNLOCODE CodeListPart3.csv` | `null` | `COORDINATE_EMPTY_IN_SOURCE` | `MISSING_IN_PRIMARY_SOURCE` (Gary `port377`, Monroe `port2270`) |
| **Primary Source Typo** | 1 | `UNLOCODE CodeListPart2.csv` | `5231N 00656E` | `PRIMARY_SOURCE_TYPO_REJECTED_PENDING_ERRATA` | `INVALID_PRIMARY_SOURCE_TYPO` (Delfzijl `port285`) |
| **Multi-Source Conflict** | 1 | `CodeListPart3 / UPPLY` | Conflicting WI vs MI coords | `UNRESOLVED_MULTI_SOURCE_CONFLICT` | `SOURCE_CONFLICT_REQUIRES_MANUAL_REVIEW` (Bay City `port130`) |
| **Unmapped Primary Universe** | 1,085 | `NONE` | `null` | `NONE` | `UNMAPPED` |
| **TOTAL** | **2,065** | — | — | — | — |

---

## 3. Scientific Caution on Source Authority

Preserving source provenance does not imply that catalog coordinates are infallible ground truth. UN/LOCODE coordinates frequently represent municipal post office centroids, customs offices, or historic road intersections rather than commercial container berths. Upply seaports data similarly incorporates uncorrected upstream errors (such as the 1-degree latitude error in Delfzijl). All coordinates must be understood as *catalog-provided reference coordinates*, subject to spatial reanalysis resolution limitations (0.25° atmosphere, 0.50° ocean waves).