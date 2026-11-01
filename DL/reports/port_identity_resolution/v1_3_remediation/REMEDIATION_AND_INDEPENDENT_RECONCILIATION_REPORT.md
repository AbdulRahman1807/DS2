# Phase 2 Spatial Governance v1.3 Remediation and Reconciliation Report

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Milestone:** Phase 2 Data Engineering — Spatial Governance v1.3 Remediation  
**Date:** 2026-11-01 02:53:58 UTC  
**Artifact Directory:** `reports/port_identity_resolution/v1_3_remediation/`  

---

## 1. Executive Summary

This remediation milestone successfully resolves all candidate-table semantics and spatial governance inconsistencies identified during expert review:
1. **Candidate Table Semantics Decoupled:** Candidate identity acceptance (`identity_candidate_accepted`) is now strictly independent from coordinate acceptance (`coordinate_candidate_accepted`).
2. **Explicit Candidate Records Added:** Gary (`USGYY`, row 18413), Monroe (`USMOI`, row 22610), and Bay City (`USBCY`, row 13506) are now explicitly represented in the candidate table, while retaining original rejected collision candidates (`USGT4`, `USM4X`).
3. **Delfzijl Candidate Alignment:** Delfzijl retains `NLDZL` as its accepted identity (`identity_candidate_accepted = True`), while its coordinate source is explicitly marked `coordinate_candidate_accepted = False` and `coordinate_validation_status = 'INVALID_PRIMARY_SOURCE_TYPO'` pending official UNECE errata.
4. **Bay City Source Conflict Preserved:** Bay City preserves `identity_candidate_accepted = False`, `coordinate_candidate_accepted = False`, and `accepted_unlocode = null` pending manual review.
5. **Categorical Governance Schema Implemented:** Governance dataset now features explicit categorical statuses (`ELIGIBLE_PRIMARY`, `ELIGIBLE_CONDITIONAL`, `INELIGIBLE`), `offshore_proxy_flag` (True for exactly 226 regional proxies), `wave_proxy_confidence`, `identity_confidence`, and `coordinate_validation_status`.
6. **Direct Grid-Cell Count Reconciliation:** Reconciled actual unique grid coordinates directly from the output rows: **936 unique surface cells** (for 976 ports) and **774 unique wave ocean cells** (for 961 extractable ports).

---

## 2. Candidate Table Remediation Audit

| PortWatch ID | Port Name | Candidate Source | Candidate ID | Identity Accepted | Coord Accepted | Coordinate Validation Status | Rejection Reason |
|:---|:---|:---:|:---:|:---:|:---:|:---|:---|
| `port377` | Gary | UNLOCODE | `USGYY` | **True** | **False** | `MISSING_IN_PRIMARY_SOURCE` | Correct commercial seaport Gary, IN (USGYY) confirmed, but coordinates empty in UNECE release. |
| `port377` | Gary | UNLOCODE / UPPLY | `USGT4` | **False** | **False** | `REJECTED_WRONG_LOCATION` | COLLISION_WRONG_STATE: Resolves to Panola County, Texas (USGT4), not Gary, Indiana. |
| `port2270` | Monroe | UNLOCODE | `USMOI` | **True** | **False** | `MISSING_IN_PRIMARY_SOURCE` | Correct commercial port Monroe, MI (USMOI) confirmed, but coordinates empty in UNECE release. |
| `port2270` | Monroe | UNLOCODE / UPPLY | `USM4X` | **False** | **False** | `REJECTED_WRONG_LOCATION` | COLLISION_WRONG_STATE: Resolves to Grafton County, New Hampshire (USM4X), not Monroe, Michigan. |
| `port285` | Delfzijl | UNLOCODE / UPPLY | `NLDZL` | **True** | **False** | `INVALID_PRIMARY_SOURCE_TYPO` | Official UNECE CodeListPart2 row 24631 records 5231N instead of 5319N (88.9 km south on farmland). Coordinate rejected pending errata. |
| `port130` | Bay City | UNLOCODE / UPPLY | `USYBZ` | **False** | **False** | `SOURCE_CONFLICT_REQUIRES_MANUAL_REVIEW` | Multi-source conflict: text LOCODE points to Bay City, WI, while commercial port is in Michigan. |
| `port130` | Bay City | UNLOCODE | `USBCY` | **False** | **False** | `SOURCE_CONFLICT_REQUIRES_MANUAL_REVIEW` | UNECE USBCY in Michigan lacks port code 1 (--34----), conflicting with text LOCODE US YBZ. |

---

## 3. Spatial Feature Governance Summary

### A. Complete Category Partition (2,065 Ports Total)
- **Surface Extraction Eligible:** Exactly **976 ports** mapped to **936 unique 0.25° grid vertices**.
- **Surface Extraction Ineligible:** Exactly **1,089 ports** (4 exceptions + 1,085 unmapped).
- **Surface Modeling Eligible:** Exactly **976 ports** (`ELIGIBLE_PRIMARY`).
- **Surface Modeling Ineligible:** Exactly **1,089 ports** (`INELIGIBLE`).

### B. Wave Extraction & Modeling Partition
- **Wave Extractable Ports:** Exactly **961 ports** mapped to **774 unique 0.50° ocean cells**.
  - **Direct Ocean (509 ports):** `wave_modeling_eligibility_status = 'ELIGIBLE_PRIMARY'`, `offshore_proxy_flag = False`.
  - **Nearby Coastal Proxies (205 ports):** `wave_modeling_eligibility_status = 'ELIGIBLE_PRIMARY'`, `offshore_proxy_flag = False` (Approach channel forcing; does not claim harbor/berth observation).
  - **Regional Marine Proxies (226 ports):** `wave_modeling_eligibility_status = 'ELIGIBLE_CONDITIONAL'`, `offshore_proxy_flag = True` (Displaced 40–100 km).
  - **Geographically Unrepresentative / Transit Only (21 ports):** `wave_modeling_eligibility_status = 'INELIGIBLE'`, `offshore_proxy_flag = False` (Extracted for regional shipping lane transit; masked for berth dwell modeling).
- **Wave Bypassed / Ineligible Ports (1,104 ports):**
  - **Blocked Inland (15 ports):** No ocean cells within radius; wave coordinates strictly null.
  - **Identity / Coordinate Uncertain (4 ports):** Gary, Bay City, Monroe, Delfzijl; wave queries bypassed.
  - **Unmapped (1,085 ports):** Primary coordinates missing; wave queries bypassed.

---

## 4. Direct Grid-Cell Count Reconciliation

- **Surface Grid Vertices:**
  - Total surface-eligible ports: 976
  - Actual unique 0.25° grid vertices from output rows: **936**
  - Mean geodesic distance to vertex: 9.62 km (max: 18.31 km)
- **Wave Ocean Grid Cells:**
  - Total wave-extractable ports: 961
  - Actual unique 0.50° valid ocean cells from output rows: **774**
  - Unique ocean cells for dwell-modeling ports (940 ports): **763**
  - Unique ocean cells for transit-only ports (21 ports): **16**
  - Overlapping cells between dwell-modeling and transit-only: **5** ($763 + 16 - 5 = 774$).

---

## 5. 15-Rule Independent Verification Audit

All 15 independent verification checks passed with exit code 0:
1. Rule 1 PASS: Every accepted registry identity appears as an accepted identity candidate.
2. Rule 2 PASS: No rejected candidate is simultaneously represented as the accepted identity.
3. Rule 3 PASS: Candidate identity acceptance is independent of coordinate acceptance (confirmed on Gary, Monroe, Delfzijl).
4. Rule 4 PASS: Accepted coordinates match governance coordinates for all 976 eligible ports.
5. Rule 5 PASS: All four exceptions have strictly null accepted coordinates.
6. Rule 6 PASS: Exactly 2,065 unique PortWatch records are represented.
7. Rule 7 PASS: Exactly 976 ports are surface eligible (and 1,089 ineligible).
8. Rule 8 PASS: Exactly 961 ports are wave extractable (940 model-eligible + 21 transit-only).
9. Rule 9 PASS: Exactly 940 ports are wave-model eligible (714 primary + 226 conditional).
10. Rule 10 PASS: Exactly 226 regional marine proxies have `offshore_proxy_flag = True`.
11. Rule 11 PASS: Zero other categories receive `offshore_proxy_flag = True`.
12. Rule 12 PASS: All 21 geographically unrepresentative ports are strictly excluded from port-level wave dwell modeling.
13. Rule 13 PASS: 15 blocked inland ports remain completely distinct from the 1,085 unmapped ports.
14. Rule 14 PASS: All four exceptions are excluded from both surface and wave modeling.
15. Rule 15 PASS: Kralendijk (`port598`) retains explicit BES/BQ country-discrepancy provenance.