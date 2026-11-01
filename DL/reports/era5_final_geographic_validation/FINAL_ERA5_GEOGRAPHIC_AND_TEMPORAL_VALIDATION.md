# Final Geographic and Temporal Validation: Global ERA5 Wave Mapping

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Milestone:** Phase 2 Data Engineering — Final Wave Extraction Validation  
**Date of Audit:** 2026-11-01 01:25:23 UTC  
**Target Extraction Period:** January 1, 2019 – December 31, 2025  

---

## 1. Executive Decision Summary

| Extraction Category | Port Count | Unique Cells | Extraction Decision | Dwell-Time Modeling Eligibility |
|:---|:---:|:---:|:---:|:---|
| **Global ERA5 Surface Weather** | 980 | 940 | **APPROVED (GO)** | Fully Eligible (flag 4 identity collisions) |
| **Direct Ocean Wave Cells** | 509 | 455 | **APPROVED (GO)** | Fully Eligible (primary marine forcing) |
| **Nearby Coastal Wave Proxies (<40 km)** | 205 | 179 | **APPROVED (GO)** | Eligible (outer approach marine forcing) |
| **Regional Marine Proxies (40–100 km)** | 226 | 184 | **CONDITIONAL GO** | Eligible with mandatory proxy flag |
| **Distant / Disconnected Proxies (>100 km / Rivers)** | 21 | 17 | **EXTRACT AS TRANSIT ONLY** | **INELIGIBLE FOR PORT-LEVEL DWELL MODELING** |
| **Identity / Coordinate Uncertain** | 4 | 4 | **DO NOT EXTRACT WAVE** | **INELIGIBLE PENDING REGISTRY REVIEW** |
| **Blocked Inland Ports** | 15 | 0 | **BYPASS WAVE QUERIES** | Wave is NaN; Surface is Fully Eligible |
| **Unmapped Ports (Missing Coords)** | 1,085 | 0 | **RETAIN NULLS** | Ineligible pending primary coordinate recovery |

**Executive Conclusion:** Global ERA5 extraction can proceed immediately under a **differentiated feature-governance policy**. Bulk extraction should ingest all validated surface grid cells (940 cells) and confirmed marine wave cells. However, downstream deep learning and dwell-time prediction models must strictly mask or nullify wave features for the 21 distant/disconnected ports and the 4 identity-uncertain ports.

---

## 2. Part 1 — Investigation of Questionable Port Geography

Our investigation resolved the underlying causes of geographical anomalies discovered in earlier audits:

### Summary of Reviewed Exceptional Records:
1. **Gary (`port377`, United States):**
   - *Actual Port:* Buffington Harbor / Gary Works in Gary, Indiana on southern Lake Michigan.
   - *Manifest Coordinates:* `32.0167°N, -94.3667°W` (Gary, Texas, Panola County, population ~300).
   - *Cause:* Upply indexed `USGT4` (Gary, TX) and omitted `USGYY` (Gary, IN). The automated registry builder accepted `USGT4`, importing Texas coordinates.
   - *Status:* `IDENTITY_OR_COORDINATE_UNCERTAIN`. Wave cell displaced 279 km to Gulf of Mexico is physically invalid for Lake Michigan.
2. **Bay City (`port130`, United States):**
   - *Actual Port:* Commercial limestone and coal port of Bay City, Michigan on Saginaw River / Saginaw Bay (Lake Huron).
   - *Manifest Coordinates:* `44.6000°N, -92.4500°W` (Bay City, Wisconsin on Lake Pepin / Mississippi River).
   - *Cause:* Upply cataloged `USYBZ` (Bay City, WI) and omitted `USBCY` (Bay City, MI). Automated matching imported Wisconsin coordinates.
   - *Status:* `IDENTITY_OR_COORDINATE_UNCERTAIN`. Wave cell displaced 290 km north to Lake Superior is invalid for Saginaw Bay.
3. **Monroe (`port2270`, United States):**
   - *Actual Port:* Port of Monroe, Michigan on western Lake Erie (`USMON`).
   - *Manifest Coordinates:* `44.2500°N, -72.0500°W` (Monroe, New Hampshire on Connecticut River, `USM4X`).
   - *Cause:* Name collision in UN/LOCODE / Upply.
   - *Status:* `IDENTITY_OR_COORDINATE_UNCERTAIN`. Displaced 187 km to Gulf of Maine.
4. **Delfzijl (`port285`, The Netherlands):**
   - *Actual Port:* Major seaport on the Ems estuary / Wadden Sea (~53.33°N, 6.92°E).
   - *Manifest Coordinates:* `52.5167°N, 6.9333°E` (inland Drenthe farmland near Emmen, 90 km south of port).
   - *Cause:* Official UNECE UN/LOCODE release and Upply contain a primary coordinate typographical error (`5231N` instead of `5319N`).
   - *Status:* `IDENTITY_OR_COORDINATE_UNCERTAIN`.
5. **Quebec (`port1075`), Rosario (`port1108`), Nanjing (`port2020`), Munguba (`port2362`), Varkaus (`port1354`):
   - *Physical Coordinates:* 100% accurate for their respective cities/terminals.
   - *Physical Reality:* All five are inland river, estuarine, or lake ports. In ERA5 WAM, narrow inland waterways are unmodeled.
   - *Algorithmic Consequence:* The 2.5° nearest-vertex search crossed major topographic barriers:
     - Quebec was mapped 310.4 km across Maine and the Appalachian mountains to the Gulf of Maine.
     - Rosario was mapped 299.8 km downriver to the open South Atlantic shelf.
     - Nanjing was mapped 234.8 km downriver to the Yellow Sea.
     - Munguba was mapped 235.4 km through the Amazon delta to the equatorial Atlantic.
     - Varkaus was mapped 201.2 km across land to Lake Ladoga / Gulf of Finland.
   - *Status:* `GEOGRAPHICALLY_UNREPRESENTATIVE` for local berth dwell-time modeling.

All records are cataloged in [`port_geography_exception_review.csv`](file:///Users/abdul/Desktop/DS/DL/reports/era5_final_geographic_validation/port_geography_exception_review.csv).

---

## 3. Part 2 — Evaluation of Wave-Proxy Representativeness (456 Displaced Proxies)

We evaluated every displaced proxy across distance, marine basin consistency, hydrological water-body type, and geographical barriers:

| Evidence-Based Classification | Port Count | % Displaced | Description & Governance Rule |
|:---|:---:|:---:|:---|
| **Nearby Coastal Proxy** | 205 | 45.0% | Distance < 40 km. Unobstructed coastal shelf cell in the immediate harbor approach. **Approved for dwell modeling.** |
| **Regional Marine Forcing Only** | 226 | 49.6% | Distance 40-100 km. Outer shelf cell in the same regional marine basin. **Eligible with mandatory proxy flag.** |
| **Geographically Unrepresentative** | 21 | 4.6% | Distance > 100 km or inland river/lake port (e.g. Baton Rouge, Stockton, Galati, Rosario, Quebec). **Excluded from port-level dwell modeling.** |
| **Identity / Coordinate Uncertain** | 4 | 0.9% | Source data name collision or coordinate typo (Gary, Bay City, Monroe, Delfzijl). **Excluded pending registry update.** |
| **Total Displaced Wave Proxies** | **456** | **100.0%** | Full displaced universe accounted for |

All 456 classifications are documented in [`wave_proxy_geographic_review.csv`](file:///Users/abdul/Desktop/DS/DL/reports/era5_final_geographic_validation/wave_proxy_geographic_review.csv).

---

## 4. Part 3 — 2019–2025 Multi-Temporal Availability Audit (84 Monthly Audits)

We audited all **776 unique selected wave cells** across **84 monthly timestamps** from January 15, 2019 to December 15, 2025:

- **100% Persistently Available:** **740 unique cells (95.36%)** returned complete, valid SWH across all 84 audited months in 2019–2025.
- **Seasonally Missing (Plausible Sea Ice):** **36 unique cells (4.64%)** exhibited missingness strictly confined to winter/spring months (December–May).
- **Intermittent Missingness:** **0 cells (0.00%)**.
- **Permanently Missing (Land/Void):** **0 cells (0.00%)**.

### Sea-Ice Verification Caveat
Neither `cadl-arco-geo-003` (wave) nor `cadl-arco-geo-002` (surface) stores a sea-ice concentration variable (`siconc`). While the 36 seasonal cells are all located in sub-polar regions with known winter freeze (Bothnian Bay, Gulf of St. Lawrence, Sea of Okhotsk, Sea of Azov), the sea-ice mechanism cannot be confirmed directly within the primary ARCO store metadata and is formally labeled as **UNCONFIRMED_IN_STORE**.

Detailed results are in [`wave_temporal_availability_2019_2025.csv`](file:///Users/abdul/Desktop/DS/DL/reports/era5_final_geographic_validation/wave_temporal_availability_2019_2025.csv).

---

## 5. Part 4 — Reassessment of the Wave-Selection Methodology

The implementation in `scripts/build_global_era5_spatial_manifest_v2.py` was evaluated against physical criteria:
1. **Reference Coordinates & Geodesic Math:** Verified correct. Uses `geod.inv` from the port coordinates, correctly sorting ellipsoidal geodesic distance.
2. **Topological Land Barriers:** **Deficient.** The algorithm operates in a blind geometric bounding box without ray-tracing or coastline intersection tests. It cannot detect when a search traverses a mountain range or peninsula into an unrelated marine basin (e.g. Quebec into Maine).
3. **Inland Port Discrimination:** **Deficient.** The algorithm does not distinguish between a coastal container terminal and an inland river grain dock.

### Recommended Future Selection Policy:
- Impose a hard displacement ceiling of **1.0° (~100 km)**.
- Automatically assign `NO_VALID_WAVE_OBSERVATION` to ports located on inland river systems.
- Require hydrological basin matching before accepting displaced proxies.

---

## 6. Part 5 — Final Extraction-Readiness Decisions

1. **Global Surface Extraction (980 ports, 940 cells):** **GO (APPROVED).** Continuous global coverage; maximum distance 18.31 km.
2. **Direct Ocean Wave Cells (509 ports, 455 cells):** **GO (APPROVED).** Persistent marine coverage.
3. **Nearby Coastal Proxies (205 ports, 179 cells):** **GO (APPROVED).** Outer approach marine forcing.
4. **Regional Marine Proxies (226 ports, 184 cells):** **CONDITIONAL GO.** Retain proxy flag; evaluate ablation.
5. **Distant / Disconnected Proxies (21 ports, 17 cells):** **NO-GO FOR PORT MODELING.** Extract only as regional transit features; mask for dwell-time prediction.
6. **Identity / Coordinate Uncertain (4 ports, 4 cells):** **DO NOT EXTRACT WAVE.** Ineligible pending primary registry review.
7. **Blocked Inland Ports (15 ports, 0 cells):** **BYPASS.** Correctly retained with null wave coordinates.

---

## 7. Protected Artifact Immutability Verification

All protected artifacts were hashed before and after execution:
- `canonical_port_registry.csv`: `03ff0ef1b93828e8...` (UNCHANGED)
- `global_era5_spatial_manifest_v1.parquet`: `d9b6dbfee21b36d8...` (UNCHANGED)
- `global_era5_spatial_manifest_v2.parquet`: `a88b5724b3992284...` (UNCHANGED)
- All frozen daily/hourly ERA5 datasets verified 100% unchanged.
