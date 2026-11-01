# Port Identity & Coordinate Exception Resolution Report

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Milestone:** Phase 2 Data Engineering — Final Spatial Governance  
**Date:** 2026-11-01 02:37:25 UTC  

---

## Executive Summary

Four critical identity and coordinate anomalies were investigated across primary UNECE UN/LOCODE releases, Upply Seaports, PortWatch source records, and Phase 1 assets. All four have been resolved strictly without coordinate fabrication:
- **2 Records Assigned `IDENTITY_CONFIRMED_COORDINATE_UNAVAILABLE`:** Gary (`port377`) and Monroe (`port2270`) represent confirmed Great Lakes ports in Indiana and Michigan, but their correct UN/LOCODE entries (`USGYY`, `USMOI`) lack coordinates in primary releases. Erroneous Texas and New Hampshire coordinates are strictly rejected, and accepted coordinates are set to null.
- **2 Records Assigned `SOURCE_CONFLICT_REQUIRES_MANUAL_REVIEW`:** Bay City (`port130`) and Delfzijl (`port285`) exhibit multi-source reference conflicts (Bay City text LOCODE vs function code collision; Delfzijl 1-degree latitude typographical error in UNECE CodeListPart2). Accepted coordinates are set to null pending manual review.

---

## Detailed Port-by-Port Evaluations

### Gary (`port377` — United States)
- **Intended Identity:** Gary, Indiana (USGYY) on southern Lake Michigan (Buffington Harbor / Gary Works).
- **Reported Problem:** Manifest resolves to Gary, Texas (USGT4, Panola County, TX, lat 32.0167, lon -94.3667) rather than Gary, Indiana.
- **Root Cause:** BOTH: Upply indexed USGT4 (Gary, TX) and omitted USGYY (Gary, IN); the automated registry builder matched USGT4 via exact name match without state verification.
- **UN/LOCODE Evidence:** UNECE UN/LOCODE CodeListPart3.csv contains USGYY (Gary, IN, func 1--4----, coords nan) and USGT4 (Gary, TX, func 1-------, coords 3201N 09422W).
- **Upply Evidence:** UPPLY-SEAPORTS.csv contains code USGT4 (Gary, TX, lat 32.0167, lon -94.3667); USGYY is not in Upply.
- **Authoritative Findings:** The commercial seaport is definitively Gary, Indiana. However, authoritative coordinates for USGYY are missing/empty in UNECE UN/LOCODE and Upply. Texas coordinates are rejected.
- **Assigned Resolution Outcome:** `IDENTITY_CONFIRMED_COORDINATE_UNAVAILABLE`
- **Action Taken:** Set accepted UN/LOCODE to USGYY (Gary, IN); set accepted coordinates to NaN; exclude from automated wave/surface extraction.

### Bay City (`port130` — United States)
- **Intended Identity:** Bay City, Michigan (USBCY) on Saginaw Bay / Lake Huron.
- **Reported Problem:** Manifest resolves to Bay City, Wisconsin (USYBZ, Pierce County, WI, lat 44.6000, lon -92.4500) rather than Bay City, Michigan.
- **Root Cause:** BOTH: Upply only cataloged USYBZ (WI); UNECE UN/LOCODE cataloged USBCY (MI) with function '--34----' (no port function), causing the registry builder to filter USBCY out and match USYBZ.
- **UN/LOCODE Evidence:** UNECE UN/LOCODE CodeListPart3.csv contains USBCY (Bay City, MI, func --34----, coords 4335N 08352W) and USYBZ (Bay City, WI, func 123-----, coords 4436N 09227W).
- **Upply Evidence:** UPPLY-SEAPORTS.csv contains code USYBZ (Bay City, WI, lat 44.6, lon -92.45); USBCY is not in Upply.
- **Authoritative Findings:** Phase 1 PortWatch ports.csv recorded Bay City at lat 43.6258, lon -83.8503 (Michigan), but text LOCODE 'US YBZ'. UNECE associates USYBZ with Wisconsin and USBCY with Michigan (without port function). A direct multi-source conflict exists between text LOCODE, function code, and coordinates.
- **Assigned Resolution Outcome:** `SOURCE_CONFLICT_REQUIRES_MANUAL_REVIEW`
- **Action Taken:** Reject Wisconsin coordinates; retain record with null accepted coordinates; flag for expert manual resolution.

### Monroe (`port2270` — United States)
- **Intended Identity:** Port of Monroe, Michigan (USMOI) on Lake Erie.
- **Reported Problem:** Manifest resolves to Monroe, New Hampshire (USM4X, Grafton County, NH, lat 44.2500, lon -72.0500) rather than Monroe, Michigan.
- **Root Cause:** BOTH: Upply only indexed USM4X (NH); in UNECE UN/LOCODE, USMOI (MI) has function '--3-----' (no port function) and null coordinates, whereas USM4X had function '1-3-----' (port function).
- **UN/LOCODE Evidence:** UNECE UN/LOCODE CodeListPart3.csv contains USMOI (Monroe, MI, func --3-----, coords nan) and USM4X (Monroe, NH, func 1-3-----, coords 4415N 07203W).
- **Upply Evidence:** UPPLY-SEAPORTS.csv contains code USM4X (Monroe, NH, lat 44.25, lon -72.05); USMOI is not in Upply.
- **Authoritative Findings:** The commercial port is Monroe, Michigan on Lake Erie. However, USMOI has null coordinates in UNECE UN/LOCODE and is absent from Upply. New Hampshire coordinates are rejected.
- **Assigned Resolution Outcome:** `IDENTITY_CONFIRMED_COORDINATE_UNAVAILABLE`
- **Action Taken:** Set accepted UN/LOCODE to USMOI (Monroe, MI); set accepted coordinates to NaN; exclude from automated wave/surface extraction.

### Delfzijl (`port285` — The Netherlands)
- **Intended Identity:** Port of Delfzijl (NLDZL) on the Ems estuary / Wadden Sea in Groningen province, Netherlands.
- **Reported Problem:** Manifest coordinates (52.5167°N, 6.9333°E) are approximately 89 km south on farmland near Emmen in Drenthe province.
- **Root Cause:** PRIMARY SOURCE TYPOGRAPHICAL ERROR: Official UNECE UN/LOCODE CodeListPart2.csv literally records coordinate string '5231N 00656E' instead of '5319N 00656E'. Upply copied the exact UN/LOCODE coordinate.
- **UN/LOCODE Evidence:** UNECE UN/LOCODE CodeListPart2.csv row 24631 records NLDZL (Delfzijl, func 123-----, coords 5231N 00656E).
- **Upply Evidence:** UPPLY-SEAPORTS.csv records NLDZL (Delfzijl, lat 52.5167, lon 6.9333).
- **Authoritative Findings:** Identity is 100% confirmed as NLDZL. However, primary source coordinates in UN/LOCODE and Upply contain an uncorrected 1-degree latitude error. Phase 1 reference has 53.3150°N, 6.9642°E. Because raw source datasets cannot be modified, a formal source conflict exists between primary UNECE records and physical truth.
- **Assigned Resolution Outcome:** `SOURCE_CONFLICT_REQUIRES_MANUAL_REVIEW`
- **Action Taken:** Retain source coordinates in raw fields; set accepted coordinates to NaN pending formal registry errata; exclude from automated wave/surface extraction.
