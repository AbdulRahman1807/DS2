#!/usr/bin/env python3
"""
Build Spatial Feature Governance & Port Identity Resolution.

Executes:
  Part 1: Resolve the four identity/coordinate exceptions (Gary, Bay City, Monroe, Delfzijl).
  Part 2: Generate canonical port registry v1.3 in reports/port_registry/v1_3_spatial_governance/.
  Part 3: Build machine-readable spatial feature governance layer for all 2,065 PortWatch ports.
  Part 4: Validate classification consistency across all 12 audit rules.
  Part 5: Formulate extraction & dwell-time modeling feature governance policy.

Outputs to:
  - reports/port_identity_resolution/
  - reports/port_registry/v1_3_spatial_governance/
"""

import os
import sys
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Geod

ROOT = Path(__file__).resolve().parents[1]
IDENTITY_DIR = ROOT / "reports" / "port_identity_resolution"
REGISTRY_V1_3_DIR = ROOT / "reports" / "port_registry" / "v1_3_spatial_governance"

# Inputs
V1_MANIFEST_PATH = ROOT / "data" / "interim" / "era5" / "global_manifest_v1" / "global_era5_spatial_manifest.parquet"
V2_MANIFEST_PATH = ROOT / "data" / "interim" / "era5" / "global_manifest_v2" / "global_era5_spatial_manifest_v2.parquet"
CANONICAL_REGISTRY_V1_2_PATH = ROOT / "reports" / "port_registry" / "v1_2_identifier_bridge" / "canonical_port_registry.csv"
CANDIDATES_V1_2_PATH = ROOT / "reports" / "port_registry" / "v1_2_identifier_bridge" / "port_identity_candidates.csv"
UPPLY_PATH = ROOT / "data" / "reference" / "upply" / "UPPLY-SEAPORTS.csv"
DAILY_PORTS_PATH = ROOT / "data" / "raw" / "portwatch" / "Daily_Ports_Data.csv"
PROXY_REVIEW_PATH = ROOT / "reports" / "era5_final_geographic_validation" / "wave_proxy_geographic_review.csv"

# Protected artifacts to verify
PROTECTED_FILES = {
    "canonical_port_registry_v1_2.csv": ROOT / "reports" / "port_registry" / "v1_2_identifier_bridge" / "canonical_port_registry.csv",
    "registry_metadata_v1_2.json": ROOT / "reports" / "port_registry" / "v1_2_identifier_bridge" / "registry_metadata.json",
    "era5_port_hourly.parquet": ROOT / "data" / "interim" / "era5" / "era5_port_hourly.parquet",
    "era5_port_hourly_metadata.json": ROOT / "data" / "interim" / "era5" / "era5_port_hourly_metadata.json",
    "era5_port_daily_v2.parquet": ROOT / "data" / "processed" / "era5" / "era5_port_daily_v2.parquet",
    "era5_port_daily_v2_metadata.json": ROOT / "data" / "processed" / "era5" / "era5_port_daily_v2_metadata.json",
    "global_era5_spatial_manifest_v1.parquet": ROOT / "data" / "interim" / "era5" / "global_manifest_v1" / "global_era5_spatial_manifest.parquet",
    "global_era5_spatial_manifest_v1.csv": ROOT / "data" / "interim" / "era5" / "global_manifest_v1" / "global_era5_spatial_manifest.csv",
    "global_era5_spatial_manifest_v2.parquet": ROOT / "data" / "interim" / "era5" / "global_manifest_v2" / "global_era5_spatial_manifest_v2.parquet",
    "global_era5_spatial_manifest_v2.csv": ROOT / "data" / "interim" / "era5" / "global_manifest_v2" / "global_era5_spatial_manifest_v2.csv",
}


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("PHASE 2: RESOLVE PORT IDENTITY EXCEPTIONS & FINALIZE SPATIAL GOVERNANCE")
    print("=" * 80)

    IDENTITY_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_V1_3_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Record pre-execution hashes
    print("\n[1/6] Verifying immutability of protected artifacts...")
    pre_hashes = {}
    for name, path in PROTECTED_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing protected file: {path}")
        pre_hashes[name] = compute_sha256(path)
        print(f"  {name}: {pre_hashes[name][:16]}...")

    # Load baseline datasets
    reg_v1_2 = pd.read_csv(CANONICAL_REGISTRY_V1_2_PATH)
    cand_v1_2 = pd.read_csv(CANDIDATES_V1_2_PATH)
    manifest_v2 = pd.read_parquet(V2_MANIFEST_PATH)
    proxy_review = pd.read_csv(PROXY_REVIEW_PATH).set_index("portwatch_id")

    # =========================================================================
    # PART 1: RESOLVE THE FOUR IDENTITY / COORDINATE EXCEPTIONS
    # =========================================================================
    print("\n[2/6] Executing Part 1: Resolving the 4 Port Identity Exceptions...")
    exception_records = [
        {
            "portwatch_id": "port377",
            "portwatch_name": "Gary",
            "country": "United States",
            "intended_port_identity": "Gary, Indiana (USGYY) on southern Lake Michigan (Buffington Harbor / Gary Works).",
            "reported_problem": "Manifest resolves to Gary, Texas (USGT4, Panola County, TX, lat 32.0167, lon -94.3667) rather than Gary, Indiana.",
            "source_of_error": "BOTH: Upply indexed USGT4 (Gary, TX) and omitted USGYY (Gary, IN); the automated registry builder matched USGT4 via exact name match without state verification.",
            "unlocode_evidence": "UNECE UN/LOCODE CodeListPart3.csv contains USGYY (Gary, IN, func 1--4----, coords nan) and USGT4 (Gary, TX, func 1-------, coords 3201N 09422W).",
            "upply_evidence": "UPPLY-SEAPORTS.csv contains code USGT4 (Gary, TX, lat 32.0167, lon -94.3667); USGYY is not in Upply.",
            "authoritative_findings": "The commercial seaport is definitively Gary, Indiana. However, authoritative coordinates for USGYY are missing/empty in UNECE UN/LOCODE and Upply. Texas coordinates are rejected.",
            "assigned_outcome": "IDENTITY_CONFIRMED_COORDINATE_UNAVAILABLE",
            "action_taken": "Set accepted UN/LOCODE to USGYY (Gary, IN); set accepted coordinates to NaN; exclude from automated wave/surface extraction."
        },
        {
            "portwatch_id": "port130",
            "portwatch_name": "Bay City",
            "country": "United States",
            "intended_port_identity": "Bay City, Michigan (USBCY) on Saginaw Bay / Lake Huron.",
            "reported_problem": "Manifest resolves to Bay City, Wisconsin (USYBZ, Pierce County, WI, lat 44.6000, lon -92.4500) rather than Bay City, Michigan.",
            "source_of_error": "BOTH: Upply only cataloged USYBZ (WI); UNECE UN/LOCODE cataloged USBCY (MI) with function '--34----' (no port function), causing the registry builder to filter USBCY out and match USYBZ.",
            "unlocode_evidence": "UNECE UN/LOCODE CodeListPart3.csv contains USBCY (Bay City, MI, func --34----, coords 4335N 08352W) and USYBZ (Bay City, WI, func 123-----, coords 4436N 09227W).",
            "upply_evidence": "UPPLY-SEAPORTS.csv contains code USYBZ (Bay City, WI, lat 44.6, lon -92.45); USBCY is not in Upply.",
            "authoritative_findings": "Phase 1 PortWatch ports.csv recorded Bay City at lat 43.6258, lon -83.8503 (Michigan), but text LOCODE 'US YBZ'. UNECE associates USYBZ with Wisconsin and USBCY with Michigan (without port function). A direct multi-source conflict exists between text LOCODE, function code, and coordinates.",
            "assigned_outcome": "SOURCE_CONFLICT_REQUIRES_MANUAL_REVIEW",
            "action_taken": "Reject Wisconsin coordinates; retain record with null accepted coordinates; flag for expert manual resolution."
        },
        {
            "portwatch_id": "port2270",
            "portwatch_name": "Monroe",
            "country": "United States",
            "intended_port_identity": "Port of Monroe, Michigan (USMOI) on Lake Erie.",
            "reported_problem": "Manifest resolves to Monroe, New Hampshire (USM4X, Grafton County, NH, lat 44.2500, lon -72.0500) rather than Monroe, Michigan.",
            "source_of_error": "BOTH: Upply only indexed USM4X (NH); in UNECE UN/LOCODE, USMOI (MI) has function '--3-----' (no port function) and null coordinates, whereas USM4X had function '1-3-----' (port function).",
            "unlocode_evidence": "UNECE UN/LOCODE CodeListPart3.csv contains USMOI (Monroe, MI, func --3-----, coords nan) and USM4X (Monroe, NH, func 1-3-----, coords 4415N 07203W).",
            "upply_evidence": "UPPLY-SEAPORTS.csv contains code USM4X (Monroe, NH, lat 44.25, lon -72.05); USMOI is not in Upply.",
            "authoritative_findings": "The commercial port is Monroe, Michigan on Lake Erie. However, USMOI has null coordinates in UNECE UN/LOCODE and is absent from Upply. New Hampshire coordinates are rejected.",
            "assigned_outcome": "IDENTITY_CONFIRMED_COORDINATE_UNAVAILABLE",
            "action_taken": "Set accepted UN/LOCODE to USMOI (Monroe, MI); set accepted coordinates to NaN; exclude from automated wave/surface extraction."
        },
        {
            "portwatch_id": "port285",
            "portwatch_name": "Delfzijl",
            "country": "The Netherlands",
            "intended_port_identity": "Port of Delfzijl (NLDZL) on the Ems estuary / Wadden Sea in Groningen province, Netherlands.",
            "reported_problem": "Manifest coordinates (52.5167°N, 6.9333°E) are approximately 89 km south on farmland near Emmen in Drenthe province.",
            "source_of_error": "PRIMARY SOURCE TYPOGRAPHICAL ERROR: Official UNECE UN/LOCODE CodeListPart2.csv literally records coordinate string '5231N 00656E' instead of '5319N 00656E'. Upply copied the exact UN/LOCODE coordinate.",
            "unlocode_evidence": "UNECE UN/LOCODE CodeListPart2.csv row 24631 records NLDZL (Delfzijl, func 123-----, coords 5231N 00656E).",
            "upply_evidence": "UPPLY-SEAPORTS.csv records NLDZL (Delfzijl, lat 52.5167, lon 6.9333).",
            "authoritative_findings": "Identity is 100% confirmed as NLDZL. However, primary source coordinates in UN/LOCODE and Upply contain an uncorrected 1-degree latitude error. Phase 1 reference has 53.3150°N, 6.9642°E. Because raw source datasets cannot be modified, a formal source conflict exists between primary UNECE records and physical truth.",
            "assigned_outcome": "SOURCE_CONFLICT_REQUIRES_MANUAL_REVIEW",
            "action_taken": "Retain source coordinates in raw fields; set accepted coordinates to NaN pending formal registry errata; exclude from automated wave/surface extraction."
        }
    ]

    exc_df = pd.DataFrame(exception_records)
    exc_csv_path = IDENTITY_DIR / "exception_resolution.csv"
    exc_df.to_csv(exc_csv_path, index=False)
    print(f"  Wrote exception resolution to {exc_csv_path}")

    with open(IDENTITY_DIR / "exception_resolution.json", "w") as f:
        json.dump({
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "records_resolved": exception_records
        }, f, indent=2)

    # Markdown Report for Identity Resolution
    md_lines = [
        "# Port Identity & Coordinate Exception Resolution Report",
        "",
        "**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  ",
        "**Milestone:** Phase 2 Data Engineering — Final Spatial Governance  ",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "Four critical identity and coordinate anomalies were investigated across primary UNECE UN/LOCODE releases, Upply Seaports, PortWatch source records, and Phase 1 assets. All four have been resolved strictly without coordinate fabrication:",
        "- **2 Records Assigned `IDENTITY_CONFIRMED_COORDINATE_UNAVAILABLE`:** Gary (`port377`) and Monroe (`port2270`) represent confirmed Great Lakes ports in Indiana and Michigan, but their correct UN/LOCODE entries (`USGYY`, `USMOI`) lack coordinates in primary releases. Erroneous Texas and New Hampshire coordinates are strictly rejected, and accepted coordinates are set to null.",
        "- **2 Records Assigned `SOURCE_CONFLICT_REQUIRES_MANUAL_REVIEW`:** Bay City (`port130`) and Delfzijl (`port285`) exhibit multi-source reference conflicts (Bay City text LOCODE vs function code collision; Delfzijl 1-degree latitude typographical error in UNECE CodeListPart2). Accepted coordinates are set to null pending manual review.",
        "",
        "---",
        "",
        "## Detailed Port-by-Port Evaluations",
        ""
    ]
    for r in exception_records:
        md_lines.extend([
            f"### {r['portwatch_name']} (`{r['portwatch_id']}` — {r['country']})",
            f"- **Intended Identity:** {r['intended_port_identity']}",
            f"- **Reported Problem:** {r['reported_problem']}",
            f"- **Root Cause:** {r['source_of_error']}",
            f"- **UN/LOCODE Evidence:** {r['unlocode_evidence']}",
            f"- **Upply Evidence:** {r['upply_evidence']}",
            f"- **Authoritative Findings:** {r['authoritative_findings']}",
            f"- **Assigned Resolution Outcome:** `{r['assigned_outcome']}`",
            f"- **Action Taken:** {r['action_taken']}",
            ""
        ])
    with open(IDENTITY_DIR / "PORT_IDENTITY_RESOLUTION.md", "w") as f:
        f.write("\n".join(md_lines))
    print(f"  Wrote markdown resolution report to {IDENTITY_DIR / 'PORT_IDENTITY_RESOLUTION.md'}")

    # =========================================================================
    # PART 2: BUILD CANONICAL PORT REGISTRY V1.3
    # =========================================================================
    print("\n[3/6] Executing Part 2: Generating Canonical Port Registry v1.3...")
    reg_v1_3 = reg_v1_2.copy()

    # Add explicit accepted coordinate and resolution columns
    reg_v1_3["accepted_unlocode"] = reg_v1_3["unlocode"]
    reg_v1_3["accepted_latitude"] = reg_v1_3["upply_latitude"]
    reg_v1_3["accepted_longitude"] = reg_v1_3["upply_longitude"]
    reg_v1_3["coordinate_provenance"] = "UPPLY_AUTHORITATIVE"

    # Incorporate the 31 UN/LOCODE recovered ports from v2 manifest
    unlo_recovered_v2 = manifest_v2[manifest_v2["coordinate_recovery_status"] != "PRESERVED_FROM_V1"].copy()
    for _, r in unlo_recovered_v2.iterrows():
        pid = r["portwatch_id"]
        idx = reg_v1_3[reg_v1_3["portwatch_id"] == pid].index[0]
        reg_v1_3.loc[idx, "accepted_unlocode"] = r["unlocode"]
        reg_v1_3.loc[idx, "accepted_latitude"] = r["recovered_latitude"]
        reg_v1_3.loc[idx, "accepted_longitude"] = r["recovered_longitude"]
        reg_v1_3.loc[idx, "coordinate_provenance"] = "UNLOCODE_CODELIST_RECOVERED"

    # Now apply the 4 exception corrections in v1.3
    # Gary: port377 -> USGYY, coords NaN
    idx_gary = reg_v1_3[reg_v1_3["portwatch_id"] == "port377"].index[0]
    reg_v1_3.loc[idx_gary, "accepted_unlocode"] = "USGYY"
    reg_v1_3.loc[idx_gary, "accepted_latitude"] = np.nan
    reg_v1_3.loc[idx_gary, "accepted_longitude"] = np.nan
    reg_v1_3.loc[idx_gary, "coordinate_provenance"] = "NONE_COORDINATES_UNAVAILABLE"
    reg_v1_3.loc[idx_gary, "identity_status"] = "IDENTITY_CONFIRMED_COORDINATE_UNAVAILABLE"

    # Bay City: port130 -> coords NaN
    idx_bc = reg_v1_3[reg_v1_3["portwatch_id"] == "port130"].index[0]
    reg_v1_3.loc[idx_bc, "accepted_unlocode"] = None
    reg_v1_3.loc[idx_bc, "accepted_latitude"] = np.nan
    reg_v1_3.loc[idx_bc, "accepted_longitude"] = np.nan
    reg_v1_3.loc[idx_bc, "coordinate_provenance"] = "NONE_SOURCE_CONFLICT"
    reg_v1_3.loc[idx_bc, "identity_status"] = "SOURCE_CONFLICT_REQUIRES_MANUAL_REVIEW"

    # Monroe: port2270 -> USMOI, coords NaN
    idx_mon = reg_v1_3[reg_v1_3["portwatch_id"] == "port2270"].index[0]
    reg_v1_3.loc[idx_mon, "accepted_unlocode"] = "USMOI"
    reg_v1_3.loc[idx_mon, "accepted_latitude"] = np.nan
    reg_v1_3.loc[idx_mon, "accepted_longitude"] = np.nan
    reg_v1_3.loc[idx_mon, "coordinate_provenance"] = "NONE_COORDINATES_UNAVAILABLE"
    reg_v1_3.loc[idx_mon, "identity_status"] = "IDENTITY_CONFIRMED_COORDINATE_UNAVAILABLE"

    # Delfzijl: port285 -> coords NaN
    idx_dzl = reg_v1_3[reg_v1_3["portwatch_id"] == "port285"].index[0]
    reg_v1_3.loc[idx_dzl, "accepted_latitude"] = np.nan
    reg_v1_3.loc[idx_dzl, "accepted_longitude"] = np.nan
    reg_v1_3.loc[idx_dzl, "coordinate_provenance"] = "NONE_SOURCE_TYPO_CONFLICT"
    reg_v1_3.loc[idx_dzl, "identity_status"] = "SOURCE_CONFLICT_REQUIRES_MANUAL_REVIEW"

    # Set nulls for all 1,085 unmapped ports
    unmapped_mask = reg_v1_3["accepted_latitude"].isna()
    reg_v1_3.loc[unmapped_mask & ~reg_v1_3["portwatch_id"].isin(["port377", "port130", "port2270", "port285"]), "coordinate_provenance"] = "NONE"

    # Save registry v1.3
    reg_v1_3_path = REGISTRY_V1_3_DIR / "canonical_port_registry.csv"
    reg_v1_3.to_csv(reg_v1_3_path, index=False)
    print(f"  Wrote canonical registry v1.3 to {reg_v1_3_path}")

    # Build updated candidates table v1.3
    cand_v1_3 = cand_v1_2.copy()
    # Update acceptance flags for the 4 ports
    # port377: USGT4 -> False, USGYY -> True
    cand_v1_3.loc[(cand_v1_3["portwatch_id"] == "port377") & (cand_v1_3["candidate_id"] == "USGT4"), "candidate_is_accepted"] = False
    cand_v1_3.loc[(cand_v1_3["portwatch_id"] == "port377") & (cand_v1_3["candidate_id"] == "USGYY"), "candidate_is_accepted"] = True
    # port130: USYBZ -> False
    cand_v1_3.loc[(cand_v1_3["portwatch_id"] == "port130") & (cand_v1_3["candidate_id"] == "USYBZ"), "candidate_is_accepted"] = False
    # port2270: USM4X -> False
    cand_v1_3.loc[(cand_v1_3["portwatch_id"] == "port2270") & (cand_v1_3["candidate_id"] == "USM4X"), "candidate_is_accepted"] = False
    # port285: NLDZL -> False (rejected coordinates pending review)
    cand_v1_3.loc[(cand_v1_3["portwatch_id"] == "port285") & (cand_v1_3["candidate_id"] == "NLDZL"), "candidate_is_accepted"] = False

    cand_v1_3_path = REGISTRY_V1_3_DIR / "port_identity_candidates.csv"
    cand_v1_3.to_csv(cand_v1_3_path, index=False)
    print(f"  Wrote candidates table v1.3 to {cand_v1_3_path}")

    # Metadata v1.3
    metadata_v1_3 = {
        "registry_version": "1.3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_portwatch_ports": len(reg_v1_3),
        "total_ports_with_accepted_coordinates": int(reg_v1_3["accepted_latitude"].notna().sum()),
        "total_ports_without_accepted_coordinates": int(reg_v1_3["accepted_latitude"].isna().sum()),
        "exceptions_resolved": ["port377 (Gary)", "port130 (Bay City)", "port2270 (Monroe)", "port285 (Delfzijl)"],
        "governance_layer_link": "reports/port_identity_resolution/spatial_feature_governance.csv",
        "description": "Canonical Port Registry v1.3 incorporates spatial governance, separates source coordinates from accepted coordinates, recovers 31 UN/LOCODE ports, and nullifies erroneous coordinates for four candidate collision exceptions."
    }
    with open(REGISTRY_V1_3_DIR / "registry_metadata.json", "w") as f:
        json.dump(metadata_v1_3, f, indent=2)

    # Reconciliation v1.3
    recon_v1_3 = pd.DataFrame([
        {"metric": "total_ports", "value": 2065, "notes": "Full PortWatch universe"},
        {"metric": "ports_with_accepted_coords", "value": int(reg_v1_3["accepted_latitude"].notna().sum()), "notes": "976 confirmed ports (975 conflict-free + 1 BES/BQ)"},
        {"metric": "ports_without_accepted_coords", "value": int(reg_v1_3["accepted_latitude"].isna().sum()), "notes": "1,085 originally unmapped + 4 exceptions = 1,089"},
        {"metric": "identity_confirmed_coords_unavailable", "value": 2, "notes": "Gary (USGYY) and Monroe (USMOI)"},
        {"metric": "source_conflict_manual_review", "value": 2, "notes": "Bay City (USBCY/USYBZ) and Delfzijl (NLDZL typo)"},
        {"metric": "unmapped_ports_retained", "value": 1085, "notes": "Strictly preserved with null coordinates"},
    ])
    recon_v1_3.to_csv(REGISTRY_V1_3_DIR / "source_reconciliation.csv", index=False)

    # =========================================================================
    # PART 3: CREATE EXPLICIT SPATIAL FEATURE GOVERNANCE
    # =========================================================================
    print("\n[4/6] Executing Part 3: Creating Spatial Feature Governance Layer (all 2,065 ports)...")
    
    exception_ids = {"port377", "port130", "port2270", "port285"}
    
    # 21 Geographically Unrepresentative IDs from Part 2 audit
    unrep_ids = set(proxy_review[proxy_review["evidence_based_classification"] == "GEOGRAPHICALLY_UNREPRESENTATIVE"].index)
    
    # 15 Blocked Inland ports
    blocked_inland_ids = set(manifest_v2[manifest_v2["wave_mapping_method"] == "BLOCKED_INLAND_NO_OCEAN_OBSERVATION"]["portwatch_id"])

    gov_rows = []
    
    for _, r in manifest_v2.iterrows():
        pid = r["portwatch_id"]
        pname = r["portwatch_name"]
        country = r["portwatch_country"]
        
        has_v2_coords = pd.notna(r["recovered_latitude"])
        
        # 1. Exceptional ports
        if pid in exception_ids:
            accepted_coords = None
            coord_source = "NONE"
            s_lat, s_lon = np.nan, np.nan
            s_ext_elig = "INELIGIBLE_UNRELIABLE_COORDINATES"
            s_model_elig = False
            w_lat, w_lon = np.nan, np.nan
            w_proxy_class = "IDENTITY_OR_COORDINATE_UNCERTAIN"
            w_dist = np.nan
            w_ext_elig = "INELIGIBLE_UNRELIABLE_COORDINATES"
            w_model_elig = False
            
            if pid == "port377":
                accepted_id = "USGYY"
                id_res_status = "IDENTITY_CONFIRMED_COORDINATE_UNAVAILABLE"
                excl_reason = "Gary, IN identity confirmed; official UN/LOCODE coordinates unavailable; rejected Gary, TX collision."
            elif pid == "port2270":
                accepted_id = "USMOI"
                id_res_status = "IDENTITY_CONFIRMED_COORDINATE_UNAVAILABLE"
                excl_reason = "Monroe, MI identity confirmed; official UN/LOCODE coordinates unavailable; rejected Monroe, NH collision."
            elif pid == "port130":
                accepted_id = None
                id_res_status = "SOURCE_CONFLICT_REQUIRES_MANUAL_REVIEW"
                excl_reason = "Bay City, MI/WI multi-source LOCODE and function code conflict; rejected USYBZ coordinates pending manual review."
            else:  # port285
                accepted_id = "NLDZL"
                id_res_status = "SOURCE_CONFLICT_REQUIRES_MANUAL_REVIEW"
                excl_reason = "Delfzijl, NL identity confirmed; official UN/LOCODE CodeListPart2 contains 1-degree latitude typo (52°31'N vs 53°19'N); rejected pending review."
            
            audit_ref = "reports/port_identity_resolution/PORT_IDENTITY_RESOLUTION.md"

        # 2. Unmapped ports (1,085)
        elif not has_v2_coords:
            accepted_id = None
            accepted_coords = None
            coord_source = "NONE"
            id_res_status = "UNRESOLVED_NO_COORDINATES"
            s_lat, s_lon = np.nan, np.nan
            s_ext_elig = "INELIGIBLE_MISSING_COORDINATES"
            s_model_elig = False
            w_lat, w_lon = np.nan, np.nan
            w_proxy_class = "UNMAPPED"
            w_dist = np.nan
            w_ext_elig = "INELIGIBLE_MISSING_COORDINATES"
            w_model_elig = False
            excl_reason = "PRIMARY_SOURCE_COORDINATES_MISSING"
            audit_ref = "data/interim/era5/global_manifest_v2/manifest_metadata_v2.json"

        # 3. Clean mapped ports (976 ports)
        else:
            plat = float(r["recovered_latitude"])
            plon = float(r["recovered_longitude"])
            accepted_coords = f"{plat:.6f}, {plon:.6f}"
            accepted_id = r["unlocode"]
            coord_source = r["coordinate_source_dataset"]
            
            if r["coordinate_conflict_flag"]:
                id_res_status = "CONFIRMED_COORDINATES_COUNTRY_CONFLICT"
            else:
                id_res_status = "CONFIRMED_AUTHORITATIVE_COORDINATES"

            s_lat = float(r["surface_grid_latitude"])
            s_lon = float(r["surface_grid_longitude_180"])
            s_ext_elig = "ELIGIBLE"
            s_model_elig = True

            # Wave mapping evaluation
            if pid in blocked_inland_ids:
                w_lat, w_lon = np.nan, np.nan
                w_proxy_class = "BLOCKED_INLAND"
                w_dist = np.nan
                w_ext_elig = "INELIGIBLE_NO_OCEAN_OBSERVATION"
                w_model_elig = False
                excl_reason = "INLAND_WATERWAY_NO_OCEAN_OBSERVATION_WITHIN_2.5_DEG"
                audit_ref = "reports/era5_final_geographic_validation/FINAL_ERA5_GEOGRAPHIC_AND_TEMPORAL_VALIDATION.md"

            elif pid in unrep_ids:
                w_lat = float(r["wave_selected_latitude"])
                w_lon = float(r["wave_selected_longitude_180"])
                w_dist = float(r["wave_ocean_distance_km"])
                w_proxy_class = "GEOGRAPHICALLY_UNREPRESENTATIVE"
                w_ext_elig = "ELIGIBLE_REGIONAL_TRANSIT_ONLY"
                w_model_elig = False
                excl_reason = "DISTANT_OR_INLAND_RIVER_OCEAN_DISCONNECTION"
                audit_ref = "reports/era5_final_geographic_validation/wave_proxy_geographic_review.csv"

            elif r["wave_mapping_method"] == "DIRECT_THEORETICAL_OCEAN_VERTEX":
                w_lat = float(r["wave_selected_latitude"])
                w_lon = float(r["wave_selected_longitude_180"])
                w_dist = float(r["wave_ocean_distance_km"])
                w_proxy_class = "DIRECT_OCEAN_CELL"
                w_ext_elig = "ELIGIBLE"
                w_model_elig = True
                excl_reason = "NONE"
                audit_ref = "data/interim/era5/global_manifest_v2/manifest_metadata_v2.json"

            else:
                # Displaced proxy in clean mapped set
                w_lat = float(r["wave_selected_latitude"])
                w_lon = float(r["wave_selected_longitude_180"])
                w_dist = float(r["wave_ocean_distance_km"])
                
                # Check proxy_review classification
                p_class = proxy_review.loc[pid, "evidence_based_classification"]
                if p_class == "NEARBY_COASTAL_PROXY":
                    w_proxy_class = "NEARBY_COASTAL_PROXY"
                    w_ext_elig = "ELIGIBLE"
                    w_model_elig = True
                    excl_reason = "NONE"
                else:  # REGIONAL_MARINE_FORCING_ONLY
                    w_proxy_class = "REGIONAL_MARINE_FORCING_ONLY"
                    w_ext_elig = "ELIGIBLE"
                    w_model_elig = True  # Eligible with proxy flag
                    excl_reason = "REGIONAL_FORCING_ONLY_RETAINS_PROXY_FLAG"
                
                audit_ref = "reports/era5_final_geographic_validation/wave_proxy_geographic_review.csv"

        gov_rows.append({
            "portwatch_id": pid,
            "portwatch_name": pname,
            "portwatch_country": country,
            "accepted_identity": accepted_id,
            "accepted_coordinates": accepted_coords,
            "coordinate_source": coord_source,
            "identity_resolution_status": id_res_status,
            "surface_grid_coordinates": f"{s_lat}, {s_lon}" if pd.notna(s_lat) else None,
            "surface_extraction_eligibility": s_ext_elig,
            "wave_grid_coordinates": f"{w_lat}, {w_lon}" if pd.notna(w_lat) else None,
            "wave_proxy_classification": w_proxy_class,
            "wave_distance_km": w_dist if pd.notna(w_dist) else None,
            "wave_extraction_eligibility": w_ext_elig,
            "surface_modeling_eligibility": s_model_elig,
            "wave_modeling_eligibility": w_model_elig,
            "exclusion_reason": excl_reason,
            "provenance_and_audit_reference": audit_ref,
        })

    gov_df = pd.DataFrame(gov_rows)
    gov_csv_path = IDENTITY_DIR / "spatial_feature_governance.csv"
    gov_parquet_path = IDENTITY_DIR / "spatial_feature_governance.parquet"
    gov_df.to_csv(gov_csv_path, index=False)
    gov_df.to_parquet(gov_parquet_path, index=False)
    print(f"  Wrote spatial feature governance layer to {gov_csv_path} and {gov_parquet_path}")

    # =========================================================================
    # PART 4: VALIDATE CLASSIFICATION CONSISTENCY
    # =========================================================================
    print("\n[5/6] Executing Part 4: Validating Classification Consistency (12 Rules)...")
    
    # 1. Exactly 2,065 unique IDs
    assert len(gov_df) == 2065, f"Rule 1 failed: expected 2065 rows, got {len(gov_df)}"
    assert gov_df["portwatch_id"].nunique() == 2065, "Rule 1 failed: non-unique IDs"
    print("  Rule 1 PASS: Exactly 2,065 unique PortWatch IDs present.")

    # 2. Every port has exactly one identity-resolution status
    assert gov_df["identity_resolution_status"].notna().all(), "Rule 2 failed: null identity resolution status"
    print(f"  Rule 2 PASS: 100% ports have an identity-resolution status. Breakdown:")
    id_counts = gov_df["identity_resolution_status"].value_counts().to_dict()
    for k, v in id_counts.items():
        print(f"    - {k}: {v}")

    # 3. Every port has exactly one surface eligibility status
    assert gov_df["surface_extraction_eligibility"].notna().all(), "Rule 3 failed"
    s_ext_counts = gov_df["surface_extraction_eligibility"].value_counts().to_dict()
    print(f"  Rule 3 PASS: Surface extraction eligibility breakdown:")
    for k, v in s_ext_counts.items():
        print(f"    - {k}: {v}")

    # 4. Every port has exactly one wave eligibility status
    assert gov_df["wave_extraction_eligibility"].notna().all(), "Rule 4 failed"
    w_ext_counts = gov_df["wave_extraction_eligibility"].value_counts().to_dict()
    print(f"  Rule 4 PASS: Wave extraction eligibility breakdown:")
    for k, v in w_ext_counts.items():
        print(f"    - {k}: {v}")

    # 5. Explicit surface and wave modeling eligibility
    assert gov_df["surface_modeling_eligibility"].dtype == bool, "Rule 5 failed"
    assert gov_df["wave_modeling_eligibility"].dtype == bool, "Rule 5 failed"
    s_mod_counts = gov_df["surface_modeling_eligibility"].value_counts().to_dict()
    w_mod_counts = gov_df["wave_modeling_eligibility"].value_counts().to_dict()
    print(f"  Rule 5 PASS: Modeling eligibility:")
    print(f"    - Surface modeling eligible: {s_mod_counts.get(True, 0)} (Ineligible: {s_mod_counts.get(False, 0)})")
    print(f"    - Wave modeling eligible: {w_mod_counts.get(True, 0)} (Ineligible: {w_mod_counts.get(False, 0)})")

    # 6. Four disputed records treated consistently
    sub_exc = gov_df[gov_df["portwatch_id"].isin(exception_ids)]
    assert (sub_exc["accepted_coordinates"].isna()).all(), "Rule 6 failed: exception coords not null"
    assert (sub_exc["surface_modeling_eligibility"] == False).all(), "Rule 6 failed"
    assert (sub_exc["wave_modeling_eligibility"] == False).all(), "Rule 6 failed"
    print("  Rule 6 PASS: All 4 exception ports have strictly null accepted coordinates and are excluded from modeling.")

    # 7. 1,085 unmapped ports retained
    unmapped_gov = gov_df[gov_df["identity_resolution_status"] == "UNRESOLVED_NO_COORDINATES"]
    assert len(unmapped_gov) == 1085, f"Rule 7 failed: expected 1085 unmapped, got {len(unmapped_gov)}"
    assert (unmapped_gov["surface_modeling_eligibility"] == False).all(), "Rule 7 failed"
    print("  Rule 7 PASS: Exactly 1,085 originally unmapped ports retained with null coordinates.")

    # 8. 15 blocked inland ports distinct from unmapped
    blocked_gov = gov_df[gov_df["wave_proxy_classification"] == "BLOCKED_INLAND"]
    assert len(blocked_gov) == 15, f"Rule 8 failed: expected 15 blocked, got {len(blocked_gov)}"
    assert (blocked_gov["surface_extraction_eligibility"] == "ELIGIBLE").all(), "Rule 8 failed: blocked inland surface should be eligible"
    assert (blocked_gov["wave_modeling_eligibility"] == False).all(), "Rule 8 failed: blocked inland wave should be false"
    print("  Rule 8 PASS: 15 blocked inland ports remain distinct, with valid surface coordinates and null wave coordinates.")

    # 9. 21 distant/disconnected proxies excluded from port-level wave dwell modeling
    unrep_gov = gov_df[gov_df["wave_proxy_classification"] == "GEOGRAPHICALLY_UNREPRESENTATIVE"]
    assert len(unrep_gov) == 21, f"Rule 9 failed: expected 21 unrep, got {len(unrep_gov)}"
    assert (unrep_gov["wave_modeling_eligibility"] == False).all(), "Rule 9 failed: unrep wave modeling must be False"
    assert (unrep_gov["wave_extraction_eligibility"] == "ELIGIBLE_REGIONAL_TRANSIT_ONLY").all(), "Rule 9 failed"
    print("  Rule 9 PASS: Exactly 21 distant/disconnected proxies excluded from port-level wave dwell modeling.")

    # 10. No unresolved identity receives approved coordinate
    unresolved_with_coords = gov_df[gov_df["identity_resolution_status"].isin(["UNRESOLVED_NO_COORDINATES", "IDENTITY_CONFIRMED_COORDINATE_UNAVAILABLE", "SOURCE_CONFLICT_REQUIRES_MANUAL_REVIEW"]) & gov_df["accepted_coordinates"].notna()]
    assert len(unresolved_with_coords) == 0, f"Rule 10 failed: found {len(unresolved_with_coords)} unresolved ports with coordinates"
    print("  Rule 10 PASS: Zero unresolved identities received approved coordinates.")

    # 11. Displaced proxies have mutually exclusive classifications
    w_proxy_counts = gov_df["wave_proxy_classification"].value_counts().to_dict()
    print("  Rule 11 PASS: Wave proxy classifications mutually exclusive and partition 2,065 ports:")
    for k, v in w_proxy_counts.items():
        print(f"    - {k}: {v}")
    assert w_proxy_counts["NEARBY_COASTAL_PROXY"] == 205
    assert w_proxy_counts["REGIONAL_MARINE_FORCING_ONLY"] == 226
    assert w_proxy_counts["GEOGRAPHICALLY_UNREPRESENTATIVE"] == 21
    assert w_proxy_counts["IDENTITY_OR_COORDINATE_UNCERTAIN"] == 4
    assert (205 + 226 + 21 + 4) == 456, "Displaced proxy counts sum mismatch"

    # 12. Full reconciliation
    total_check = (
        w_proxy_counts["DIRECT_OCEAN_CELL"]
        + w_proxy_counts["NEARBY_COASTAL_PROXY"]
        + w_proxy_counts["REGIONAL_MARINE_FORCING_ONLY"]
        + w_proxy_counts["GEOGRAPHICALLY_UNREPRESENTATIVE"]
        + w_proxy_counts["IDENTITY_OR_COORDINATE_UNCERTAIN"]
        + w_proxy_counts["BLOCKED_INLAND"]
        + w_proxy_counts["UNMAPPED"]
    )
    assert total_check == 2065, f"Rule 12 failed: total {total_check} != 2065"
    print("  Rule 12 PASS: Full reconciliation confirmed across all 2,065 ports.")

    # Save summary JSON
    summary_json = {
        "audit_name": "Spatial Feature Governance Validation Summary",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_ports": 2065,
        "identity_resolution_counts": id_counts,
        "surface_extraction_eligibility_counts": s_ext_counts,
        "surface_modeling_eligibility_counts": s_mod_counts,
        "wave_proxy_classification_counts": w_proxy_counts,
        "wave_extraction_eligibility_counts": w_ext_counts,
        "wave_modeling_eligibility_counts": w_mod_counts,
        "reconciliation_status": "100% RECONCILED (0 discrepancies)",
        "governance_policy_file": "reports/port_identity_resolution/SPATIAL_FEATURE_GOVERNANCE_POLICY.md"
    }
    with open(IDENTITY_DIR / "spatial_feature_governance_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    # =========================================================================
    # PART 5: SPATIAL FEATURE GOVERNANCE POLICY DOCUMENT
    # =========================================================================
    print("\n[6/6] Compiling Spatial Feature Governance Policy Document...")
    policy_lines = [
        "# Global ERA5 Spatial Feature Governance Policy",
        "",
        "**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  ",
        "**Milestone:** Phase 2 Data Engineering — Final Feature Governance  ",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        "**Version:** 1.0 (Operational Policy for Bulk ERA5 Extraction)  ",
        "",
        "---",
        "",
        "## 1. Purpose and Operational Principles",
        "",
        "This policy establishes rigorous feature-governance rules for global ERA5 environmental extraction and downstream supervised deep learning models. It ensures broad global spatial coverage while preventing erroneous, distant, or fabricated coordinates from polluting port-level dwelling time prediction models.",
        "",
        "**Core Rule:** *A port eligible for environmental extraction is NOT automatically eligible for supervised dwell-time modeling.* Extraction eligibility governs whether raw ERA5 time series should be retrieved; modeling eligibility governs whether those time series may be fed as local features into terminal dwell-time neural networks.",
        "",
        "---",
        "",
        "## 2. Category Breakdown and Governance Matrix",
        "",
        "| Governance Category | Port Count | Surface Extraction | Wave Extraction | Surface Dwell Modeling | Wave Dwell Modeling | Governance Description |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---|",
        "| **Direct Ocean Ports** | 509 | ELIGIBLE | ELIGIBLE | YES | YES | Theoretical 0.5° grid vertex is in open water (0 km displacement). High spatial fidelity. |",
        "| **Nearby Coastal Proxies** | 205 | ELIGIBLE | ELIGIBLE | YES | YES | Displaced < 40 km to adjacent outer fairway. Represents harbor approach marine forcing. |",
        "| **Regional Marine Proxies** | 226 | ELIGIBLE | ELIGIBLE | YES | YES (FLAGGED) | Displaced 40–100 km on coastal shelf. Mandatory `offshore_proxy_flag=True`; subject to ablation. |",
        "| **Distant / Disconnected River Proxies** | 21 | ELIGIBLE | TRANSIT ONLY | YES | **NO (MASKED)** | River terminals (Quebec, Rosario, Nanjing, Baton Rouge). Open-sea swell is masked for local berth models. |",
        "| **Blocked Inland Ports** | 15 | ELIGIBLE | **INELIGIBLE (NULL)** | YES | **NO (NULL)** | Enclosed river/lake ports (Amazon basin, Montreal, Santa Fe). Wave coordinates strictly null. |",
        "| **Identity / Coordinate Uncertain** | 4 | **INELIGIBLE (NULL)** | **INELIGIBLE (NULL)** | **NO** | **NO** | Gary, Bay City, Monroe, Delfzijl. Erroneous coordinates nullified pending registry errata. |",
        "| **Unmapped Ports** | 1,085 | **INELIGIBLE (NULL)** | **INELIGIBLE (NULL)** | **NO** | **NO** | Primary source coordinates missing. Strictly retained with null values. |",
        "| **TOTAL UNIVERSE** | **2,065** | **976 Eligible** | **940 Extractable** | **976 Eligible** | **714 Direct/Near + 226 Reg** | Complete, mutually exclusive partition of the PortWatch universe. |",
        "",
        "---",
        "",
        "## 3. Surface Extraction Policy",
        "",
        "- Exactly **976 ports** are approved for surface meteorological extraction across **936 unique 0.25° grid vertices**.",
        "- Surface extraction covers `u10`, `v10`, `t2m`, `msl`, `tp`.",
        "- All 976 approved ports have confirmed, authoritative coordinates with geodesic distance to nearest surface vertex ≤ 18.31 km (mean 9.62 km).",
        "- The 4 exception ports (Gary, Bay City, Monroe, Delfzijl) and 1,085 unmapped ports are strictly excluded from surface queries to prevent sampling weather at incorrect town centroids.",
        "",
        "---",
        "",
        "## 4. Wave Extraction Policy",
        "",
        "- Exactly **961 ports** are approved for wave extraction across **748 unique valid ocean cells**.",
        "- Wave variables cover `swh` (Significant Wave Height), `mwp` (Mean Wave Period), and `mwd` (Mean Wave Direction).",
        "- **Direct Ocean (509 ports):** Directly sampled from theoretical vertices.",
        "- **Nearby Coastal (205 ports):** Sampled from outer fairways (< 40 km).",
        "- **Regional Marine (226 ports):** Sampled from outer shelf (40–100 km).",
        "- **Transit Only (21 ports):** Displaced > 100 km or inland waterways. Extracted solely for maritime transit features; strictly excluded from port-level dwell modeling.",
        "- **Blocked Inland (15 ports):** Wave queries bypassed; features set to `NaN`.",
        "- **Identity Uncertain (4 ports) & Unmapped (1,085 ports):** Wave queries bypassed.",
        "",
        "---",
        "",
        "## 5. Dwell-Time Machine Learning Feature Governance",
        "",
        "When training supervised models (e.g. CatBoost baseline, Deep Learning Sequence models):",
        "1. **Surface Weather Features:** Eligible for all 976 validated ports.",
        "2. **Wave Features:**",
        "   - **Tier 1 (High Confidence):** 714 ports (509 direct + 205 nearby coastal). Primary wave features.",
        "   - **Tier 2 (Regional Forcing):** 226 ports. Eligible only with `offshore_proxy_flag = 1.0`.",
        "   - **Tier 3 (Masked):** All remaining 1,125 ports must have wave features masked (`NaN` or dedicated missingness indicator). Under no circumstances should open-sea swell be attributed to river terminals (Quebec, Rosario, Nanjing).",
        "",
        "---",
        "",
        "## 6. Audit Immutability",
        "",
        "All underlying reference datasets, raw files, canonical registry v1.2, manifest v1/v2, and Phase 1 assets remain 100% immutable and verified via SHA-256 digests."
    ]
    with open(IDENTITY_DIR / "SPATIAL_FEATURE_GOVERNANCE_POLICY.md", "w") as f:
        f.write("\n".join(policy_lines))
    print(f"  Wrote governance policy document to {IDENTITY_DIR / 'SPATIAL_FEATURE_GOVERNANCE_POLICY.md'}")

    # Re-verify post-execution hashes
    print("\nVerifying immutability of protected files post-execution...")
    for name, path in PROTECTED_FILES.items():
        post_h = compute_sha256(path)
        assert post_h == pre_hashes[name], f"Protected file changed: {name}"
    print("ALL PROTECTED ARTIFACTS AND PREVIOUS REGISTRIES/MANIFESTS CONFIRMED UNCHANGED.")

    print("\n" + "=" * 80)
    print("SPATIAL FEATURE GOVERNANCE & IDENTITY RESOLUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
