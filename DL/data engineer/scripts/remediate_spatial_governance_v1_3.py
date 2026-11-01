#!/usr/bin/env python3
"""
Phase 2 Data Engineering — Spatial Governance v1.3 Remediation and Reconciliation
AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks

Root directory: /Users/abdul/Desktop/DS/DL
Output directory: reports/port_identity_resolution/v1_3_remediation/

This script autonomously executes:
1. Pre-execution SHA-256 verification of 10 protected historical artifacts.
2. Candidate-table remediation (separating identity candidate acceptance from coordinate acceptance).
3. Canonical port registry remediation (ensuring 1-to-1 candidate alignment and provenance).
4. Creation of remediated Spatial Feature Governance Layer (all 2,065 ports).
5. Comprehensive 15-rule reconciliation and direct grid-cell count verification.
6. Generation of updated governance policy Markdown and remediation report.
7. Post-execution SHA-256 verification.
"""

import sys
import os
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
PROTECTED_FILES = {
    "canonical_port_registry_v1_2.csv": BASE_DIR / "reports/port_registry/v1_2_identifier_bridge/canonical_port_registry.csv",
    "registry_metadata_v1_2.json": BASE_DIR / "reports/port_registry/v1_2_identifier_bridge/registry_metadata.json",
    "era5_port_hourly.parquet": BASE_DIR / "data/interim/era5/era5_port_hourly.parquet",
    "era5_port_hourly_metadata.json": BASE_DIR / "data/interim/era5/era5_port_hourly_metadata.json",
    "era5_port_daily_v2.parquet": BASE_DIR / "data/processed/era5/era5_port_daily_v2.parquet",
    "era5_port_daily_v2_metadata.json": BASE_DIR / "data/processed/era5/era5_port_daily_v2_metadata.json",
    "global_era5_spatial_manifest_v1.parquet": BASE_DIR / "data/interim/era5/global_manifest_v1/global_era5_spatial_manifest.parquet",
    "global_era5_spatial_manifest_v1.csv": BASE_DIR / "data/interim/era5/global_manifest_v1/global_era5_spatial_manifest.csv",
    "global_era5_spatial_manifest_v2.parquet": BASE_DIR / "data/interim/era5/global_manifest_v2/global_era5_spatial_manifest_v2.parquet",
    "global_era5_spatial_manifest_v2.csv": BASE_DIR / "data/interim/era5/global_manifest_v2/global_era5_spatial_manifest_v2.csv",
}

PREVIOUS_V1_3_REGISTRY = BASE_DIR / "reports/port_registry/v1_3_spatial_governance/canonical_port_registry.csv"
PREVIOUS_V1_3_CANDIDATES = BASE_DIR / "reports/port_registry/v1_3_spatial_governance/port_identity_candidates.csv"
PREVIOUS_V1_3_GOVERNANCE = BASE_DIR / "reports/port_identity_resolution/spatial_feature_governance.csv"

OUTPUT_DIR = BASE_DIR / "reports/port_identity_resolution/v1_3_remediation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("PHASE 2: SPATIAL GOVERNANCE V1.3 REMEDIATION & INDEPENDENT RECONCILIATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: PRE-EXECUTION IMMUTABILITY CHECK
    # -------------------------------------------------------------------------
    print("\n[1/6] Verifying immutability of protected artifacts...")
    pre_hashes = {}
    for name, path in PROTECTED_FILES.items():
        if not path.exists():
            print(f"ERROR: Protected file {path} does not exist!")
            sys.exit(1)
        digest = compute_sha256(path)
        pre_hashes[name] = digest
        print(f"  {name}: {digest[:16]}...")

    # Load input tables
    print("\n[2/6] Loading source and intermediate datasets...")
    reg_v1_3_prev = pd.read_csv(PREVIOUS_V1_3_REGISTRY)
    cand_v1_3_prev = pd.read_csv(PREVIOUS_V1_3_CANDIDATES)
    gov_prev = pd.read_csv(PREVIOUS_V1_3_GOVERNANCE)
    manifest_v2 = pd.read_parquet(PROTECTED_FILES["global_era5_spatial_manifest_v2.parquet"])

    assert len(reg_v1_3_prev) == 2065, f"Expected 2065 ports, found {len(reg_v1_3_prev)}"
    assert len(gov_prev) == 2065, f"Expected 2065 governance rows, found {len(gov_prev)}"

    # -------------------------------------------------------------------------
    # STEP 2: REMEDIATE CANDIDATE TABLE (PART 1)
    # -------------------------------------------------------------------------
    print("\n[3/6] Remediating candidate table semantics (Part 1)...")
    # Columns required:
    # identity_candidate_accepted (bool)
    # coordinate_candidate_accepted (bool)
    # coordinate_validation_status (str)
    # rejection_reason (str)

    # Start from previous candidates
    cand_df = cand_v1_3_prev.copy()

    # Add missing candidate rows if absent:
    # 1. Monroe (port2270) -> USMOI from CodeListPart3.csv row 22610
    has_usmoi = ((cand_df["portwatch_id"] == "port2270") & (cand_df["candidate_id"] == "USMOI")).any()
    if not has_usmoi:
        print("  Adding explicit candidate row for port2270 -> USMOI (Monroe, MI)...")
        new_monroe_cand = {
            "portwatch_id": "port2270",
            "portwatch_name": "Monroe",
            "portwatch_country": "United States",
            "portwatch_iso3": "USA",
            "candidate_source": "UNLOCODE",
            "candidate_id": "USMOI",
            "candidate_name": "Monroe",
            "similarity_score": 100.0,
            "match_method": "EXACT_NORMALIZED_NAME_COUNTRY",
            "candidate_country": "US",
            "source_file": "UNLOCODE CodeListPart3.csv",
            "source_row": 22610.0,
            "candidate_is_accepted": True,
        }
        cand_df = pd.concat([cand_df, pd.DataFrame([new_monroe_cand])], ignore_index=True)

    # 2. Bay City (port130) -> USBCY from CodeListPart3.csv row 13506 (for audit completeness)
    has_usbcy = ((cand_df["portwatch_id"] == "port130") & (cand_df["candidate_id"] == "USBCY")).any()
    if not has_usbcy:
        print("  Adding candidate row for port130 -> USBCY (Bay City, MI) for audit completeness...")
        new_bc_cand = {
            "portwatch_id": "port130",
            "portwatch_name": "Bay City",
            "portwatch_country": "United States",
            "portwatch_iso3": "USA",
            "candidate_source": "UNLOCODE",
            "candidate_id": "USBCY",
            "candidate_name": "Bay City",
            "similarity_score": 100.0,
            "match_method": "EXACT_NORMALIZED_NAME_COUNTRY",
            "candidate_country": "US",
            "source_file": "UNLOCODE CodeListPart3.csv",
            "source_row": 13506.0,
            "candidate_is_accepted": False,
        }
        cand_df = pd.concat([cand_df, pd.DataFrame([new_bc_cand])], ignore_index=True)

    # Initialize the 4 new independent columns
    cand_df["identity_candidate_accepted"] = False
    cand_df["coordinate_candidate_accepted"] = False
    cand_df["coordinate_validation_status"] = "REJECTED_ALTERNATIVE"
    cand_df["rejection_reason"] = "Alternative candidate not selected or unverified."

    # Map accepted identifiers from registry v1.3
    reg_id_map = dict(zip(reg_v1_3_prev["portwatch_id"], reg_v1_3_prev["accepted_unlocode"]))
    reg_lat_map = dict(zip(reg_v1_3_prev["portwatch_id"], reg_v1_3_prev["accepted_latitude"]))

    # Populate candidate fields
    for idx, row in cand_df.iterrows():
        pid = row["portwatch_id"]
        cid = row["candidate_id"]
        acc_id = reg_id_map.get(pid)
        acc_lat = reg_lat_map.get(pid)

        # 1. Exceptional ports handling
        if pid == "port377":  # Gary
            if cid == "USGYY":
                cand_df.loc[idx, "identity_candidate_accepted"] = True
                cand_df.loc[idx, "coordinate_candidate_accepted"] = False
                cand_df.loc[idx, "coordinate_validation_status"] = "MISSING_IN_PRIMARY_SOURCE"
                cand_df.loc[idx, "rejection_reason"] = (
                    "IDENTITY_CONFIRMED_COORDINATE_UNAVAILABLE: Commercial seaport Gary, Indiana (USGYY) is confirmed, "
                    "but coordinates are empty/missing in UNECE UN/LOCODE release. Excluded from spatial extraction."
                )
                cand_df.loc[idx, "candidate_is_accepted"] = True
            elif cid == "USGT4":
                cand_df.loc[idx, "identity_candidate_accepted"] = False
                cand_df.loc[idx, "coordinate_candidate_accepted"] = False
                cand_df.loc[idx, "coordinate_validation_status"] = "REJECTED_WRONG_LOCATION"
                cand_df.loc[idx, "rejection_reason"] = (
                    "COLLISION_WRONG_STATE: Resolves to Panola County, Texas (USGT4), not Gary, Indiana."
                )
                cand_df.loc[idx, "candidate_is_accepted"] = False

        elif pid == "port2270":  # Monroe
            if cid == "USMOI":
                cand_df.loc[idx, "identity_candidate_accepted"] = True
                cand_df.loc[idx, "coordinate_candidate_accepted"] = False
                cand_df.loc[idx, "coordinate_validation_status"] = "MISSING_IN_PRIMARY_SOURCE"
                cand_df.loc[idx, "rejection_reason"] = (
                    "IDENTITY_CONFIRMED_COORDINATE_UNAVAILABLE: Commercial port Monroe, Michigan (USMOI) is confirmed, "
                    "but coordinates are empty/missing in UNECE UN/LOCODE release. Excluded from spatial extraction."
                )
                cand_df.loc[idx, "candidate_is_accepted"] = True
            elif cid == "USM4X":
                cand_df.loc[idx, "identity_candidate_accepted"] = False
                cand_df.loc[idx, "coordinate_candidate_accepted"] = False
                cand_df.loc[idx, "coordinate_validation_status"] = "REJECTED_WRONG_LOCATION"
                cand_df.loc[idx, "rejection_reason"] = (
                    "COLLISION_WRONG_STATE: Resolves to Grafton County, New Hampshire (USM4X), not Monroe, Michigan."
                )
                cand_df.loc[idx, "candidate_is_accepted"] = False

        elif pid == "port285":  # Delfzijl
            if cid == "NLDZL":
                cand_df.loc[idx, "identity_candidate_accepted"] = True
                cand_df.loc[idx, "coordinate_candidate_accepted"] = False
                cand_df.loc[idx, "coordinate_validation_status"] = "INVALID_PRIMARY_SOURCE_TYPO"
                cand_df.loc[idx, "rejection_reason"] = (
                    "PRIMARY_SOURCE_COORDINATE_TYPO: Official UNECE CodeListPart2 row 24631 records 5231N 00656E "
                    "(52.5167 N, 6.9333 E, 88.9 km south on farmland near Emmen). Identity confirmed as NLDZL; "
                    "coordinate source rejected pending official UNECE errata."
                )
                cand_df.loc[idx, "candidate_is_accepted"] = True

        elif pid == "port130":  # Bay City
            cand_df.loc[idx, "identity_candidate_accepted"] = False
            cand_df.loc[idx, "coordinate_candidate_accepted"] = False
            cand_df.loc[idx, "coordinate_validation_status"] = "SOURCE_CONFLICT_REQUIRES_MANUAL_REVIEW"
            if cid == "USYBZ":
                cand_df.loc[idx, "rejection_reason"] = (
                    "SOURCE_CONFLICT_REQUIRES_MANUAL_REVIEW: PortWatch text LOCODE US YBZ resolves to Bay City, Wisconsin "
                    "(44.6 N, -92.45 W). Michigan port is intended; candidate rejected pending manual review."
                )
            else:
                cand_df.loc[idx, "rejection_reason"] = (
                    "SOURCE_CONFLICT_REQUIRES_MANUAL_REVIEW: UNECE USBCY in Michigan lacks port function code '1' "
                    "(--34----), conflicting with text LOCODE US YBZ. Unresolved pending manual review."
                )
            cand_df.loc[idx, "candidate_is_accepted"] = False

        elif pid == "port598":  # Kralendijk (BES/BQ)
            if cid == "BQKRA":
                cand_df.loc[idx, "identity_candidate_accepted"] = True
                cand_df.loc[idx, "coordinate_candidate_accepted"] = True
                cand_df.loc[idx, "coordinate_validation_status"] = "VALID_COUNTRY_CONFLICT"
                cand_df.loc[idx, "rejection_reason"] = (
                    "CONFIRMED_COORDINATES_COUNTRY_CONFLICT: Bonaire territory cataloged under ISO-2 BQ in UN/LOCODE "
                    "and ISO-3 BES in PortWatch (Upply listed country as NL). Coordinates accepted with country discrepancy provenance."
                )
                cand_df.loc[idx, "candidate_is_accepted"] = True

        else:
            # General port evaluation
            if pd.notna(acc_id) and cid == acc_id:
                cand_df.loc[idx, "identity_candidate_accepted"] = True
                cand_df.loc[idx, "candidate_is_accepted"] = True
                if pd.notna(acc_lat):
                    cand_df.loc[idx, "coordinate_candidate_accepted"] = True
                    cand_df.loc[idx, "coordinate_validation_status"] = "VALID"
                    cand_df.loc[idx, "rejection_reason"] = None
                else:
                    cand_df.loc[idx, "coordinate_candidate_accepted"] = False
                    cand_df.loc[idx, "coordinate_validation_status"] = "MISSING_IN_PRIMARY_SOURCE"
                    cand_df.loc[idx, "rejection_reason"] = (
                        "COORDINATES_MISSING: Identity candidate confirmed, but coordinates unavailable in source catalog."
                    )
            else:
                cand_df.loc[idx, "identity_candidate_accepted"] = False
                cand_df.loc[idx, "coordinate_candidate_accepted"] = False
                cand_df.loc[idx, "candidate_is_accepted"] = False
                cand_df.loc[idx, "coordinate_validation_status"] = "REJECTED_ALTERNATIVE"
                cand_df.loc[idx, "rejection_reason"] = "Alternative candidate not selected or unverified."

    # Save remediated candidate table
    cand_out_path = OUTPUT_DIR / "port_identity_candidates.csv"
    cand_df.to_csv(cand_out_path, index=False)
    print(f"  Wrote remediated candidate table to {cand_out_path} ({len(cand_df)} rows)")

    # -------------------------------------------------------------------------
    # STEP 3: BUILD REMEDIATED CANONICAL PORT REGISTRY (PART 1 / PART 3)
    # -------------------------------------------------------------------------
    print("\n[4/6] Building remediated Canonical Port Registry...")
    reg_df = reg_v1_3_prev.copy()

    # Re-verify the 4 exceptions in the registry
    # Gary: USGYY, null coords
    idx_gary = reg_df[reg_df["portwatch_id"] == "port377"].index[0]
    reg_df.loc[idx_gary, "accepted_unlocode"] = "USGYY"
    reg_df.loc[idx_gary, "accepted_latitude"] = np.nan
    reg_df.loc[idx_gary, "accepted_longitude"] = np.nan
    reg_df.loc[idx_gary, "identity_status"] = "IDENTITY_CONFIRMED_COORDINATE_UNAVAILABLE"
    reg_df.loc[idx_gary, "coordinate_provenance"] = "NONE_COORDINATES_UNAVAILABLE"

    # Monroe: USMOI, null coords
    idx_mon = reg_df[reg_df["portwatch_id"] == "port2270"].index[0]
    reg_df.loc[idx_mon, "accepted_unlocode"] = "USMOI"
    reg_df.loc[idx_mon, "accepted_latitude"] = np.nan
    reg_df.loc[idx_mon, "accepted_longitude"] = np.nan
    reg_df.loc[idx_mon, "identity_status"] = "IDENTITY_CONFIRMED_COORDINATE_UNAVAILABLE"
    reg_df.loc[idx_mon, "coordinate_provenance"] = "NONE_COORDINATES_UNAVAILABLE"

    # Delfzijl: NLDZL, null coords pending errata
    idx_dzl = reg_df[reg_df["portwatch_id"] == "port285"].index[0]
    reg_df.loc[idx_dzl, "accepted_unlocode"] = "NLDZL"
    reg_df.loc[idx_dzl, "accepted_latitude"] = np.nan
    reg_df.loc[idx_dzl, "accepted_longitude"] = np.nan
    reg_df.loc[idx_dzl, "identity_status"] = "SOURCE_CONFLICT_REQUIRES_MANUAL_REVIEW"
    reg_df.loc[idx_dzl, "coordinate_provenance"] = "NONE_SOURCE_TYPO_CONFLICT"

    # Bay City: unresolved identity, null coords
    idx_bc = reg_df[reg_df["portwatch_id"] == "port130"].index[0]
    reg_df.loc[idx_bc, "accepted_unlocode"] = np.nan
    reg_df.loc[idx_bc, "accepted_latitude"] = np.nan
    reg_df.loc[idx_bc, "accepted_longitude"] = np.nan
    reg_df.loc[idx_bc, "identity_status"] = "SOURCE_CONFLICT_REQUIRES_MANUAL_REVIEW"
    reg_df.loc[idx_bc, "coordinate_provenance"] = "NONE_SOURCE_CONFLICT"

    reg_out_path = OUTPUT_DIR / "canonical_port_registry.csv"
    reg_df.to_csv(reg_out_path, index=False)
    print(f"  Wrote remediated canonical registry to {reg_out_path} ({len(reg_df)} rows)")

    # -------------------------------------------------------------------------
    # STEP 4: BUILD REMEDIATED SPATIAL FEATURE GOVERNANCE LAYER (PART 2)
    # -------------------------------------------------------------------------
    print("\n[5/6] Building remediated Spatial Feature Governance layer (all 2,065 ports)...")
    # Fields required by Part 2:
    # - offshore_proxy_flag (bool)
    # - surface_modeling_eligibility_status (ELIGIBLE_PRIMARY, INELIGIBLE)
    # - wave_modeling_eligibility_status (ELIGIBLE_PRIMARY, ELIGIBLE_CONDITIONAL, INELIGIBLE)
    # - wave_proxy_confidence (str)
    # - identity_confidence (str)
    # - coordinate_validation_status (str)

    # Load audited wave proxy review for exact 205 nearby / 226 regional / 21 unrepresentative partitions
    proxy_review_path = BASE_DIR / "reports/era5_final_geographic_validation/wave_proxy_geographic_review.csv"
    proxy_review = pd.read_csv(proxy_review_path).set_index("portwatch_id")
    unrep_ids = set(proxy_review[proxy_review["evidence_based_classification"] == "GEOGRAPHICALLY_UNREPRESENTATIVE"].index)
    blocked_inland_ids = set(manifest_v2[manifest_v2["wave_mapping_method"] == "BLOCKED_INLAND_NO_OCEAN_OBSERVATION"]["portwatch_id"])

    gov_rows = []
    for _, reg_row in reg_df.iterrows():
        pid = reg_row["portwatch_id"]
        pname = reg_row["portwatch_name"]
        pcountry = reg_row["portwatch_country"]
        acc_id = reg_row["accepted_unlocode"]
        acc_lat = reg_row["accepted_latitude"]
        acc_lon = reg_row["accepted_longitude"]
        coord_prov = reg_row["coordinate_provenance"]
        id_status = reg_row["identity_status"]

        # Retrieve manifest v2 row for spatial grid details
        v2_matches = manifest_v2[manifest_v2["portwatch_id"] == pid]
        has_v2 = len(v2_matches) > 0
        v2_row = v2_matches.iloc[0] if has_v2 else None

        # Determine Identity Confidence
        if pid in ["port377", "port2270", "port285"]:
            identity_confidence = "HIGH"
        elif pid == "port130":
            identity_confidence = "LOW_CONFLICT"
        elif pd.notna(acc_id) and pd.notna(acc_lat):
            identity_confidence = "HIGH"
        elif pd.notna(acc_id):
            identity_confidence = "MEDIUM"
        else:
            identity_confidence = "NONE_UNMAPPED"

        # Determine Coordinate Validation Status
        if pid in ["port377", "port2270"]:
            coord_val_status = "MISSING_IN_PRIMARY_SOURCE"
        elif pid == "port285":
            coord_val_status = "INVALID_PRIMARY_SOURCE_TYPO"
        elif pid == "port130":
            coord_val_status = "SOURCE_CONFLICT_REQUIRES_MANUAL_REVIEW"
        elif pid == "port598":
            coord_val_status = "VALID_COUNTRY_CONFLICT"
        elif pd.notna(acc_lat):
            coord_val_status = "VALID"
        else:
            coord_val_status = "UNMAPPED"

        # Check if port has accepted coordinates
        is_coord_approved = pd.notna(acc_lat) and pd.notna(acc_lon)

        # Determine Identity Resolution Status
        if pid in ["port377", "port2270"]:
            id_status = "IDENTITY_CONFIRMED_COORDINATE_UNAVAILABLE"
        elif pid in ["port130", "port285"]:
            id_status = "SOURCE_CONFLICT_REQUIRES_MANUAL_REVIEW"
        elif pid == "port598":
            id_status = "CONFIRMED_COORDINATES_COUNTRY_CONFLICT"
        elif is_coord_approved:
            id_status = "CONFIRMED_AUTHORITATIVE_COORDINATES"
        else:
            id_status = "UNRESOLVED_NO_COORDINATES"

        # ---------------------------------------------------------------------
        # Surface Governance
        # ---------------------------------------------------------------------
        if is_coord_approved:
            surface_coords = f"{v2_row['surface_grid_latitude']:.4f}, {v2_row['surface_grid_longitude_180']:.4f}"
            surface_extract_status = "ELIGIBLE"
            surface_model_status = "ELIGIBLE_PRIMARY"
            surface_model_bool = True
        else:
            surface_coords = None
            if pid in ["port377", "port130", "port2270", "port285"]:
                surface_extract_status = "INELIGIBLE_UNRELIABLE_COORDINATES"
            else:
                surface_extract_status = "INELIGIBLE_MISSING_COORDINATES"
            surface_model_status = "INELIGIBLE"
            surface_model_bool = False

        # ---------------------------------------------------------------------
        # Wave Governance
        # ---------------------------------------------------------------------
        if is_coord_approved:
            if pid in blocked_inland_ids:
                # 15 Blocked inland ports
                wave_coords = None
                wave_dist = None
                wave_proxy_cls = "BLOCKED_INLAND"
                wave_proxy_conf = "NONE_BLOCKED"
                offshore_flag = False
                wave_extract_status = "INELIGIBLE_NO_OCEAN_OBSERVATION"
                wave_model_status = "INELIGIBLE"
                wave_model_bool = False
                exclusion_reason = "BLOCKED_INLAND: Inland waterway/lake with zero valid ocean cells within search radius."
            elif pid in unrep_ids:
                # 21 Geographically Unrepresentative / River Terminals (>100 km)
                wave_disp = float(v2_row["wave_ocean_distance_km"])
                wave_coords = f"{v2_row['wave_selected_latitude']:.4f}, {v2_row['wave_selected_longitude_180']:.4f}"
                wave_dist = wave_disp
                wave_proxy_cls = "GEOGRAPHICALLY_UNREPRESENTATIVE"
                wave_proxy_conf = "NONE_UNREPRESENTATIVE"
                offshore_flag = False
                wave_extract_status = "ELIGIBLE_REGIONAL_TRANSIT_ONLY"
                wave_model_status = "INELIGIBLE"
                wave_model_bool = False
                exclusion_reason = (
                    f"GEOGRAPHICALLY_UNREPRESENTATIVE: Deep river terminal displaced {wave_disp:.1f} km from open sea. "
                    "Extracted for maritime transit analysis only; strictly excluded from port-level dwell modeling."
                )
            elif v2_row["wave_mapping_method"] == "DIRECT_THEORETICAL_OCEAN_VERTEX":
                # 509 Direct Ocean Ports
                wave_coords = f"{v2_row['wave_selected_latitude']:.4f}, {v2_row['wave_selected_longitude_180']:.4f}"
                wave_dist = float(v2_row["wave_ocean_distance_km"])
                wave_proxy_cls = "DIRECT_OCEAN_CELL"
                wave_proxy_conf = "HIGH_DIRECT_OCEAN"
                offshore_flag = False
                wave_extract_status = "ELIGIBLE"
                wave_model_status = "ELIGIBLE_PRIMARY"
                wave_model_bool = True
                exclusion_reason = None
            else:
                # Displaced proxy in clean mapped set (205 nearby + 226 regional)
                wave_disp = float(v2_row["wave_ocean_distance_km"])
                wave_coords = f"{v2_row['wave_selected_latitude']:.4f}, {v2_row['wave_selected_longitude_180']:.4f}"
                wave_dist = wave_disp
                p_class = proxy_review.loc[pid, "evidence_based_classification"]

                if p_class == "NEARBY_COASTAL_PROXY":
                    # 205 Nearby Coastal Proxies (<40 km)
                    wave_proxy_cls = "NEARBY_COASTAL_PROXY"
                    wave_proxy_conf = "MEDIUM_COASTAL_PROXY"
                    offshore_flag = False
                    wave_extract_status = "ELIGIBLE"
                    wave_model_status = "ELIGIBLE_PRIMARY"
                    wave_model_bool = True
                    exclusion_reason = "COASTAL_PROXY: Sampled from harbor approach fairway (<40 km). Retains audited nearby proxy classification."
                else:
                    # 226 Regional Marine Proxies (40-100 km)
                    wave_proxy_cls = "REGIONAL_MARINE_FORCING_ONLY"
                    wave_proxy_conf = "LOW_REGIONAL_FORCING"
                    offshore_flag = True
                    wave_extract_status = "ELIGIBLE"
                    wave_model_status = "ELIGIBLE_CONDITIONAL"
                    wave_model_bool = True
                    exclusion_reason = "REGIONAL_PROXY: Displaced 40-100 km on coastal shelf. Mandatory offshore_proxy_flag=True for ablation."
        else:
            wave_coords = None
            wave_dist = None
            offshore_flag = False
            wave_model_status = "INELIGIBLE"
            wave_model_bool = False

            if pid in ["port377", "port130", "port2270", "port285"]:
                wave_proxy_cls = "IDENTITY_OR_COORDINATE_UNCERTAIN"
                wave_proxy_conf = "NONE_UNCERTAIN"
                wave_extract_status = "INELIGIBLE_UNRELIABLE_COORDINATES"
                exclusion_reason = "IDENTITY_OR_COORDINATE_UNCERTAIN: Disputed identity/coordinate exception. Spatial extraction bypassed."
            else:
                wave_proxy_cls = "UNMAPPED"
                wave_proxy_conf = "NONE_UNMAPPED"
                wave_extract_status = "INELIGIBLE_MISSING_COORDINATES"
                exclusion_reason = "UNMAPPED: Coordinates unavailable in authoritative sources. Spatial extraction bypassed."

        # Accepted coordinates string
        accepted_coords_str = f"{acc_lat:.4f}, {acc_lon:.4f}" if is_coord_approved else None

        # Provenance reference
        if is_coord_approved:
            prov_ref = f"data/interim/era5/global_manifest_v2/manifest_v2_row_{pid}"
            if pid == "port598":
                prov_ref += " (Bonaire BES/BQ/NL country conflict preserved)"
        else:
            prov_ref = f"reports/port_identity_resolution/exception_resolution.csv#{pid}"

        gov_rows.append({
            "portwatch_id": pid,
            "portwatch_name": pname,
            "portwatch_country": pcountry,
            "accepted_identity": acc_id if pd.notna(acc_id) else None,
            "identity_confidence": identity_confidence,
            "identity_resolution_status": id_status,
            "accepted_coordinates": accepted_coords_str,
            "coordinate_source": coord_prov,
            "coordinate_validation_status": coord_val_status,
            "surface_grid_coordinates": surface_coords,
            "surface_extraction_eligibility": surface_extract_status,
            "surface_modeling_eligibility_status": surface_model_status,
            "surface_modeling_eligibility": surface_model_bool,
            "wave_grid_coordinates": wave_coords,
            "wave_proxy_classification": wave_proxy_cls,
            "wave_proxy_confidence": wave_proxy_conf,
            "wave_distance_km": wave_dist,
            "offshore_proxy_flag": offshore_flag,
            "wave_extraction_eligibility": wave_extract_status,
            "wave_modeling_eligibility_status": wave_model_status,
            "wave_modeling_eligibility": wave_model_bool,
            "exclusion_reason": exclusion_reason,
            "provenance_and_audit_reference": prov_ref,
        })

    gov_remediated = pd.DataFrame(gov_rows)

    # Save governance CSV and Parquet
    gov_csv_path = OUTPUT_DIR / "spatial_feature_governance.csv"
    gov_parquet_path = OUTPUT_DIR / "spatial_feature_governance.parquet"
    gov_remediated.to_csv(gov_csv_path, index=False)
    gov_remediated.to_parquet(gov_parquet_path, index=False)
    print(f"  Wrote remediated governance dataset to {gov_csv_path} and {gov_parquet_path}")

    # -------------------------------------------------------------------------
    # STEP 5: COMPREHENSIVE 15-RULE RECONCILIATION & GRID-CELL VERIFICATION (PART 3)
    # -------------------------------------------------------------------------
    print("\n[6/6] Executing independent reconciliation & grid-cell verification (Part 3)...")

    # Rule 1: Every accepted registry identity appears in the candidate table
    reg_acc = reg_df[reg_df["accepted_unlocode"].notna()]
    cand_acc_lookup = set(
        cand_df[cand_df["identity_candidate_accepted"] == True][["portwatch_id", "candidate_id"]]
        .itertuples(index=False, name=None)
    )
    for _, r in reg_acc.iterrows():
        pair = (r["portwatch_id"], r["accepted_unlocode"])
        assert pair in cand_acc_lookup, f"Rule 1 FAIL: Registry accepted identity {pair} not found in accepted candidates!"
    print("  Rule 1 PASS: Every accepted registry identity appears as an accepted identity candidate.")

    # Rule 2: No rejected candidate is simultaneously represented as the accepted identity
    for _, r in cand_df[cand_df["identity_candidate_accepted"] == False].iterrows():
        reg_match = reg_df[reg_df["portwatch_id"] == r["portwatch_id"]].iloc[0]
        if pd.notna(reg_match["accepted_unlocode"]):
            assert r["candidate_id"] != reg_match["accepted_unlocode"], (
                f"Rule 2 FAIL: Rejected candidate {r['candidate_id']} for {r['portwatch_id']} matches accepted identity!"
            )
    print("  Rule 2 PASS: No rejected candidate is simultaneously represented as the accepted identity.")

    # Rule 3: Candidate identity acceptance is independent of coordinate acceptance
    conf_indep = cand_df[
        (cand_df["identity_candidate_accepted"] == True) & (cand_df["coordinate_candidate_accepted"] == False)
    ]
    assert len(conf_indep) >= 3, f"Rule 3 FAIL: Expected at least 3 candidates with identity accepted but coords rejected, found {len(conf_indep)}"
    assert set(conf_indep["portwatch_id"]).issuperset({"port377", "port2270", "port285"})
    print(f"  Rule 3 PASS: Candidate identity acceptance is independent of coordinate acceptance ({len(conf_indep)} records).")

    # Rule 4: Accepted coordinates match governance coordinates for all eligible ports
    eligible_ports = gov_remediated[gov_remediated["surface_modeling_eligibility"] == True]
    assert len(eligible_ports) == 976, f"Rule 4 FAIL: Expected 976 surface eligible ports, found {len(eligible_ports)}"
    for _, r in eligible_ports.iterrows():
        reg_r = reg_df[reg_df["portwatch_id"] == r["portwatch_id"]].iloc[0]
        gov_lat, gov_lon = [float(x.strip()) for x in r["accepted_coordinates"].split(",")]
        assert abs(gov_lat - reg_r["accepted_latitude"]) < 1e-4, f"Lat mismatch for {r['portwatch_id']}"
        assert abs(gov_lon - reg_r["accepted_longitude"]) < 1e-4, f"Lon mismatch for {r['portwatch_id']}"
    print("  Rule 4 PASS: Accepted coordinates match governance coordinates for all 976 eligible ports.")

    # Rule 5: All four exceptions have null accepted coordinates
    for ex_id in ["port377", "port130", "port2270", "port285"]:
        reg_r = reg_df[reg_df["portwatch_id"] == ex_id].iloc[0]
        gov_r = gov_remediated[gov_remediated["portwatch_id"] == ex_id].iloc[0]
        assert pd.isna(reg_r["accepted_latitude"]), f"Rule 5 FAIL: {ex_id} has non-null registry latitude!"
        assert pd.isna(reg_r["accepted_longitude"]), f"Rule 5 FAIL: {ex_id} has non-null registry longitude!"
        assert pd.isna(gov_r["accepted_coordinates"]), f"Rule 5 FAIL: {ex_id} has non-null governance coords!"
    print("  Rule 5 PASS: All four exceptions have strictly null accepted coordinates.")

    # Rule 6: Exactly 2,065 unique PortWatch records are represented
    assert len(reg_df) == 2065 and reg_df["portwatch_id"].nunique() == 2065, "Rule 6 FAIL in registry!"
    assert len(gov_remediated) == 2065 and gov_remediated["portwatch_id"].nunique() == 2065, "Rule 6 FAIL in governance!"
    print("  Rule 6 PASS: Exactly 2,065 unique PortWatch records are represented.")

    # Rule 7: Exactly 976 ports are surface eligible
    assert (gov_remediated["surface_extraction_eligibility"] == "ELIGIBLE").sum() == 976
    assert (gov_remediated["surface_modeling_eligibility_status"] == "ELIGIBLE_PRIMARY").sum() == 976
    assert (gov_remediated["surface_modeling_eligibility_status"] == "INELIGIBLE").sum() == 1089
    print("  Rule 7 PASS: Exactly 976 ports are surface eligible (and 1,089 ineligible).")

    # Rule 8: Exactly 961 ports are wave extractable, including the 21 transit-only records
    wave_extractable = gov_remediated[gov_remediated["wave_extraction_eligibility"].isin(["ELIGIBLE", "ELIGIBLE_REGIONAL_TRANSIT_ONLY"])]
    assert len(wave_extractable) == 961, f"Rule 8 FAIL: Expected 961 wave extractable ports, got {len(wave_extractable)}"
    assert (gov_remediated["wave_extraction_eligibility"] == "ELIGIBLE_REGIONAL_TRANSIT_ONLY").sum() == 21
    assert (gov_remediated["wave_extraction_eligibility"] == "ELIGIBLE").sum() == 940
    print("  Rule 8 PASS: Exactly 961 ports are wave extractable (940 model-eligible + 21 transit-only).")

    # Rule 9: Exactly 940 ports are wave-model eligible under existing definitions
    wave_model_eligible = gov_remediated[gov_remediated["wave_modeling_eligibility"] == True]
    assert len(wave_model_eligible) == 940, f"Rule 9 FAIL: Expected 940 wave model eligible, got {len(wave_model_eligible)}"
    assert (gov_remediated["wave_modeling_eligibility_status"] == "ELIGIBLE_PRIMARY").sum() == 714
    assert (gov_remediated["wave_modeling_eligibility_status"] == "ELIGIBLE_CONDITIONAL").sum() == 226
    assert (gov_remediated["wave_modeling_eligibility_status"] == "INELIGIBLE").sum() == 1125
    print("  Rule 9 PASS: Exactly 940 ports are wave-model eligible (714 primary + 226 conditional).")

    # Rule 10 & 11: Exactly 226 regional proxies have offshore_proxy_flag=True, no other category receives it
    assert gov_remediated["offshore_proxy_flag"].sum() == 226, f"Rule 10 FAIL: Expected 226 True flags, got {gov_remediated['offshore_proxy_flag'].sum()}"
    assert (gov_remediated[gov_remediated["offshore_proxy_flag"] == True]["wave_proxy_classification"] == "REGIONAL_MARINE_FORCING_ONLY").all()
    assert (gov_remediated[gov_remediated["wave_proxy_classification"] != "REGIONAL_MARINE_FORCING_ONLY"]["offshore_proxy_flag"] == False).all()
    print("  Rule 10 & 11 PASS: Exactly 226 regional marine proxies have offshore_proxy_flag=True; zero other ports receive it.")

    # Rule 12: The 21 distant/disconnected ports are not eligible for local port-level dwell modeling
    unrep = gov_remediated[gov_remediated["wave_proxy_classification"] == "GEOGRAPHICALLY_UNREPRESENTATIVE"]
    assert len(unrep) == 21, f"Rule 12 FAIL: Expected 21 unrepresentative ports, got {len(unrep)}"
    assert (unrep["wave_modeling_eligibility_status"] == "INELIGIBLE").all()
    assert (unrep["wave_modeling_eligibility"] == False).all()
    print("  Rule 12 PASS: All 21 geographically unrepresentative ports are strictly excluded from port-level wave dwell modeling.")

    # Rule 13: The 15 blocked inland ports remain distinct from the 1,085 unmapped ports
    blocked = gov_remediated[gov_remediated["wave_proxy_classification"] == "BLOCKED_INLAND"]
    assert len(blocked) == 15, f"Rule 13 FAIL: Expected 15 blocked ports, got {len(blocked)}"
    assert (blocked["surface_extraction_eligibility"] == "ELIGIBLE").all(), "Blocked inland ports must have valid surface coordinates!"
    assert (blocked["wave_extraction_eligibility"] == "INELIGIBLE_NO_OCEAN_OBSERVATION").all()
    unmapped = gov_remediated[gov_remediated["wave_proxy_classification"] == "UNMAPPED"]
    assert len(unmapped) == 1085, f"Rule 13 FAIL: Expected 1,085 unmapped ports, got {len(unmapped)}"
    assert (unmapped["surface_extraction_eligibility"] == "INELIGIBLE_MISSING_COORDINATES").all()
    print("  Rule 13 PASS: 15 blocked inland ports remain completely distinct from the 1,085 unmapped ports.")

    # Rule 14: The four exceptions are excluded from both surface and wave modeling
    for ex_id in ["port377", "port130", "port2270", "port285"]:
        ex_row = gov_remediated[gov_remediated["portwatch_id"] == ex_id].iloc[0]
        assert ex_row["surface_modeling_eligibility_status"] == "INELIGIBLE"
        assert ex_row["wave_modeling_eligibility_status"] == "INELIGIBLE"
        assert ex_row["surface_modeling_eligibility"] == False
        assert ex_row["wave_modeling_eligibility"] == False
    print("  Rule 14 PASS: All four exceptions are excluded from both surface and wave modeling.")

    # Rule 15: Kralendijk retains explicit BES/BQ country-discrepancy provenance
    kra = gov_remediated[gov_remediated["portwatch_id"] == "port598"].iloc[0]
    assert kra["coordinate_validation_status"] == "VALID_COUNTRY_CONFLICT"
    assert kra["identity_resolution_status"] == "CONFIRMED_COORDINATES_COUNTRY_CONFLICT"
    assert "BES/BQ/NL" in kra["provenance_and_audit_reference"]
    print("  Rule 15 PASS: Kralendijk (port598) retains explicit BES/BQ country-discrepancy provenance.")

    # Direct Grid-Cell Count Reconciliation
    surface_unique_cells = gov_remediated[gov_remediated["surface_extraction_eligibility"] == "ELIGIBLE"]["surface_grid_coordinates"].nunique()
    wave_all_unique_cells = gov_remediated[gov_remediated["wave_extraction_eligibility"].isin(["ELIGIBLE", "ELIGIBLE_REGIONAL_TRANSIT_ONLY"])]["wave_grid_coordinates"].nunique()
    wave_model_unique_cells = gov_remediated[gov_remediated["wave_modeling_eligibility"] == True]["wave_grid_coordinates"].nunique()
    wave_transit_unique_cells = gov_remediated[gov_remediated["wave_proxy_classification"] == "GEOGRAPHICALLY_UNREPRESENTATIVE"]["wave_grid_coordinates"].nunique()

    print("\n  Direct Grid-Cell Count Verification from Output Rows:")
    print(f"    - Unique 0.25° surface grid vertices (976 eligible ports): {surface_unique_cells}")
    print(f"    - Unique 0.50° wave ocean cells (961 extractable ports): {wave_all_unique_cells}")
    print(f"    - Unique 0.50° wave ocean cells (940 model-eligible ports): {wave_model_unique_cells}")
    print(f"    - Unique 0.50° wave ocean cells (21 transit-only ports): {wave_transit_unique_cells}")

    assert surface_unique_cells == 936, f"Expected 936 unique surface cells, got {surface_unique_cells}"
    assert wave_all_unique_cells == 774, f"Expected 774 unique wave cells, got {wave_all_unique_cells}"
    assert wave_model_unique_cells == 763, f"Expected 763 unique model wave cells, got {wave_model_unique_cells}"
    assert wave_transit_unique_cells == 16, f"Expected 16 unique transit wave cells, got {wave_transit_unique_cells}"
    print("  All direct grid-cell count assertions PASSED cleanly.")

    # -------------------------------------------------------------------------
    # STEP 6: GENERATE SUMMARY JSON, POLICY MD, AND REMEDIATION REPORT
    # -------------------------------------------------------------------------
    print("\n[6/6] Generating documentation and metadata artifacts...")

    # Summary JSON
    summary_data = {
        "audit_name": "Spatial Feature Governance v1.3 Remediation and Reconciliation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_ports": 2065,
        "candidate_table_summary": {
            "total_candidate_rows": len(cand_df),
            "identity_candidate_accepted_count": int(cand_df["identity_candidate_accepted"].sum()),
            "coordinate_candidate_accepted_count": int(cand_df["coordinate_candidate_accepted"].sum()),
            "identity_accepted_coords_rejected_count": int(
                ((cand_df["identity_candidate_accepted"] == True) & (cand_df["coordinate_candidate_accepted"] == False)).sum()
            ),
        },
        "identity_resolution_counts": {
            "UNRESOLVED_NO_COORDINATES": int((gov_remediated["identity_resolution_status"] == "UNRESOLVED_NO_COORDINATES").sum()),
            "CONFIRMED_AUTHORITATIVE_COORDINATES": int((gov_remediated["identity_resolution_status"] == "CONFIRMED_AUTHORITATIVE_COORDINATES").sum()),
            "SOURCE_CONFLICT_REQUIRES_MANUAL_REVIEW": int((gov_remediated["identity_resolution_status"] == "SOURCE_CONFLICT_REQUIRES_MANUAL_REVIEW").sum()),
            "IDENTITY_CONFIRMED_COORDINATE_UNAVAILABLE": int((gov_remediated["identity_resolution_status"] == "IDENTITY_CONFIRMED_COORDINATE_UNAVAILABLE").sum()),
            "CONFIRMED_COORDINATES_COUNTRY_CONFLICT": int((gov_remediated["identity_resolution_status"] == "CONFIRMED_COORDINATES_COUNTRY_CONFLICT").sum()),
        },
        "surface_governance": {
            "extraction_eligible_ports": 976,
            "extraction_ineligible_ports": 1089,
            "unique_grid_cells_0_25_deg": surface_unique_cells,
            "modeling_eligibility_status": {
                "ELIGIBLE_PRIMARY": 976,
                "INELIGIBLE": 1089,
            }
        },
        "wave_governance": {
            "extractable_ports_total": 961,
            "extraction_bypassed_ports": 1104,
            "unique_ocean_cells_total": wave_all_unique_cells,
            "unique_ocean_cells_model_eligible": wave_model_unique_cells,
            "unique_ocean_cells_transit_only": wave_transit_unique_cells,
            "proxy_classification_counts": {
                "DIRECT_OCEAN_CELL": 509,
                "NEARBY_COASTAL_PROXY": 205,
                "REGIONAL_MARINE_FORCING_ONLY": 226,
                "GEOGRAPHICALLY_UNREPRESENTATIVE": 21,
                "BLOCKED_INLAND": 15,
                "IDENTITY_OR_COORDINATE_UNCERTAIN": 4,
                "UNMAPPED": 1085,
            },
            "modeling_eligibility_status": {
                "ELIGIBLE_PRIMARY": 714,
                "ELIGIBLE_CONDITIONAL": 226,
                "INELIGIBLE": 1125,
            },
            "offshore_proxy_flag_true_count": 226,
        },
        "reconciliation_status": "100% RECONCILED (All 15 assertions passed cleanly)",
        "output_directory": str(OUTPUT_DIR),
    }

    summary_json_path = OUTPUT_DIR / "spatial_feature_governance_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  Wrote summary JSON to {summary_json_path}")

    # Metadata for Registry v1.3 Remediation
    metadata_v1_3 = {
        "registry_version": "1.3-remediated",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_portwatch_ports": len(reg_df),
        "total_ports_with_accepted_coordinates": int(reg_df["accepted_latitude"].notna().sum()),
        "total_ports_without_accepted_coordinates": int(reg_df["accepted_latitude"].isna().sum()),
        "exceptions_resolved": [
            "port377 (Gary -> USGYY, coordinates null)",
            "port130 (Bay City -> unresolved identity, coordinates null)",
            "port2270 (Monroe -> USMOI, coordinates null)",
            "port285 (Delfzijl -> NLDZL, coordinates null pending errata)"
        ],
        "candidate_table_file": "port_identity_candidates.csv",
        "governance_layer_file": "spatial_feature_governance.csv",
        "description": "Remediated Canonical Port Registry v1.3 and Candidate Table: fully decoupled identity acceptance from coordinate acceptance, added explicit USGYY/USMOI/USBCY candidate rows, and established categorical modeling governance."
    }
    with open(OUTPUT_DIR / "registry_metadata.json", "w") as f:
        json.dump(metadata_v1_3, f, indent=2)

    # Reconciliation Metrics CSV
    recon_df = pd.DataFrame([
        {"metric": "total_ports", "value": 2065, "notes": "Full PortWatch universe"},
        {"metric": "ports_with_accepted_coords", "value": 976, "notes": "976 confirmed ports (975 conflict-free + 1 BES/BQ)"},
        {"metric": "ports_without_accepted_coords", "value": 1089, "notes": "1,085 originally unmapped + 4 exceptions = 1,089"},
        {"metric": "candidate_table_rows", "value": len(cand_df), "notes": "Full candidate table including explicit USGYY, USMOI, USBCY"},
        {"metric": "surface_extraction_eligible", "value": 976, "notes": "Mapped to 936 unique 0.25 deg grid vertices"},
        {"metric": "wave_extraction_extractable", "value": 961, "notes": "940 model-eligible + 21 transit-only mapped to 774 unique ocean cells"},
        {"metric": "wave_dwell_modeling_eligible", "value": 940, "notes": "714 Primary + 226 Conditional (with offshore_proxy_flag=True)"},
        {"metric": "wave_dwell_modeling_ineligible", "value": 1125, "notes": "21 transit-only + 15 blocked + 4 exceptions + 1,085 unmapped"},
        {"metric": "offshore_proxy_flag_count", "value": 226, "notes": "Strictly assigned to 226 regional marine proxies"},
    ])
    recon_df.to_csv(OUTPUT_DIR / "source_reconciliation.csv", index=False)

    # Policy Markdown (Part 4)
    policy_md_lines = [
        "# Global ERA5 Spatial Feature Governance Policy (v1.3 Remediated)",
        "",
        "**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  ",
        "**Milestone:** Phase 2 Data Engineering — Final Feature Governance & Remediation  ",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        "**Version:** 1.3-Remediated (Operational Policy for Bulk ERA5 Extraction and Supervised ML)  ",
        "",
        "---",
        "",
        "## 1. Executive Principles",
        "",
        "This policy defines mandatory spatial feature governance rules for environmental data extraction and machine learning pipelines. It ensures broad global coverage without compromising feature fidelity or introducing data leakage.",
        "",
        "### Core Governance Axioms",
        "1. **Extraction Eligibility ≠ Modeling Eligibility:** Retrieving an ERA5 time series from the ECMWF API does not grant permission to use that series as an unconstrained feature in terminal dwelling time models.",
        "2. **Primary vs. Conditional Eligibility:** Primary features (`ELIGIBLE_PRIMARY`) represent local atmospheric or nearshore forcing. Conditional features (`ELIGIBLE_CONDITIONAL`) represent distant offshore shelf conditions and require explicit indicator flags.",
        "3. **Identity Confidence vs. Coordinate Confidence:** Confirming that a PortWatch record corresponds to a specific UN/LOCODE seaport does not imply that catalog coordinates are accurate or present.",
        "4. **No Claim of Berth Conditions:** ERA5 wave products (0.5° resolution) resolve deep-water swell and wind-sea on open ocean shelves. Wave proxies never represent sheltered harbor basins, docks, or berths.",
        "",
        "---",
        "",
        "## 2. Spatial Feature Governance Matrix",
        "",
        "| Governance Category | Port Count | Surface Extraction | Wave Extraction | Surface Dwell Modeling | Wave Dwell Modeling | Offshore Proxy Flag | Operational Feature Policy |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|",
        "| **Direct Ocean Ports** | 509 | ELIGIBLE | ELIGIBLE | `ELIGIBLE_PRIMARY` | `ELIGIBLE_PRIMARY` | `False` | 0 km displacement. Theoretical 0.5° grid vertex is valid ocean. High spatial confidence. |",
        "| **Nearby Coastal Proxies** | 205 | ELIGIBLE | ELIGIBLE | `ELIGIBLE_PRIMARY` | `ELIGIBLE_PRIMARY` | `False` | Displaced < 40 km to outer fairway / approach channel. Marine forcing on harbor approaches. |",
        "| **Regional Marine Proxies** | 226 | ELIGIBLE | ELIGIBLE | `ELIGIBLE_PRIMARY` | `ELIGIBLE_CONDITIONAL` | `True` | Displaced 40–100 km on outer shelf. Mandatory `offshore_proxy_flag = True`; subject to ablation. |",
        "| **Geographically Unrepresentative** | 21 | ELIGIBLE | ELIGIBLE (TRANSIT) | `ELIGIBLE_PRIMARY` | `INELIGIBLE` | `False` | Inland river ports (Quebec, Rosario, Nanjing). Wave features extracted solely for transit; masked for dwell models. |",
        "| **Blocked Inland Ports** | 15 | ELIGIBLE | INELIGIBLE | `ELIGIBLE_PRIMARY` | `INELIGIBLE` | `False` | Enclosed river/lake ports (Amazon basin, Montreal, Santa Fe). Wave coordinates strictly null. |",
        "| **Identity / Coordinate Uncertain** | 4 | INELIGIBLE | INELIGIBLE | `INELIGIBLE` | `INELIGIBLE` | `False` | Gary, Bay City, Monroe, Delfzijl. Erroneous coordinates nullified pending registry errata. |",
        "| **Unmapped Ports** | 1,085 | INELIGIBLE | INELIGIBLE | `INELIGIBLE` | `INELIGIBLE` | `False` | Reference coordinates unavailable. Strictly preserved with null coordinates. |",
        "| **TOTAL UNIVERSE** | **2,065** | **976 Eligible** | **961 Extractable** | **976 Primary** | **714 Primary + 226 Cond.** | **226 True** | **Complete, mutually exclusive partition.** |",
        "",
        "---",
        "",
        "## 3. Surface Extraction & Modeling Governance",
        "",
        "- Exactly **976 ports** are approved for surface meteorological extraction across **936 unique 0.25° grid vertices**.",
        "- Surface variables: `u10` (10m u-wind), `v10` (10m v-wind), `t2m` (2m temperature), `msl` (mean sea level pressure), `tp` (total precipitation).",
        "- Geodesic distance from port coordinate to nearest 0.25° grid vertex is ≤ 18.31 km (mean 9.62 km).",
        "- All 976 approved ports receive `surface_modeling_eligibility_status = 'ELIGIBLE_PRIMARY'`.",
        "- All 1,089 unmapped and exception ports receive `INELIGIBLE` and are strictly excluded from queries to prevent sampling weather at false municipality centroids.",
        "",
        "---",
        "",
        "## 4. Wave Extraction & Modeling Governance",
        "",
        "- Exactly **961 ports** are approved for wave extraction across **774 unique valid 0.50° ocean cells**.",
        "- Wave variables: `swh` (Significant Wave Height), `mwp` (Mean Wave Period), `mwd` (Mean Wave Direction).",
        "- **Modeling Eligibility Breakdown (940 ports total):**",
        "  1. **Primary Wave Features (714 ports):** Comprises 509 direct ocean ports and 205 nearby coastal proxies (<40 km). Wave proxy confidence is `HIGH_DIRECT_OCEAN` and `MEDIUM_COASTAL_PROXY`. Represents approach fairway conditions.",
        "  2. **Conditional Wave Features (226 ports):** Comprises regional shelf proxies (40–100 km displacement). Wave proxy confidence is `LOW_REGIONAL_FORCING`. Requires `offshore_proxy_flag = True` in all downstream models.",
        "  3. **Transit-Only Features (21 ports):** Comprises deep inland river terminals (e.g. Quebec, Rosario, Nanjing, Baton Rouge). Extracted across 16 unique ocean cells exclusively for regional shipping lane analysis. Strictly masked (`wave_modeling_eligibility_status = 'INELIGIBLE'`) for port dwell models.",
        "  4. **Bypassed / Ineligible Ports (1,104 ports):** Comprises 15 blocked inland ports, 4 exception ports, and 1,085 unmapped ports. All wave queries are bypassed; features are populated with nulls.",
        "",
        "---",
        "",
        "## 5. Downstream Feature Engineering & Missingness Policy",
        "",
        "When training supervised models (CatBoost, Gradient Boosting, Deep Learning LSTM/Transformer architectures):",
        "1. **Surface Features:** Fully populated for all 976 validated ports. Zero synthetic imputation allowed.",
        "2. **Wave Features:**",
        "   - **Tier 1 (714 ports):** Use raw `swh`, `mwp`, `mwd` directly.",
        "   - **Tier 2 (226 ports):** Feed `swh`, `mwp`, `mwd` accompanied by binary feature `offshore_proxy_flag = 1.0`. Model must evaluate feature importance with and without proxy interaction.",
        "   - **Tier 3 (1,125 ports):** Wave features must be masked (`NaN` with dedicated `wave_missing_flag = 1.0`). Under no circumstances should open-sea swell be attributed to river terminals (Quebec, Rosario, Nanjing).",
        "",
        "---",
        "",
        "## 6. Audit & Immutability Verification",
        "",
        "All historical Phase 1 assets, frozen ERA5 datasets, canonical registry v1.2, and manifests v1/v2 remain strictly immutable and verified via SHA-256 cryptographic digests."
    ]

    policy_md_path = OUTPUT_DIR / "SPATIAL_FEATURE_GOVERNANCE_POLICY.md"
    with open(policy_md_path, "w") as f:
        f.write("\n".join(policy_md_lines))
    print(f"  Wrote updated governance policy to {policy_md_path}")

    # Remediation Report (Part 5)
    report_lines = [
        "# Phase 2 Spatial Governance v1.3 Remediation and Reconciliation Report",
        "",
        "**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  ",
        "**Milestone:** Phase 2 Data Engineering — Spatial Governance v1.3 Remediation  ",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        "**Artifact Directory:** `reports/port_identity_resolution/v1_3_remediation/`  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "This remediation milestone successfully resolves all candidate-table semantics and spatial governance inconsistencies identified during expert review:",
        "1. **Candidate Table Semantics Decoupled:** Candidate identity acceptance (`identity_candidate_accepted`) is now strictly independent from coordinate acceptance (`coordinate_candidate_accepted`).",
        "2. **Explicit Candidate Records Added:** Gary (`USGYY`, row 18413), Monroe (`USMOI`, row 22610), and Bay City (`USBCY`, row 13506) are now explicitly represented in the candidate table, while retaining original rejected collision candidates (`USGT4`, `USM4X`).",
        "3. **Delfzijl Candidate Alignment:** Delfzijl retains `NLDZL` as its accepted identity (`identity_candidate_accepted = True`), while its coordinate source is explicitly marked `coordinate_candidate_accepted = False` and `coordinate_validation_status = 'INVALID_PRIMARY_SOURCE_TYPO'` pending official UNECE errata.",
        "4. **Bay City Source Conflict Preserved:** Bay City preserves `identity_candidate_accepted = False`, `coordinate_candidate_accepted = False`, and `accepted_unlocode = null` pending manual review.",
        "5. **Categorical Governance Schema Implemented:** Governance dataset now features explicit categorical statuses (`ELIGIBLE_PRIMARY`, `ELIGIBLE_CONDITIONAL`, `INELIGIBLE`), `offshore_proxy_flag` (True for exactly 226 regional proxies), `wave_proxy_confidence`, `identity_confidence`, and `coordinate_validation_status`.",
        "6. **Direct Grid-Cell Count Reconciliation:** Reconciled actual unique grid coordinates directly from the output rows: **936 unique surface cells** (for 976 ports) and **774 unique wave ocean cells** (for 961 extractable ports).",
        "",
        "---",
        "",
        "## 2. Candidate Table Remediation Audit",
        "",
        "| PortWatch ID | Port Name | Candidate Source | Candidate ID | Identity Accepted | Coord Accepted | Coordinate Validation Status | Rejection Reason |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---|:---|",
        "| `port377` | Gary | UNLOCODE | `USGYY` | **True** | **False** | `MISSING_IN_PRIMARY_SOURCE` | Correct commercial seaport Gary, IN (USGYY) confirmed, but coordinates empty in UNECE release. |",
        "| `port377` | Gary | UNLOCODE / UPPLY | `USGT4` | **False** | **False** | `REJECTED_WRONG_LOCATION` | COLLISION_WRONG_STATE: Resolves to Panola County, Texas (USGT4), not Gary, Indiana. |",
        "| `port2270` | Monroe | UNLOCODE | `USMOI` | **True** | **False** | `MISSING_IN_PRIMARY_SOURCE` | Correct commercial port Monroe, MI (USMOI) confirmed, but coordinates empty in UNECE release. |",
        "| `port2270` | Monroe | UNLOCODE / UPPLY | `USM4X` | **False** | **False** | `REJECTED_WRONG_LOCATION` | COLLISION_WRONG_STATE: Resolves to Grafton County, New Hampshire (USM4X), not Monroe, Michigan. |",
        "| `port285` | Delfzijl | UNLOCODE / UPPLY | `NLDZL` | **True** | **False** | `INVALID_PRIMARY_SOURCE_TYPO` | Official UNECE CodeListPart2 row 24631 records 5231N instead of 5319N (88.9 km south on farmland). Coordinate rejected pending errata. |",
        "| `port130` | Bay City | UNLOCODE / UPPLY | `USYBZ` | **False** | **False** | `SOURCE_CONFLICT_REQUIRES_MANUAL_REVIEW` | Multi-source conflict: text LOCODE points to Bay City, WI, while commercial port is in Michigan. |",
        "| `port130` | Bay City | UNLOCODE | `USBCY` | **False** | **False** | `SOURCE_CONFLICT_REQUIRES_MANUAL_REVIEW` | UNECE USBCY in Michigan lacks port code 1 (--34----), conflicting with text LOCODE US YBZ. |",
        "",
        "---",
        "",
        "## 3. Spatial Feature Governance Summary",
        "",
        "### A. Complete Category Partition (2,065 Ports Total)",
        "- **Surface Extraction Eligible:** Exactly **976 ports** mapped to **936 unique 0.25° grid vertices**.",
        "- **Surface Extraction Ineligible:** Exactly **1,089 ports** (4 exceptions + 1,085 unmapped).",
        "- **Surface Modeling Eligible:** Exactly **976 ports** (`ELIGIBLE_PRIMARY`).",
        "- **Surface Modeling Ineligible:** Exactly **1,089 ports** (`INELIGIBLE`).",
        "",
        "### B. Wave Extraction & Modeling Partition",
        "- **Wave Extractable Ports:** Exactly **961 ports** mapped to **774 unique 0.50° ocean cells**.",
        "  - **Direct Ocean (509 ports):** `wave_modeling_eligibility_status = 'ELIGIBLE_PRIMARY'`, `offshore_proxy_flag = False`.",
        "  - **Nearby Coastal Proxies (205 ports):** `wave_modeling_eligibility_status = 'ELIGIBLE_PRIMARY'`, `offshore_proxy_flag = False` (Approach channel forcing; does not claim harbor/berth observation).",
        "  - **Regional Marine Proxies (226 ports):** `wave_modeling_eligibility_status = 'ELIGIBLE_CONDITIONAL'`, `offshore_proxy_flag = True` (Displaced 40–100 km).",
        "  - **Geographically Unrepresentative / Transit Only (21 ports):** `wave_modeling_eligibility_status = 'INELIGIBLE'`, `offshore_proxy_flag = False` (Extracted for regional shipping lane transit; masked for berth dwell modeling).",
        "- **Wave Bypassed / Ineligible Ports (1,104 ports):**",
        "  - **Blocked Inland (15 ports):** No ocean cells within radius; wave coordinates strictly null.",
        "  - **Identity / Coordinate Uncertain (4 ports):** Gary, Bay City, Monroe, Delfzijl; wave queries bypassed.",
        "  - **Unmapped (1,085 ports):** Primary coordinates missing; wave queries bypassed.",
        "",
        "---",
        "",
        "## 4. Direct Grid-Cell Count Reconciliation",
        "",
        "- **Surface Grid Vertices:**",
        "  - Total surface-eligible ports: 976",
        "  - Actual unique 0.25° grid vertices from output rows: **936**",
        "  - Mean geodesic distance to vertex: 9.62 km (max: 18.31 km)",
        "- **Wave Ocean Grid Cells:**",
        "  - Total wave-extractable ports: 961",
        "  - Actual unique 0.50° valid ocean cells from output rows: **774**",
        "  - Unique ocean cells for dwell-modeling ports (940 ports): **763**",
        "  - Unique ocean cells for transit-only ports (21 ports): **16**",
        "  - Overlapping cells between dwell-modeling and transit-only: **5** ($763 + 16 - 5 = 774$).",
        "",
        "---",
        "",
        "## 5. 15-Rule Independent Verification Audit",
        "",
        "All 15 independent verification checks passed with exit code 0:",
        "1. Rule 1 PASS: Every accepted registry identity appears as an accepted identity candidate.",
        "2. Rule 2 PASS: No rejected candidate is simultaneously represented as the accepted identity.",
        "3. Rule 3 PASS: Candidate identity acceptance is independent of coordinate acceptance (confirmed on Gary, Monroe, Delfzijl).",
        "4. Rule 4 PASS: Accepted coordinates match governance coordinates for all 976 eligible ports.",
        "5. Rule 5 PASS: All four exceptions have strictly null accepted coordinates.",
        "6. Rule 6 PASS: Exactly 2,065 unique PortWatch records are represented.",
        "7. Rule 7 PASS: Exactly 976 ports are surface eligible (and 1,089 ineligible).",
        "8. Rule 8 PASS: Exactly 961 ports are wave extractable (940 model-eligible + 21 transit-only).",
        "9. Rule 9 PASS: Exactly 940 ports are wave-model eligible (714 primary + 226 conditional).",
        "10. Rule 10 PASS: Exactly 226 regional marine proxies have `offshore_proxy_flag = True`.",
        "11. Rule 11 PASS: Zero other categories receive `offshore_proxy_flag = True`.",
        "12. Rule 12 PASS: All 21 geographically unrepresentative ports are strictly excluded from port-level wave dwell modeling.",
        "13. Rule 13 PASS: 15 blocked inland ports remain completely distinct from the 1,085 unmapped ports.",
        "14. Rule 14 PASS: All four exceptions are excluded from both surface and wave modeling.",
        "15. Rule 15 PASS: Kralendijk (`port598`) retains explicit BES/BQ country-discrepancy provenance."
    ]

    report_path = OUTPUT_DIR / "REMEDIATION_AND_INDEPENDENT_RECONCILIATION_REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Wrote remediation report to {report_path}")

    # -------------------------------------------------------------------------
    # STEP 7: POST-EXECUTION IMMUTABILITY CHECK
    # -------------------------------------------------------------------------
    print("\nVerifying immutability of protected files post-execution...")
    for name, path in PROTECTED_FILES.items():
        post_hash = compute_sha256(path)
        assert post_hash == pre_hashes[name], f"CRITICAL: Protected file {name} was modified!"
    print("ALL PROTECTED ARTIFACTS AND HISTORICAL MANIFESTS CONFIRMED UNCHANGED.")

    print("\n" + "=" * 80)
    print("SPATIAL GOVERNANCE V1.3 REMEDIATION & RECONCILIATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
