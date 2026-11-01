"""Build and Validate Global ERA5 Extraction Manifest for PortWatch Universe.

Phase 2: AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks
Milestone: Global ERA5 Spatial Extraction Manifest (Surface 0.25° & Wave 0.50°)

Strict Safety & Reproducibility Constraints:
- Treats existing generated artifacts and canonical registry v1.2 as immutable.
- Refuses to overwrite existing versioned outputs.
- Does not download ERA5 global datasets or trigger bulk extraction.
- Preserves full source provenance and original identity fields.
- Never fabricates coordinates for missing or unmatched ports.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Geod

# =====================================================================
# CONFIGURATION & IMMUTABLE PATHS
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]

# Input Canonical Registry (v1.2)
REGISTRY_DIR = ROOT / "reports" / "port_registry" / "v1_2_identifier_bridge"
CANONICAL_REGISTRY_CSV = REGISTRY_DIR / "canonical_port_registry.csv"
REGISTRY_METADATA_JSON = REGISTRY_DIR / "registry_metadata.json"

# Frozen ERA5 Artifacts for Immutability Verification
FROZEN_ERA5_HOURLY = ROOT / "data" / "interim" / "era5" / "era5_port_hourly.parquet"
FROZEN_ERA5_HOURLY_META = ROOT / "data" / "interim" / "era5" / "era5_port_hourly_metadata.json"
FROZEN_ERA5_DAILY_V2 = ROOT / "data" / "processed" / "era5" / "era5_port_daily_v2.parquet"
FROZEN_ERA5_DAILY_V2_META = ROOT / "data" / "processed" / "era5" / "era5_port_daily_v2_metadata.json"

# Output Directory (New Versioned Directory)
OUTPUT_DIR = ROOT / "data" / "interim" / "era5" / "global_manifest_v1"

MANIFEST_PARQUET = OUTPUT_DIR / "global_era5_spatial_manifest.parquet"
MANIFEST_CSV = OUTPUT_DIR / "global_era5_spatial_manifest.csv"
AUDIT_JSON = OUTPUT_DIR / "spatial_coverage_audit.json"
COUNTRY_COVERAGE_CSV = OUTPUT_DIR / "spatial_coverage_by_country.csv"
DISTANCE_REVIEW_CSV = OUTPUT_DIR / "spatial_distance_review.csv"
MANIFEST_METADATA_JSON = OUTPUT_DIR / "manifest_metadata.json"

OUTPUT_FILES = [
    MANIFEST_PARQUET,
    MANIFEST_CSV,
    AUDIT_JSON,
    COUNTRY_COVERAGE_CSV,
    DISTANCE_REVIEW_CSV,
    MANIFEST_METADATA_JSON,
]

# ERA5 ARCO Grid Specifications
SURFACE_RES = 0.25
WAVE_RES = 0.50

# Thresholds for Review
SURFACE_LARGE_DIST_KM = 15.0
WAVE_LARGE_DIST_KM = 30.0


# =====================================================================
# GEODETIC & GRID UTILITIES
# =====================================================================

def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found for hashing: {filepath}")
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def normalize_longitude_arco(lon: float) -> float:
    """Normalize longitude to ECMWF ARCO ERA5 convention [0, 360).

    Handles negative longitudes (west of prime meridian) and antimeridian.
    """
    norm = lon % 360.0
    if norm < 0:
        norm += 360.0
    return norm


def find_nearest_grid_vertex(lat: float, lon: float, resolution: float) -> tuple[float, float]:
    """Find nearest ERA5 ARCO grid vertex for a valid (lat, lon).

    Surface (0.25°):
      Latitude:  90.0 to -90.0 in 0.25° steps
      Longitude: 0.0 to 359.75 in 0.25° steps

    Wave (0.50°):
      Latitude:  90.0 to -90.0 in 0.50° steps
      Longitude: 0.0 to 359.5 in 0.50° steps
    """
    # Round latitude to nearest multiple of resolution
    nearest_lat = round(lat / resolution) * resolution
    # Clamp to valid [-90.0, 90.0]
    nearest_lat = max(-90.0, min(90.0, nearest_lat))
    decimals = 2 if resolution == 0.25 else 1
    nearest_lat = round(nearest_lat, decimals)

    # Normalize longitude to [0, 360)
    norm_lon = normalize_longitude_arco(lon)
    nearest_lon = round(norm_lon / resolution) * resolution
    # Wrap 360.0 back to 0.0
    if nearest_lon >= 360.0:
        nearest_lon = 0.0
    nearest_lon = round(nearest_lon, decimals)

    return nearest_lat, nearest_lon


class GeodesicCalculator:
    """High-precision WGS-84 geodesic distance calculator.

    Uses pyproj.Geod with exact Vincenty WGS-84 inverse ellipsoid fallback.
    """

    def __init__(self):
        try:
            self._geod = Geod(ellps="WGS84")
            self._backend = "pyproj.Geod(ellps='WGS84')"
        except Exception:
            self._geod = None
            self._backend = "Vincenty_WGS84_ellipsoid_fallback"

    @property
    def backend(self) -> str:
        return self._backend

    def calculate_distance_km(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Calculate geodesic distance in kilometers between two points."""
        if self._geod is not None:
            _, _, dist_m = self._geod.inv(lon1, lat1, lon2, lat2)
            return abs(dist_m) / 1000.0

        return self._vincenty_inverse(lat1, lon1, lat2, lon2)

    @staticmethod
    def _vincenty_inverse(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Vincenty inverse solution for WGS-84 ellipsoid geodesic distance."""
        a = 6378.1370  # semi-major axis (km)
        f = 1 / 298.257223563  # flattening
        b = (1 - f) * a

        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        U1 = math.atan((1 - f) * math.tan(phi1))
        U2 = math.atan((1 - f) * math.tan(phi2))
        L = math.radians(lon2 - lon1)

        sin_U1, cos_U1 = math.sin(U1), math.cos(U1)
        sin_U2, cos_U2 = math.sin(U2), math.cos(U2)

        lam = L
        for _ in range(100):
            sin_lam = math.sin(lam)
            cos_lam = math.cos(lam)
            sin_sigma = math.sqrt(
                (cos_U2 * sin_lam) ** 2
                + (cos_U1 * sin_U2 - sin_U1 * cos_U2 * cos_lam) ** 2
            )
            if sin_sigma == 0:
                return 0.0  # coincident
            cos_sigma = sin_U1 * sin_U2 + cos_U1 * cos_U2 * cos_lam
            sigma = math.atan2(sin_sigma, cos_sigma)
            sin_alpha = cos_U1 * cos_U2 * sin_lam / sin_sigma
            cos2_alpha = 1 - sin_alpha**2
            cos_2sigma_m = (
                cos_sigma - 2 * sin_U1 * sin_U2 / cos2_alpha
                if cos2_alpha != 0
                else 0.0
            )
            C = f / 16 * cos2_alpha * (4 + f * (4 - 3 * cos2_alpha))
            lam_prev = lam
            lam = L + (1 - C) * f * sin_alpha * (
                sigma
                + C * sin_sigma * (
                    cos_2sigma_m
                    + C * cos_sigma * (-1 + 2 * cos_2sigma_m**2)
                )
            )
            if abs(lam - lam_prev) < 1e-12:
                break
        else:
            # Fallback to great-circle Haversine
            dlat = phi2 - phi1
            dlon = math.radians(lon2 - lon1)
            h = math.sin(dlat / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlon / 2)**2
            return 2 * 6371.0088 * math.asin(math.sqrt(h))

        u2 = cos2_alpha * (a**2 - b**2) / (b**2)
        A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
        B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
        delta_sigma = B * sin_sigma * (
            cos_2sigma_m
            + B / 4 * (
                cos_sigma * (-1 + 2 * cos_2sigma_m**2)
                - B / 6 * cos_2sigma_m * (-3 + 4 * sin_sigma**2) * (-3 + 4 * cos_2sigma_m**2)
            )
        )
        return abs(b * A * (sigma - delta_sigma))


# =====================================================================
# SPATIAL MANIFEST BUILDER
# =====================================================================

def build_manifest(
    registry_df: pd.DataFrame,
    calculator: GeodesicCalculator,
) -> pd.DataFrame:
    """Build global spatial extraction manifest for all PortWatch ports."""
    manifest_rows = []

    for row in registry_df.to_dict("records"):
        port_id = row["portwatch_id"]
        port_name = row["portwatch_name"]
        ref_lat = row["upply_latitude"]
        ref_lon = row["upply_longitude"]
        country_conflict = bool(row["country_conflict"])

        # Determine reference coordinate validity
        is_missing = pd.isna(ref_lat) or pd.isna(ref_lon)
        if is_missing:
            coord_status = "MISSING"
            is_valid = False
        elif not (-90.0 <= float(ref_lat) <= 90.0 and -180.0 <= float(ref_lon) <= 180.0):
            coord_status = "INVALID"
            is_valid = False
        else:
            coord_status = "VALID"
            is_valid = True

        out_row = dict(row)  # preserve all original columns

        # Explicit reference coordinate fields
        out_row["reference_latitude"] = float(ref_lat) if is_valid else np.nan
        out_row["reference_longitude"] = float(ref_lon) if is_valid else np.nan
        out_row["reference_coordinate_source"] = "UPPLY" if is_valid else "NONE"
        out_row["coordinate_validity_status"] = coord_status

        if not is_valid:
            # Explicit failure status with null grid cells
            out_row["surface_grid_latitude"] = np.nan
            out_row["surface_grid_longitude"] = np.nan
            out_row["surface_distance_km"] = np.nan
            out_row["surface_selection_method"] = None
            out_row["surface_spatial_mapping_status"] = (
                "FAILED_MISSING_COORDINATES"
                if coord_status == "MISSING"
                else "FAILED_INVALID_COORDINATES"
            )

            out_row["wave_grid_latitude"] = np.nan
            out_row["wave_grid_longitude"] = np.nan
            out_row["wave_distance_km"] = np.nan
            out_row["wave_selection_method"] = None
            out_row["wave_spatial_mapping_status"] = (
                "FAILED_MISSING_COORDINATES"
                if coord_status == "MISSING"
                else "FAILED_INVALID_COORDINATES"
            )

            out_row["likely_offshore_proxy"] = False
            out_row["overall_spatial_mapping_status"] = (
                "FAILED_MISSING_COORDINATES"
                if coord_status == "MISSING"
                else "FAILED_INVALID_COORDINATES"
            )
            out_row["spatial_mapping_reason_code"] = (
                "MISSING_PORT_COORDINATES"
                if coord_status == "MISSING"
                else "INVALID_PORT_COORDINATES"
            )
        else:
            lat = float(ref_lat)
            lon = float(ref_lon)

            # Surface grid mapping (0.25°)
            s_lat, s_lon = find_nearest_grid_vertex(lat, lon, SURFACE_RES)
            s_dist = calculator.calculate_distance_km(lat, lon, s_lat, s_lon)

            out_row["surface_grid_latitude"] = s_lat
            out_row["surface_grid_longitude"] = s_lon
            out_row["surface_distance_km"] = round(s_dist, 4)
            out_row["surface_selection_method"] = "NEAREST_GRID_VERTEX_0_25"
            out_row["surface_spatial_mapping_status"] = "MAPPED_NEAREST_VERTEX"

            # Wave grid mapping (0.50°)
            w_lat, w_lon = find_nearest_grid_vertex(lat, lon, WAVE_RES)
            w_dist = calculator.calculate_distance_km(lat, lon, w_lat, w_lon)

            out_row["wave_grid_latitude"] = w_lat
            out_row["wave_grid_longitude"] = w_lon
            out_row["wave_distance_km"] = round(w_dist, 4)
            out_row["wave_selection_method"] = "NEAREST_GRID_VERTEX_0_50"
            out_row["wave_spatial_mapping_status"] = "MAPPED_NEAREST_VERTEX"

            # Flag potential offshore proxy requirements
            # (e.g. wave distance >= 30 km represents the upper 5% of theoretical grid distances)
            is_large_wave = w_dist >= WAVE_LARGE_DIST_KM
            out_row["likely_offshore_proxy"] = bool(is_large_wave)

            if country_conflict:
                out_row["overall_spatial_mapping_status"] = "WARNING_COUNTRY_CONFLICT"
                out_row["spatial_mapping_reason_code"] = (
                    "MAPPING_SUCCESS_WITH_COUNTRY_CONFLICT"
                )
            else:
                out_row["overall_spatial_mapping_status"] = "SUCCESS"
                out_row["spatial_mapping_reason_code"] = "MAPPING_SUCCESS"

        manifest_rows.append(out_row)

    manifest_df = pd.DataFrame(manifest_rows)
    return manifest_df


# =====================================================================
# SPATIAL QUALITY AUDIT & REVIEWS
# =====================================================================

def generate_spatial_quality_audit(
    manifest_df: pd.DataFrame,
) -> dict[str, Any]:
    """Generate comprehensive spatial quality audit as required by specification."""
    total_ports = len(manifest_df)
    valid_mask = manifest_df["coordinate_validity_status"] == "VALID"
    missing_mask = manifest_df["coordinate_validity_status"] == "MISSING"
    invalid_mask = manifest_df["coordinate_validity_status"] == "INVALID"

    valid_df = manifest_df[valid_mask].copy()

    s_mapped = (manifest_df["surface_spatial_mapping_status"] == "MAPPED_NEAREST_VERTEX").sum()
    w_mapped = (manifest_df["wave_spatial_mapping_status"] == "MAPPED_NEAREST_VERTEX").sum()

    # Unique cells
    unique_surface = valid_df[["surface_grid_latitude", "surface_grid_longitude"]].drop_duplicates()
    unique_wave = valid_df[["wave_grid_latitude", "wave_grid_longitude"]].drop_duplicates()

    # Ports sharing cells
    surf_counts = valid_df.groupby(["surface_grid_latitude", "surface_grid_longitude"]).size()
    shared_surf_cells = (surf_counts > 1).sum()
    ports_sharing_surf = surf_counts[surf_counts > 1].sum()

    wave_counts = valid_df.groupby(["wave_grid_latitude", "wave_grid_longitude"]).size()
    shared_wave_cells = (wave_counts > 1).sum()
    ports_sharing_wave = wave_counts[wave_counts > 1].sum()

    # Distance distributions
    def get_distribution(series: pd.Series) -> dict[str, float]:
        if len(series) == 0:
            return {}
        return {
            "min": round(float(series.min()), 4),
            "median": round(float(series.median()), 4),
            "mean": round(float(series.mean()), 4),
            "p90": round(float(series.quantile(0.90)), 4),
            "p95": round(float(series.quantile(0.95)), 4),
            "p99": round(float(series.quantile(0.99)), 4),
            "max": round(float(series.max()), 4),
        }

    s_dist_stats = get_distribution(valid_df["surface_distance_km"])
    w_dist_stats = get_distribution(valid_df["wave_distance_km"])

    # Large distance review
    large_surf_ports = valid_df[valid_df["surface_distance_km"] >= SURFACE_LARGE_DIST_KM]
    large_wave_ports = valid_df[valid_df["wave_distance_km"] >= WAVE_LARGE_DIST_KM]

    # Country coverage
    country_summary = manifest_df.groupby("portwatch_country").agg(
        total=("portwatch_id", "count"),
        mapped=("surface_distance_km", lambda s: s.notna().sum()),
    )
    total_countries = len(country_summary)
    countries_with_mapped = int((country_summary["mapped"] > 0).sum())
    countries_without_mapped = int((country_summary["mapped"] == 0).sum())

    audit = {
        "audit_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_portwatch_ports": total_ports,
            "ports_with_valid_coordinates": int(valid_mask.sum()),
            "ports_with_missing_coordinates": int(missing_mask.sum()),
            "ports_with_invalid_coordinates": int(invalid_mask.sum()),
            "ports_mapped_surface_grid": int(s_mapped),
            "ports_mapped_wave_grid": int(w_mapped),
            "unique_surface_grid_cells": len(unique_surface),
            "unique_wave_grid_cells": len(unique_wave),
            "shared_surface_cells_count": int(shared_surf_cells),
            "ports_sharing_surface_cell": int(ports_sharing_surf),
            "shared_wave_cells_count": int(shared_wave_cells),
            "ports_sharing_wave_cell": int(ports_sharing_wave),
            "ports_with_unusually_large_surface_distance": len(large_surf_ports),
            "ports_with_unusually_large_wave_distance": len(large_wave_ports),
            "ports_flagged_likely_offshore_proxy": int(valid_df["likely_offshore_proxy"].sum()),
            "countries_total": total_countries,
            "countries_with_mapped_ports": countries_with_mapped,
            "countries_without_mapped_ports": countries_without_mapped,
        },
        "distance_distributions_km": {
            "surface_grid_0_25": s_dist_stats,
            "wave_grid_0_50": w_dist_stats,
        },
        "thresholds_km": {
            "surface_large_distance_threshold": SURFACE_LARGE_DIST_KM,
            "wave_large_distance_threshold": WAVE_LARGE_DIST_KM,
        },
        "top_5_largest_surface_distance_ports": [
            {
                "portwatch_id": r["portwatch_id"],
                "portwatch_name": r["portwatch_name"],
                "country": r["portwatch_country"],
                "ref_lat": r["reference_latitude"],
                "ref_lon": r["reference_longitude"],
                "surface_grid": [r["surface_grid_latitude"], r["surface_grid_longitude"]],
                "surface_distance_km": r["surface_distance_km"],
            }
            for r in valid_df.sort_values("surface_distance_km", ascending=False).head(5).to_dict("records")
        ],
        "top_5_largest_wave_distance_ports": [
            {
                "portwatch_id": r["portwatch_id"],
                "portwatch_name": r["portwatch_name"],
                "country": r["portwatch_country"],
                "ref_lat": r["reference_latitude"],
                "ref_lon": r["reference_longitude"],
                "wave_grid": [r["wave_grid_latitude"], r["wave_grid_longitude"]],
                "wave_distance_km": r["wave_distance_km"],
                "likely_offshore_proxy": r["likely_offshore_proxy"],
            }
            for r in valid_df.sort_values("wave_distance_km", ascending=False).head(5).to_dict("records")
        ],
    }

    return audit


def generate_country_coverage(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Generate geographic coverage summary aggregated by country."""
    grouped = manifest_df.groupby(
        ["portwatch_country", "portwatch_iso3", "country_iso2"],
        dropna=False,
    )

    records = []
    for (country, iso3, iso2), group in grouped:
        valid_grp = group[group["coordinate_validity_status"] == "VALID"]
        mapped_count = len(valid_grp)
        unmapped_count = len(group) - mapped_count

        u_surf = valid_grp[["surface_grid_latitude", "surface_grid_longitude"]].drop_duplicates()
        u_wave = valid_grp[["wave_grid_latitude", "wave_grid_longitude"]].drop_duplicates()

        mean_s_dist = round(float(valid_grp["surface_distance_km"].mean()), 4) if mapped_count > 0 else np.nan
        mean_w_dist = round(float(valid_grp["wave_distance_km"].mean()), 4) if mapped_count > 0 else np.nan
        max_w_dist = round(float(valid_grp["wave_distance_km"].max()), 4) if mapped_count > 0 else np.nan

        records.append({
            "portwatch_country": country,
            "portwatch_iso3": iso3,
            "country_iso2": iso2,
            "total_ports": len(group),
            "mapped_ports": mapped_count,
            "unmapped_ports": unmapped_count,
            "unique_surface_cells": len(u_surf),
            "unique_wave_cells": len(u_wave),
            "mean_surface_distance_km": mean_s_dist,
            "mean_wave_distance_km": mean_w_dist,
            "max_wave_distance_km": max_w_dist,
        })

    country_df = pd.DataFrame(records).sort_values("total_ports", ascending=False).reset_index(drop=True)
    return country_df


def generate_distance_review(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Generate spatial distance review table for flagged ports."""
    valid_df = manifest_df[manifest_df["coordinate_validity_status"] == "VALID"].copy()

    # Identify cell sharing
    surf_counts = valid_df.groupby(["surface_grid_latitude", "surface_grid_longitude"])["portwatch_id"].transform("count")
    wave_counts = valid_df.groupby(["wave_grid_latitude", "wave_grid_longitude"])["portwatch_id"].transform("count")

    valid_df["shared_surface_cell"] = surf_counts > 1
    valid_df["shared_wave_cell"] = wave_counts > 1

    review_records = []
    for row in valid_df.to_dict("records"):
        flags = []
        notes = []

        if row["wave_distance_km"] >= WAVE_LARGE_DIST_KM:
            flags.append("LARGE_WAVE_DISTANCE")
            notes.append(
                f"Wave distance ({row['wave_distance_km']:.1f} km) is in top ~5% of grid distances; "
                "likely inland, estuary, or distant from nearest 0.5-deg ocean grid vertex."
            )

        if row["surface_distance_km"] >= SURFACE_LARGE_DIST_KM:
            flags.append("LARGE_SURFACE_DISTANCE")
            notes.append(
                f"Surface distance ({row['surface_distance_km']:.1f} km) is in top ~5% of grid distances; "
                "port lies near center of 0.25-deg cell."
            )

        if row["likely_offshore_proxy"]:
            flags.append("LIKELY_OFFSHORE_PROXY")
            notes.append(
                "Wave cell is an environmental proxy; ECMWF WAM model does not represent enclosed shallow water."
            )

        if row["shared_wave_cell"]:
            flags.append("SHARED_WAVE_CELL")
            notes.append("Shares 0.5-degree wave cell with adjacent port(s).")

        if row["country_conflict"]:
            flags.append("COUNTRY_CONFLICT_IN_REGISTRY")
            notes.append("PortWatch and candidate reference source country codes disagreed in registry v1.2.")

        if flags:
            review_records.append({
                "portwatch_id": row["portwatch_id"],
                "portwatch_name": row["portwatch_name"],
                "portwatch_country": row["portwatch_country"],
                "portwatch_iso3": row["portwatch_iso3"],
                "reference_latitude": row["reference_latitude"],
                "reference_longitude": row["reference_longitude"],
                "surface_grid_latitude": row["surface_grid_latitude"],
                "surface_grid_longitude": row["surface_grid_longitude"],
                "surface_distance_km": row["surface_distance_km"],
                "wave_grid_latitude": row["wave_grid_latitude"],
                "wave_grid_longitude": row["wave_grid_longitude"],
                "wave_distance_km": row["wave_distance_km"],
                "review_flags": " | ".join(flags),
                "review_notes": " ".join(notes),
            })

    review_df = pd.DataFrame(review_records)
    if not review_df.empty:
        review_df = review_df.sort_values("wave_distance_km", ascending=False).reset_index(drop=True)
    return review_df


# =====================================================================
# VALIDATION CHECKS
# =====================================================================

def run_automated_validations(
    manifest_df: pd.DataFrame,
    registry_df: pd.DataFrame,
    pre_hashes: dict[str, str],
) -> dict[str, Any]:
    """Execute automated validation checks required by specification."""
    checks = {}

    # 1. Exactly one manifest row per PortWatch ID
    checks["exactly_one_row_per_portwatch_id"] = (
        len(manifest_df) == manifest_df["portwatch_id"].nunique()
    )

    # 2. No duplicate PortWatch IDs
    checks["no_duplicate_portwatch_ids"] = not manifest_df["portwatch_id"].duplicated().any()

    # 3. All 2,065 registry ports represented
    checks["all_2065_ports_represented"] = (
        len(manifest_df) == 2065
        and set(manifest_df["portwatch_id"]) == set(registry_df["portwatch_id"])
    )

    # 4. All original identity fields preserved
    orig_cols = [
        "portwatch_id", "portwatch_name", "portwatch_country", "portwatch_iso3",
        "country_iso2", "portwatch_name_normalized", "unlocode", "unlocode_match_status",
        "unlocode_candidate_count", "unlocode_source_names", "unlocode_coordinate_strings",
        "upply_code", "upply_match_status", "upply_candidate_count", "upply_name",
        "upply_latitude", "upply_longitude", "upply_country_code", "country_conflict",
        "identity_status",
    ]
    preserved = True
    for col in orig_cols:
        if col not in manifest_df.columns:
            preserved = False
            break
        # Check value equivalence
        s1 = registry_df[col].astype(object).fillna("__NA__")
        s2 = manifest_df[col].astype(object).fillna("__NA__")
        if not (s1.astype(str) == s2.astype(str)).all():
            preserved = False
            break
    checks["all_original_identity_fields_preserved"] = preserved

    # 5. Valid coordinates remain within geographic bounds
    valid = manifest_df[manifest_df["coordinate_validity_status"] == "VALID"]
    checks["valid_coordinates_within_bounds"] = bool(
        valid["reference_latitude"].between(-90.0, 90.0).all()
        and valid["reference_longitude"].between(-180.0, 180.0).all()
    )

    # 6. Grid coordinates align with resolutions
    # Surface: 0.25 steps
    s_lat_align = np.isclose(valid["surface_grid_latitude"] % 0.25, 0.0) | np.isclose(valid["surface_grid_latitude"] % 0.25, 0.25)
    s_lon_align = np.isclose(valid["surface_grid_longitude"] % 0.25, 0.0) | np.isclose(valid["surface_grid_longitude"] % 0.25, 0.25)
    # Wave: 0.50 steps
    w_lat_align = np.isclose(valid["wave_grid_latitude"] % 0.50, 0.0) | np.isclose(valid["wave_grid_latitude"] % 0.50, 0.50)
    w_lon_align = np.isclose(valid["wave_grid_longitude"] % 0.50, 0.0) | np.isclose(valid["wave_grid_longitude"] % 0.50, 0.50)

    checks["surface_grid_coordinates_aligned_0_25"] = bool(s_lat_align.all() and s_lon_align.all())
    checks["wave_grid_coordinates_aligned_0_50"] = bool(w_lat_align.all() and w_lon_align.all())

    # 7. Longitude normalization convention [0, 360)
    checks["surface_longitudes_normalized_0_to_359_75"] = bool(
        valid["surface_grid_longitude"].between(0.0, 359.75).all()
    )
    checks["wave_longitudes_normalized_0_to_359_50"] = bool(
        valid["wave_grid_longitude"].between(0.0, 359.50).all()
    )

    # 8. Geodesic distances are finite and nonnegative wherever mappings exist
    checks["surface_distances_finite_and_nonnegative"] = bool(
        (valid["surface_distance_km"] >= 0.0).all()
        and np.isfinite(valid["surface_distance_km"]).all()
    )
    checks["wave_distances_finite_and_nonnegative"] = bool(
        (valid["wave_distance_km"] >= 0.0).all()
        and np.isfinite(valid["wave_distance_km"]).all()
    )

    # 9. Missing coordinates never receive invented grid cells
    missing = manifest_df[manifest_df["coordinate_validity_status"] != "VALID"]
    checks["missing_coordinates_have_null_grid_cells"] = bool(
        missing["surface_grid_latitude"].isna().all()
        and missing["surface_grid_longitude"].isna().all()
        and missing["surface_distance_km"].isna().all()
        and missing["wave_grid_latitude"].isna().all()
        and missing["wave_grid_longitude"].isna().all()
        and missing["wave_distance_km"].isna().all()
    )

    # 10. Existing frozen artifacts & registry remain unchanged
    post_hashes = {
        path_str: compute_sha256(Path(path_str))
        for path_str in pre_hashes
    }
    immutability_ok = True
    for p, pre_h in pre_hashes.items():
        if post_hashes[p] != pre_h:
            immutability_ok = False
            break
    checks["immutable_inputs_unchanged"] = immutability_ok

    all_passed = all(checks.values())
    return {
        "all_checks_passed": all_passed,
        "checks": checks,
    }


# =====================================================================
# MAIN EXECUTION ROUTINE
# =====================================================================

def main():
    print("=" * 70)
    print("BUILDING GLOBAL ERA5 SPATIAL EXTRACTION MANIFEST — PHASE 2")
    print("=" * 70)

    # 1. Check immutability preconditions & compute pre-hashes
    immutable_targets = [
        CANONICAL_REGISTRY_CSV,
        REGISTRY_METADATA_JSON,
        FROZEN_ERA5_HOURLY,
        FROZEN_ERA5_HOURLY_META,
        FROZEN_ERA5_DAILY_V2,
        FROZEN_ERA5_DAILY_V2_META,
    ]

    print("\n[1/7] Recording pre-execution hashes for immutable artifacts...")
    pre_hashes = {}
    for p in immutable_targets:
        if p.exists():
            h = compute_sha256(p)
            pre_hashes[str(p)] = h
            print(f"  {p.name}: {h[:16]}...")
        else:
            print(f"  WARNING: Optional target {p} not found.")

    # 2. Refuse to overwrite existing outputs
    print("\n[2/7] Checking output directory safety constraints...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_outputs = [p for p in OUTPUT_FILES if p.exists()]
    if existing_outputs:
        raise FileExistsError(
            f"Refusing to overwrite existing outputs in {OUTPUT_DIR}:\n"
            + "\n".join(f"  - {p.name}" for p in existing_outputs)
            + "\nDelete, rename, or bump the version directory before rerunning."
        )

    # 3. Load canonical port registry v1.2
    print(f"\n[3/7] Loading canonical port registry from {CANONICAL_REGISTRY_CSV}...")
    registry_df = pd.read_csv(CANONICAL_REGISTRY_CSV)
    print(f"  Loaded {len(registry_df)} rows. Columns: {len(registry_df.columns)}")
    if len(registry_df) != 2065:
        raise ValueError(f"Expected exactly 2065 ports, found {len(registry_df)}")

    # 4. Build manifest with Geodesic calculations
    print("\n[4/7] Generating spatial mappings (Surface 0.25° & Wave 0.50°)...")
    calculator = GeodesicCalculator()
    print(f"  Geodesic backend: {calculator.backend}")

    manifest_df = build_manifest(registry_df, calculator)
    print(f"  Generated manifest rows: {len(manifest_df)}")

    # 5. Generate Audits, Country Coverage, and Distance Review
    print("\n[5/7] Compiling spatial quality audit and review tables...")
    audit_data = generate_spatial_quality_audit(manifest_df)
    country_coverage_df = generate_country_coverage(manifest_df)
    distance_review_df = generate_distance_review(manifest_df)

    # 6. Execute Automated Validation Checks
    print("\n[6/7] Running automated validation checks...")
    validation_results = run_automated_validations(manifest_df, registry_df, pre_hashes)
    for check_name, status in validation_results["checks"].items():
        print(f"  - {check_name}: {'PASS' if status else 'FAIL'}")

    if not validation_results["all_checks_passed"]:
        raise RuntimeError("Validation failed! Aborting artifact persistence.")

    # Compile metadata
    metadata = {
        "manifest_version": "1.0",
        "dataset_name": "Global PortWatch ERA5 Spatial Extraction Manifest",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_registry": {
            "path": str(CANONICAL_REGISTRY_CSV),
            "sha256": pre_hashes[str(CANONICAL_REGISTRY_CSV)],
            "row_count": len(registry_df),
            "version": "1.2",
        },
        "target_grids": {
            "surface_grid": {
                "dataset": "ERA5 Reanalysis Single Levels Surface",
                "resolution_deg": SURFACE_RES,
                "latitude_bounds": [90.0, -90.0],
                "longitude_bounds": [0.0, 359.75],
                "spacing_deg": SURFACE_RES,
                "convention": "ECMWF ARCO Zarr sfc/geoChunked",
                "selection_method": "NEAREST_GRID_VERTEX_0_25",
            },
            "wave_grid": {
                "dataset": "ERA5 Reanalysis Single Levels Wave",
                "resolution_deg": WAVE_RES,
                "latitude_bounds": [90.0, -90.0],
                "longitude_bounds": [0.0, 359.5],
                "spacing_deg": WAVE_RES,
                "convention": "ECMWF ARCO Zarr wav/geoChunked",
                "selection_method": "NEAREST_GRID_VERTEX_0_50",
            },
        },
        "distance_calculation": {
            "ellipsoid": "WGS-84",
            "backend": calculator.backend,
            "units": "kilometers",
        },
        "summary_statistics": audit_data["summary"],
        "distance_distributions_km": audit_data["distance_distributions_km"],
        "validation_results": validation_results,
        "limitations": [
            "1,116 PortWatch ports have no accepted reference coordinates in canonical registry v1.2; these retain null grid cells.",
            "Upply coordinates are port reference points, not terminal, berth, or gate coordinates.",
            "ERA5 wave model (WAM) is an ocean-only model; nearest theoretical 0.5-degree grid cells for inland waterway, estuarine, or shallow fjord ports may fall on land cells, necessitating offshore proxy cells during extraction.",
            "A nearest ocean-grid cell is an environmental proxy, not a terminal or operational measurement.",
            "Bulk ERA5 extraction is not executed as part of this manifest milestone.",
        ],
        "output_artifacts": [p.name for p in OUTPUT_FILES],
    }

    # 7. Write All Artifacts
    print(f"\n[7/7] Persisting versioned artifacts to {OUTPUT_DIR}...")
    manifest_df.to_parquet(MANIFEST_PARQUET, index=False)
    manifest_df.to_csv(MANIFEST_CSV, index=False)
    AUDIT_JSON.write_text(json.dumps(audit_data, indent=2), encoding="utf-8")
    country_coverage_df.to_csv(COUNTRY_COVERAGE_CSV, index=False)
    distance_review_df.to_csv(DISTANCE_REVIEW_CSV, index=False)
    MANIFEST_METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nArtifacts successfully written:")
    for p in OUTPUT_FILES:
        size_kb = round(p.stat().st_size / 1024, 1)
        print(f"  - {p.name} ({size_kb} KB)")

    print("\n" + "=" * 70)
    print("GLOBAL ERA5 EXTRACTION MANIFEST BUILD & VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
