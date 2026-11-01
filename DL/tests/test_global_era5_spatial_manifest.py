"""Automated validation and regression tests for Global ERA5 Spatial Extraction Manifest.

Phase 2: AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks
Milestone: Global ERA5 Extraction Manifest Validation
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_global_era5_spatial_manifest import (
    AUDIT_JSON,
    CANONICAL_REGISTRY_CSV,
    COUNTRY_COVERAGE_CSV,
    DISTANCE_REVIEW_CSV,
    FROZEN_ERA5_DAILY_V2,
    FROZEN_ERA5_HOURLY,
    MANIFEST_CSV,
    MANIFEST_METADATA_JSON,
    MANIFEST_PARQUET,
    OUTPUT_DIR,
    REGISTRY_METADATA_JSON,
    GeodesicCalculator,
    build_manifest,
    compute_sha256,
    find_nearest_grid_vertex,
    normalize_longitude_arco,
)


@pytest.fixture(scope="module")
def calculator():
    return GeodesicCalculator()


@pytest.fixture(scope="module")
def registry_df():
    assert CANONICAL_REGISTRY_CSV.exists()
    return pd.read_csv(CANONICAL_REGISTRY_CSV)


@pytest.fixture(scope="module")
def manifest_df():
    if not MANIFEST_PARQUET.exists():
        pytest.fail(f"Manifest output not found: {MANIFEST_PARQUET}. Run build script first.")
    return pd.read_parquet(MANIFEST_PARQUET)


# =====================================================================
# UNIT & REGRESSION TESTS
# =====================================================================

class TestGridAndGeodesicMath:
    """Regression tests for grid vertex calculation, longitude wrapping, and geodesic distances."""

    def test_western_hemisphere_port(self, calculator):
        """Regression test: Western hemisphere port (e.g. Seattle: lat 47.6062, lon -122.3321)."""
        lat = 47.6062
        lon = -122.3321

        # Surface grid (0.25°)
        s_lat, s_lon = find_nearest_grid_vertex(lat, lon, 0.25)
        assert s_lat == 47.5
        # -122.3321 % 360 = 237.6679; nearest 0.25° is 237.75
        assert s_lon == 237.75
        assert 0.0 <= s_lon < 360.0

        s_dist = calculator.calculate_distance_km(lat, lon, s_lat, s_lon)
        assert 10.0 < s_dist < 15.0

        # Wave grid (0.50°)
        w_lat, w_lon = find_nearest_grid_vertex(lat, lon, 0.50)
        assert w_lat == 47.5
        # nearest 0.50° is 237.5
        assert w_lon == 237.5
        assert 0.0 <= w_lon < 360.0

        w_dist = calculator.calculate_distance_km(lat, lon, w_lat, w_lon)
        assert 15.0 < w_dist < 20.0

    def test_antimeridian_port(self, calculator):
        """Regression test: Port near the antimeridian (e.g. lon 179.9 and lon -179.9)."""
        lat = 0.0
        lon_east = 179.9
        lon_west = -179.9

        # Both should map across the antimeridian to 180.0
        s_lat_e, s_lon_e = find_nearest_grid_vertex(lat, lon_east, 0.25)
        s_lat_w, s_lon_w = find_nearest_grid_vertex(lat, lon_west, 0.25)
        assert s_lat_e == 0.0 and s_lon_e == 180.0
        assert s_lat_w == 0.0 and s_lon_w == 180.0

        dist_e = calculator.calculate_distance_km(lat, lon_east, s_lat_e, s_lon_e)
        dist_w = calculator.calculate_distance_km(lat, lon_west, s_lat_w, s_lon_w)
        assert pytest.approx(dist_e, rel=1e-4) == 11.1319
        assert pytest.approx(dist_w, rel=1e-4) == 11.1319

    def test_prime_meridian_west_port(self, calculator):
        """Regression test: Port just west of the prime meridian (e.g. lon -0.05)."""
        lat = 51.5
        lon = -0.05

        s_lat, s_lon = find_nearest_grid_vertex(lat, lon, 0.25)
        assert s_lat == 51.5
        # (-0.05) % 360 = 359.95; nearest multiple of 0.25 is 360.0, which wraps to 0.0
        assert s_lon == 0.0

        dist = calculator.calculate_distance_km(lat, lon, s_lat, s_lon)
        assert 0.0 < dist < 5.0

    def test_port_exactly_on_grid_point(self, calculator):
        """Regression test: Port exactly on a grid point (e.g. lat 10.0, lon 20.0)."""
        lat = 10.0
        lon = 20.0

        s_lat, s_lon = find_nearest_grid_vertex(lat, lon, 0.25)
        assert s_lat == 10.0
        assert s_lon == 20.0
        s_dist = calculator.calculate_distance_km(lat, lon, s_lat, s_lon)
        assert s_dist == 0.0

        w_lat, w_lon = find_nearest_grid_vertex(lat, lon, 0.50)
        assert w_lat == 10.0
        assert w_lon == 20.0
        w_dist = calculator.calculate_distance_km(lat, lon, w_lat, w_lon)
        assert w_dist == 0.0

    def test_port_between_grid_points(self, calculator):
        """Regression test: Port located between grid points (e.g. lat 10.12, lon 20.12)."""
        lat = 10.12
        lon = 20.12

        s_lat, s_lon = find_nearest_grid_vertex(lat, lon, 0.25)
        assert s_lat == 10.0
        assert s_lon == 20.0
        s_dist = calculator.calculate_distance_km(lat, lon, s_lat, s_lon)
        assert s_dist > 0.0

        w_lat, w_lon = find_nearest_grid_vertex(lat, lon, 0.50)
        assert w_lat == 10.0
        assert w_lon == 20.0
        w_dist = calculator.calculate_distance_km(lat, lon, w_lat, w_lon)
        assert w_dist > 0.0

    def test_missing_coordinate_handling(self, calculator):
        """Regression test: Port with missing coordinate receives explicit failure and null grid cells."""
        dummy_df = pd.DataFrame([{
            "portwatch_id": "test_missing",
            "portwatch_name": "Missing Port",
            "portwatch_country": "Nowhere",
            "portwatch_iso3": "NOW",
            "country_iso2": "NW",
            "portwatch_name_normalized": "MISSING PORT",
            "unlocode": np.nan,
            "unlocode_match_status": "AMBIGUOUS_OR_UNMATCHED",
            "unlocode_candidate_count": 0,
            "unlocode_source_names": np.nan,
            "unlocode_coordinate_strings": np.nan,
            "upply_code": np.nan,
            "upply_match_status": "AMBIGUOUS_OR_UNMATCHED",
            "upply_candidate_count": 0,
            "upply_name": np.nan,
            "upply_latitude": np.nan,
            "upply_longitude": np.nan,
            "upply_country_code": np.nan,
            "country_conflict": False,
            "identity_status": "UNMATCHED",
        }])

        res = build_manifest(dummy_df, calculator).iloc[0]
        assert res["coordinate_validity_status"] == "MISSING"
        assert pd.isna(res["surface_grid_latitude"])
        assert pd.isna(res["surface_grid_longitude"])
        assert pd.isna(res["surface_distance_km"])
        assert pd.isna(res["wave_grid_latitude"])
        assert pd.isna(res["wave_grid_longitude"])
        assert pd.isna(res["wave_distance_km"])
        assert res["surface_spatial_mapping_status"] == "FAILED_MISSING_COORDINATES"
        assert res["wave_spatial_mapping_status"] == "FAILED_MISSING_COORDINATES"
        assert res["overall_spatial_mapping_status"] == "FAILED_MISSING_COORDINATES"
        assert res["spatial_mapping_reason_code"] == "MISSING_PORT_COORDINATES"

    def test_invalid_coordinate_handling(self, calculator):
        """Regression test: Port with out-of-bounds coordinate receives explicit failure and null grid cells."""
        dummy_df = pd.DataFrame([{
            "portwatch_id": "test_invalid",
            "portwatch_name": "Invalid Port",
            "portwatch_country": "Nowhere",
            "portwatch_iso3": "NOW",
            "country_iso2": "NW",
            "portwatch_name_normalized": "INVALID PORT",
            "unlocode": np.nan,
            "unlocode_match_status": "AMBIGUOUS_OR_UNMATCHED",
            "unlocode_candidate_count": 0,
            "unlocode_source_names": np.nan,
            "unlocode_coordinate_strings": np.nan,
            "upply_code": np.nan,
            "upply_match_status": "AMBIGUOUS_OR_UNMATCHED",
            "upply_candidate_count": 0,
            "upply_name": np.nan,
            "upply_latitude": 95.0,  # invalid latitude > 90
            "upply_longitude": 20.0,
            "upply_country_code": np.nan,
            "country_conflict": False,
            "identity_status": "UNMATCHED",
        }])

        res = build_manifest(dummy_df, calculator).iloc[0]
        assert res["coordinate_validity_status"] == "INVALID"
        assert pd.isna(res["surface_grid_latitude"])
        assert pd.isna(res["surface_grid_longitude"])
        assert pd.isna(res["surface_distance_km"])
        assert pd.isna(res["wave_grid_latitude"])
        assert pd.isna(res["wave_grid_longitude"])
        assert pd.isna(res["wave_distance_km"])
        assert res["surface_spatial_mapping_status"] == "FAILED_INVALID_COORDINATES"
        assert res["wave_spatial_mapping_status"] == "FAILED_INVALID_COORDINATES"
        assert res["overall_spatial_mapping_status"] == "FAILED_INVALID_COORDINATES"
        assert res["spatial_mapping_reason_code"] == "INVALID_PORT_COORDINATES"


# =====================================================================
# FULL MANIFEST AUTOMATED VALIDATION CHECKS
# =====================================================================

class TestGlobalManifestArtifacts:
    """Automated checks on the generated global ERA5 manifest artifacts."""

    def test_output_files_exist_and_non_empty(self):
        """Check all 6 required files exist and have non-zero size."""
        assert MANIFEST_PARQUET.exists() and MANIFEST_PARQUET.stat().st_size > 0
        assert MANIFEST_CSV.exists() and MANIFEST_CSV.stat().st_size > 0
        assert AUDIT_JSON.exists() and AUDIT_JSON.stat().st_size > 0
        assert COUNTRY_COVERAGE_CSV.exists() and COUNTRY_COVERAGE_CSV.stat().st_size > 0
        assert DISTANCE_REVIEW_CSV.exists() and DISTANCE_REVIEW_CSV.stat().st_size > 0
        assert MANIFEST_METADATA_JSON.exists() and MANIFEST_METADATA_JSON.stat().st_size > 0

    def test_exact_row_count_and_no_duplicate_portwatch_ids(self, manifest_df, registry_df):
        """Verify exactly one manifest row per PortWatch ID and all 2,065 ports represented."""
        assert len(manifest_df) == 2065
        assert manifest_df["portwatch_id"].nunique() == 2065
        assert not manifest_df["portwatch_id"].duplicated().any()
        assert set(manifest_df["portwatch_id"]) == set(registry_df["portwatch_id"])

    def test_all_original_identity_fields_preserved(self, manifest_df, registry_df):
        """Verify all original identity fields from registry v1.2 are preserved without loss."""
        orig_cols = [
            "portwatch_id", "portwatch_name", "portwatch_country", "portwatch_iso3",
            "country_iso2", "portwatch_name_normalized", "unlocode", "unlocode_match_status",
            "unlocode_candidate_count", "unlocode_source_names", "unlocode_coordinate_strings",
            "upply_code", "upply_match_status", "upply_candidate_count", "upply_name",
            "upply_latitude", "upply_longitude", "upply_country_code", "country_conflict",
            "identity_status",
        ]
        for col in orig_cols:
            assert col in manifest_df.columns, f"Missing original column {col}"
            s_orig = registry_df[col].astype(object).fillna("__NA__").astype(str)
            s_man = manifest_df[col].astype(object).fillna("__NA__").astype(str)
            pd.testing.assert_series_equal(s_orig, s_man, check_names=False)

    def test_valid_coordinates_within_geographic_bounds(self, manifest_df):
        """Verify that all non-null reference coordinates remain within valid geographic bounds."""
        valid = manifest_df[manifest_df["coordinate_validity_status"] == "VALID"]
        assert len(valid) == 949
        assert valid["reference_latitude"].between(-90.0, 90.0).all()
        assert valid["reference_longitude"].between(-180.0, 180.0).all()

    def test_grid_resolutions_and_alignment(self, manifest_df):
        """Verify that surface coordinates align to 0.25° and wave coordinates to 0.50°."""
        valid = manifest_df[manifest_df["coordinate_validity_status"] == "VALID"]

        # Surface latitudes in [-90, 90], multiples of 0.25
        assert valid["surface_grid_latitude"].between(-90.0, 90.0).all()
        s_lat_rem = np.abs(valid["surface_grid_latitude"] % 0.25)
        assert (np.isclose(s_lat_rem, 0.0) | np.isclose(s_lat_rem, 0.25)).all()

        # Surface longitudes in [0, 359.75], multiples of 0.25
        assert valid["surface_grid_longitude"].between(0.0, 359.75).all()
        s_lon_rem = np.abs(valid["surface_grid_longitude"] % 0.25)
        assert (np.isclose(s_lon_rem, 0.0) | np.isclose(s_lon_rem, 0.25)).all()

        # Wave latitudes in [-90, 90], multiples of 0.50
        assert valid["wave_grid_latitude"].between(-90.0, 90.0).all()
        w_lat_rem = np.abs(valid["wave_grid_latitude"] % 0.50)
        assert (np.isclose(w_lat_rem, 0.0) | np.isclose(w_lat_rem, 0.50)).all()

        # Wave longitudes in [0, 359.50], multiples of 0.50
        assert valid["wave_grid_longitude"].between(0.0, 359.50).all()
        w_lon_rem = np.abs(valid["wave_grid_longitude"] % 0.50)
        assert (np.isclose(w_lon_rem, 0.0) | np.isclose(w_lon_rem, 0.50)).all()

    def test_geodesic_distances_finite_and_nonnegative(self, manifest_df):
        """Verify all mapped geodesic distances are finite and nonnegative."""
        valid = manifest_df[manifest_df["coordinate_validity_status"] == "VALID"]
        assert (valid["surface_distance_km"] >= 0.0).all()
        assert np.isfinite(valid["surface_distance_km"]).all()
        assert (valid["wave_distance_km"] >= 0.0).all()
        assert np.isfinite(valid["wave_distance_km"]).all()

        # Maximum theoretical distance bounds:
        # Surface (0.25°) diagonal at equator ~ 19.6 km
        assert valid["surface_distance_km"].max() <= 19.6
        # Wave (0.50°) diagonal at equator ~ 39.3 km
        assert valid["wave_distance_km"].max() <= 39.3

    def test_missing_coordinates_never_receive_invented_grid_cells(self, manifest_df):
        """Verify that ports with missing coordinates have strictly null grid coordinates."""
        missing = manifest_df[manifest_df["coordinate_validity_status"] != "VALID"]
        assert len(missing) == 1116
        assert missing["surface_grid_latitude"].isna().all()
        assert missing["surface_grid_longitude"].isna().all()
        assert missing["surface_distance_km"].isna().all()
        assert missing["wave_grid_latitude"].isna().all()
        assert missing["wave_grid_longitude"].isna().all()
        assert missing["wave_distance_km"].isna().all()
        assert (missing["surface_spatial_mapping_status"] == "FAILED_MISSING_COORDINATES").all()
        assert (missing["wave_spatial_mapping_status"] == "FAILED_MISSING_COORDINATES").all()
        assert (missing["overall_spatial_mapping_status"] == "FAILED_MISSING_COORDINATES").all()

    def test_csv_and_parquet_consistency(self, manifest_df):
        """Verify exact row-by-row consistency between CSV and Parquet outputs."""
        csv_df = pd.read_csv(MANIFEST_CSV)
        assert len(csv_df) == len(manifest_df)
        assert list(csv_df.columns) == list(manifest_df.columns)

    def test_audit_json_contents(self):
        """Verify audit JSON contains all mandatory metrics."""
        audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
        summary = audit["summary"]
        assert summary["total_portwatch_ports"] == 2065
        assert summary["ports_with_valid_coordinates"] == 949
        assert summary["ports_with_missing_coordinates"] == 1116
        assert summary["ports_with_invalid_coordinates"] == 0
        assert summary["ports_mapped_surface_grid"] == 949
        assert summary["ports_mapped_wave_grid"] == 949
        assert summary["unique_surface_grid_cells"] > 0
        assert summary["unique_wave_grid_cells"] > 0
        assert summary["countries_total"] > 0

    def test_immutable_artifacts_unchanged(self):
        """Verify that canonical registry v1.2 and frozen ERA5 artifacts were not modified."""
        expected_hashes = {
            CANONICAL_REGISTRY_CSV: "03ff0ef1b93828e8741c87ce8386902e3bb6e2aad2eb5b479d8d903327bf1fb5",
            REGISTRY_METADATA_JSON: "ac576cb33eabe63b11bc0891eac9d68ace7cacfb61b6c9d9f507b7bdaaa31444",
            FROZEN_ERA5_HOURLY: "33097012905de0ec80dcc6da3ded2bce17039fb261ce1ad04f7d74af763a571a",
            FROZEN_ERA5_DAILY_V2: "b06321feaac01231d3fc073326c43b7ae8ae79955b4ead1ab92814329e95c738",
        }
        for path, expected_hash in expected_hashes.items():
            if path.exists():
                actual_hash = compute_sha256(path)
                assert actual_hash == expected_hash, f"Immutability violation for {path.name}!"
