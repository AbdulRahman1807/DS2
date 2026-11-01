"""test_eda_engineer.py - Verification suite for Stage 02 Deep Learning EDA Deliverables.
"""

import os
import pytest
import pandas as pd
from pathlib import Path

DL_ROOT = Path(__file__).resolve().parents[1]
EDA_DIR = DL_ROOT / "eda" / "eda_reports"
TABLES_DIR = EDA_DIR / "tables"
FIGURES_DIR = EDA_DIR / "figures"
TELEMETRY_CSV = DL_ROOT / "data" / "hourly_node_telemetry.csv"
EDGES_CSV = DL_ROOT / "data" / "edges.csv"

def test_telemetry_dataset_integrity():
    assert TELEMETRY_CSV.exists(), "hourly_node_telemetry.csv does not exist"
    df = pd.read_csv(TELEMETRY_CSV)
    assert len(df) == 21810, f"Expected 21,810 records, got {len(df)}"
    assert df["node_id"].nunique() == 6, "Expected 6 nodes"
    
    # Check zero duplicates
    dups = df.duplicated(subset=["node_id", "timestamp"]).sum()
    assert dups == 0, f"Detected {dups} duplicate records"

def test_slide18_node_mean_dwell_reconciliation():
    df = pd.read_csv(TELEMETRY_CSV)
    means = df.groupby("node_id")["dwell_time"].mean().to_dict()
    
    expected = {
        "DC_US_MW_JOLIET": 8.4,
        "PORT_CN_SHA": 17.8,
        "PORT_US_LAX": 31.1,
        "PORT_US_LGB": 23.8,
        "PORT_US_OAK": 19.1,
        "RAIL_US_CHI_BNSF": 18.6
    }
    
    for nid, exp_mean in expected.items():
        assert nid in means, f"Node {nid} missing from telemetry"
        assert abs(means[nid] - exp_mean) < 0.15, f"{nid} mean {means[nid]:.2f} deviates from expected {exp_mean}"

def test_all_eda_tables_exist_and_non_empty():
    expected_tables = [
        "01_signal_descriptive_stats.csv",
        "02_temporal_gaps_and_balance.csv",
        "03_signal_correlation_matrix.csv",
        "04_lead_lag_cross_correlation.csv",
        "05_graph_neighbour_baseline.csv",
        "06_split_drift_ks_psi.csv",
        "07_cross_resolution_benchmark.csv",
        "08_node_profile_summary.csv",
        "09_sliding_window_metadata.csv"
    ]
    for tbl in expected_tables:
        p = TABLES_DIR / tbl
        assert p.exists(), f"Missing table: {tbl}"
        df = pd.read_csv(p)
        assert len(df) > 0, f"Table {tbl} is empty"

def test_all_eda_figures_exist_and_valid():
    expected_figures = [
        "01_hourly_dwell_trends_by_node.png",
        "02_eight_signal_distributions.png",
        "03_correlation_heatmap.png",
        "04_lead_lag_cross_correlation.png",
        "05_network_topology_and_neighbour_influence.png",
        "06_train_test_split_drift.png",
        "07_sliding_window_trajectory_profiles.png"
    ]
    for fig in expected_figures:
        p = FIGURES_DIR / fig
        assert p.exists(), f"Missing figure: {fig}"
        assert p.stat().st_size > 10000, f"Figure {fig} file size suspiciously small"

def test_canonical_reports_exist():
    report_txt = EDA_DIR / "eda_report.txt"
    findings_md = EDA_DIR / "findings.md"
    assert report_txt.exists(), "eda_report.txt is missing"
    assert findings_md.exists(), "findings.md is missing"
    assert report_txt.stat().st_size > 1000, "eda_report.txt is too short"
    assert findings_md.stat().st_size > 2000, "findings.md is too short"

