"""Tests for the EDA Engineer modules (profiler, drift analysis, runner)."""

import numpy as np
import pandas as pd
import pytest

from ML.eda_engineer.drift_analysis import DriftAnalyzer
from ML.eda_engineer.eda_runner import EdaRunner, md_table
from ML.eda_engineer.profiler import EventLogProfiler


def _events():
    """Two cases: one clean, one with a 1954 sentinel start, a date-only customs end and a null start."""
    rows = [
        ("A-2021-07-10", "BAPLIE", "2021-07-09 10:00", "2021-07-09 11:00"),
        ("A-2021-07-10", "DISCHARGE", "2021-07-10 08:00", "2021-07-10 09:00"),
        ("A-2021-07-10", "CUSTOMS_DEL", "1954-04-02 07:26", "1954-07-14 00:00"),
        ("A-2021-07-10", "TRUCK_OUT", "2021-07-12 08:00", "2021-07-12 09:00"),
        ("B-2021-07-10", "BAPLIE", None, "2021-07-09 11:00"),
        ("B-2021-07-10", "DISCHARGE", "2021-07-10 08:00", "2021-07-10 09:00"),
        ("B-2021-07-10", "CUSTOMS_DEL", "2021-07-10 05:00", "2021-07-10 00:00"),
        ("B-2021-07-10", "TRUCK_OUT", "2021-07-11 08:00", "2021-07-11 09:00"),
    ]
    df = pd.DataFrame(rows, columns=["case_id", "activity", "Start Timestamp", "End Timestamp"])
    for c in ("Start Timestamp", "End Timestamp"):
        df[c] = pd.to_datetime(df[c])
    df["container_id"] = df["case_id"].str[:1]
    df["carrier_prefix"] = "AAA"
    df["event_duration_minutes"] = (df["End Timestamp"] - df["Start Timestamp"]).dt.total_seconds() / 60
    return df


def test_timestamp_quality_flags_sentinel_and_date_only():
    ledger = EventLogProfiler().timestamp_quality(_events()).set_index("check")
    assert ledger.loc["null_start_timestamp", "count"] == 1
    assert ledger.loc["implausible_start_year", "count"] == 1
    assert ledger.loc["date_only_end_timestamp:CUSTOMS_DEL", "count"] == 2  # both CUSTOMS_DEL ends are at midnight
    assert ledger.loc["end_before_start", "count"] == 1  # case B customs end precedes its start


def test_milestone_timing_marks_pre_discharge_activities():
    timing = EventLogProfiler().milestone_timing(_events()).set_index("activity")
    assert timing.loc["BAPLIE", "pct_before_discharge"] == 100.0
    assert timing.loc["TRUCK_OUT", "pct_before_discharge"] == 0.0
    assert timing.loc["CUSTOMS_DEL", "n"] == 1  # the 1954 sentinel is excluded from offsets


@pytest.fixture
def window_cases():
    """Stays with gate-outs from 2021-07-01 to 2021-09-30; dwell lognormal around 50h."""
    rng = np.random.default_rng(0)
    n = 6000
    discharge = pd.Timestamp("2021-06-15") + pd.to_timedelta(rng.uniform(0, 107 * 24, n), unit="h")
    dwell = np.exp(rng.normal(np.log(50), 0.8, n))
    out = discharge + pd.to_timedelta(dwell, unit="h")
    df = pd.DataFrame({"discharge_start_ts": discharge, "truck_out_end_ts": out, "dwell_hours": dwell})
    df = df[(df["truck_out_end_ts"] >= "2021-07-01") & (df["truck_out_end_ts"] < "2021-10-01")].copy()
    df["call_date"] = df["discharge_start_ts"].dt.strftime("%Y-%m-%d")
    df["split_tag"] = "train"
    return df.reset_index(drop=True)


def test_observation_window_and_clean_band(window_cases):
    da = DriftAnalyzer()
    band = da.observation_window(window_cases)
    assert band["window_start"] == pd.Timestamp("2021-07-01")
    assert band["window_end"] == pd.Timestamp("2021-09-30")
    assert band["clean_end"] < band["window_end"]
    mask = da.clean_mask(window_cases, band)
    assert 0 < mask.sum() < len(window_cases)
    call = pd.to_datetime(window_cases.loc[mask, "call_date"])
    assert call.min() >= band["clean_start"] and call.max() <= band["clean_end"]


def test_observable_share_drops_at_both_edges(window_cases):
    da = DriftAnalyzer()
    band = da.observation_window(window_cases)
    share = da.observable_share(window_cases, band)
    call = pd.to_datetime(window_cases["call_date"])
    assert share[call >= "2021-09-28"].mean() < 0.9      # right edge: slow stays not yet out
    assert share[call < "2021-06-28"].mean() < 0.9       # left edge: only slow stays reach the window
    mid = (call >= "2021-07-10") & (call <= "2021-08-31")
    assert share[mid].mean() > 0.99


def test_psi_detects_shift_and_ignores_identity():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 5000)
    assert DriftAnalyzer._psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert DriftAnalyzer._psi(a, rng.normal(1.5, 1, 5000)) > 0.25
    flags_a, flags_b = np.r_[np.zeros(900), np.ones(100)], np.r_[np.zeros(500), np.ones(500)]
    assert DriftAnalyzer._psi(flags_a, flags_b) > 0.25  # binary features fall back to category shares


def test_noise_floor_matches_binomial_for_iid_series():
    rng = np.random.default_rng(2)
    days = pd.date_range("2021-07-01", periods=90, freq="D")
    volume = np.full(90, 1000)
    delayed = rng.binomial(volume, 0.4)
    dep = pd.DataFrame({"departure_date": days, "volume": volume, "delayed": delayed,
                        "mean_dwell_h": 50.0, "median_dwell_h": 50.0, "delay_share": delayed / volume})
    row = DriftAnalyzer().noise_floor(dep).set_index("metric").loc["daily_delay_share"]
    assert 0.6 < row["overdispersion"] < 1.6


def test_coverage_variation_flags_real_movement():
    weeks = pd.date_range("2021-07-05", periods=8, freq="7D")
    cov = pd.DataFrame({"STABLE": [30.0] * 8, "MOVING": [10, 20, 30, 40, 10, 20, 30, 40]}, index=weeks)
    out = DriftAnalyzer.coverage_variation(cov, pd.Series(5000, index=weeks)).set_index("activity")
    assert out.loc["MOVING", "ratio"] > 10 > out.loc["STABLE", "ratio"]


def test_md_table_renders_booleans_and_nan():
    text = md_table(pd.DataFrame({"a": [True, False], "b": [np.nan, 1.5], "c": [1200, 3]}))
    assert "yes" in text and "no" in text and "–" in text and "1,200" in text


def test_runner_end_to_end_on_real_outputs(tmp_path):
    from pathlib import Path
    if not (Path("data/processed/train.parquet").exists() and Path("data/interim/cleaned_event_log.parquet").exists()):
        pytest.skip("run ML/data_engineer/build_pipeline.py first")
    report = EdaRunner(out_dir=str(tmp_path), make_figures=False).run()
    text = report.read_text()
    assert "## 1. Headline findings" in text and "clean band" in text
    assert (tmp_path / "tables" / "truncation.csv").exists()
    assert (tmp_path / "tables" / "signal_arrival.csv").exists()
    assert (tmp_path / "tables" / "signal_diagnostic.csv").exists()


def test_feature_signal_annotates_eligibility_and_filters_arrival_leakage():
    """Verifies that feature_signal tags eligibility tiers and filter_eligible_signals removes post-discharge leakage."""
    rng = np.random.default_rng(42)
    n = 200
    dwell = rng.uniform(20, 200, n)
    df = pd.DataFrame({
        "dwell_hours": dwell,
        "carrier_prefix": rng.choice(["MSK", "CMA", "ONE"], n),
        "CTR_SIZE": rng.choice([20, 40], n),
        "JOB_DEL_DOCTYPE": rng.choice(["SPPB", "DO", "BC23"], n),
        "total_events_in_stay": rng.integers(5, 25, n),
        "has_behandle_inspection": rng.choice([0, 1], n),
        "trailing_7d_volume": rng.uniform(500, 2000, n),
    })

    profiler = EventLogProfiler()
    numeric = ["CTR_SIZE", "total_events_in_stay", "has_behandle_inspection", "trailing_7d_volume"]
    categorical = ["carrier_prefix", "JOB_DEL_DOCTYPE"]

    sig = profiler.feature_signal(df, numeric=numeric, categorical=categorical)
    assert "eligibility_tier" in sig.columns
    assert "arrival_eligible" in sig.columns

    sig_map = sig.set_index("feature")["eligibility_tier"].to_dict()
    assert sig_map["carrier_prefix"] == "ARRIVAL_TIME_ELIGIBLE"
    assert sig_map["CTR_SIZE"] == "ARRIVAL_TIME_ELIGIBLE"
    assert sig_map["trailing_7d_volume"] == "ARRIVAL_TIME_ELIGIBLE"
    assert sig_map["JOB_DEL_DOCTYPE"] == "POST_DISCHARGE_DIAGNOSTIC"
    assert sig_map["total_events_in_stay"] == "POST_DISCHARGE_DIAGNOSTIC"
    assert sig_map["has_behandle_inspection"] == "POST_DISCHARGE_DIAGNOSTIC"

    # Filter for arrival-time models: post-discharge features MUST NOT be present
    arrival_sig = profiler.filter_eligible_signals(sig, scenario="ARRIVAL_TIME")
    arrival_features = set(arrival_sig["feature"])
    assert "JOB_DEL_DOCTYPE" not in arrival_features
    assert "total_events_in_stay" not in arrival_features
    assert "has_behandle_inspection" not in arrival_features
    assert "CTR_SIZE" in arrival_features
    assert "carrier_prefix" in arrival_features


def test_date_only_milestone_same_vs_different_day():
    """Verifies that date-only milestones report timestamp_precision='date_only' and calendar day breakdown."""
    rows = [
        # Case 1: Same calendar day (discharge at 14:00, customs midnight 00:00 -> sub-day negative offset -14h)
        ("C1", "DISCHARGE", "2021-07-10 14:00:00", "2021-07-10 15:00:00"),
        ("C1", "CUSTOMS_DEL", None, "2021-07-10 00:00:00"),
        # Case 2: Strictly prior calendar day (discharge 2021-07-11 10:00, customs 2021-07-09 00:00 -> -34h)
        ("C2", "DISCHARGE", "2021-07-11 10:00:00", "2021-07-11 11:00:00"),
        ("C2", "CUSTOMS_DEL", None, "2021-07-09 00:00:00"),
        # Case 3: Later calendar day (discharge 2021-07-10 10:00, customs 2021-07-12 00:00 -> +38h)
        ("C3", "DISCHARGE", "2021-07-10 10:00:00", "2021-07-10 11:00:00"),
        ("C3", "CUSTOMS_DEL", None, "2021-07-12 00:00:00"),
    ]
    df = pd.DataFrame(rows, columns=["case_id", "activity", "Start Timestamp", "End Timestamp"])
    for c in ("Start Timestamp", "End Timestamp"):
        df[c] = pd.to_datetime(df[c])

    profiler = EventLogProfiler()
    timing = profiler.milestone_timing(df).set_index("activity")

    assert "CUSTOMS_DEL" in timing.index
    row = timing.loc["CUSTOMS_DEL"]
    assert row["timestamp_precision"] == "date_only"
    # 1 of 3 strictly prior calendar day = 33.3%
    assert 30.0 < row["pct_calendar_prior"] < 35.0
    # 1 of 3 same calendar day = 33.3%
    assert 30.0 < row["pct_calendar_same_day"] < 35.0
    assert "Date-only" in row["timing_note"]
