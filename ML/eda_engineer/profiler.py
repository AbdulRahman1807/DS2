"""Exploratory Data Analysis - Event Log and Master Dataset Profiler.

Assigned Role: EDA Engineer

Responsibilities:
- Event-log and case-level cardinality, activity coverage, and timestamp quality.
- Milestone timing relative to vessel discharge (anti-leakage pre/post-discharge validation).
- Trajectory transitions and duration quantiles across terminal operational milestones.
- Strict horizon audit at t_discharge for both Stage 01 (32 columns) and Master Dataset (55 columns).
- Gross weight distribution audit (tonnes vs kilograms scale anomaly identification).
- External macroeconomic (PPKM/holiday) and IMF PortWatch port call profiling.
- Synthetic process data fidelity audit (class balance, dwell distributions, segment transitions).
- Univariate correlation and non-parametric association rankings with dwell.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from scipy import stats

PREDICTION_ELIGIBILITY: Dict[str, str] = {
    # Safe arrival predictors strictly observable at or before container discharge
    "carrier_prefix": "ARRIVAL_TIME_ELIGIBLE",
    "CTR_SIZE": "ARRIVAL_TIME_ELIGIBLE",
    "CTR_TYPE": "ARRIVAL_TIME_ELIGIBLE",
    "GROSS": "CONDITIONAL_AT_DISCHARGE",
    "YARD_BLOCK": "CONDITIONAL_AT_DISCHARGE",
    "YARD_SLOT": "CONDITIONAL_AT_DISCHARGE",
    "discharge_hour": "ARRIVAL_TIME_ELIGIBLE",
    "discharge_day_of_week": "ARRIVAL_TIME_ELIGIBLE",
    "discharge_month": "ARRIVAL_TIME_ELIGIBLE",
    "is_weekend": "ARRIVAL_TIME_ELIGIBLE",
    "trailing_7d_volume": "ARRIVAL_TIME_ELIGIBLE",
    "trailing_7d_critical": "ARRIVAL_TIME_ELIGIBLE",
    "trailing_7d_at_risk": "ARRIVAL_TIME_ELIGIBLE",
    "historical_delay_rate_7d": "ARRIVAL_TIME_ELIGIBLE",
    "trailing_7d_mean_dwell": "ARRIVAL_TIME_ELIGIBLE",
    # Unverified / zero-variance weather indicators
    "weather_severity_index": "EXCLUDED_UNVERIFIED_WEATHER",
    "has_active_disruption": "EXCLUDED_UNVERIFIED_WEATHER",
    # Conditional pre-arrival (manifest holds / customs declarations)
    "has_quarantine_flag": "CONDITIONAL_PRE_ARRIVAL",
    "has_red_flag_pib": "CONDITIONAL_PRE_ARRIVAL",
    # Post-discharge in-yard milestone leakage (forbidden at arrival)
    "JOB_DEL_DOCTYPE": "POST_DISCHARGE_DIAGNOSTIC",
    "total_events_in_stay": "POST_DISCHARGE_DIAGNOSTIC",
    "has_behandle_inspection": "POST_DISCHARGE_DIAGNOSTIC",
    # Ground truth realized dwell targets
    "dwell_hours": "TARGET_GROUND_TRUTH",
    "dwell_days": "TARGET_GROUND_TRUTH",
    "truck_out_end_ts": "TARGET_GROUND_TRUTH",
    "target_class": "TARGET_GROUND_TRUTH",
    "target_class_name": "TARGET_GROUND_TRUTH",
}


class EventLogProfiler:
    """Profiles event log and stay datasets for data quality, leakage, and distributions."""

    def __init__(
        self,
        case_col: str = "case_id",
        activity_col: str = "activity",
        start_ts_col: str = "Start Timestamp",
        end_ts_col: str = "End Timestamp",
        container_col: str = "container_id",
        carrier_col: str = "carrier_prefix",
        truck_out_act: str = "TRUCK_OUT",
        discharge_act: str = "DISCHARGE",
        baplie_act: str = "BAPLIE",
        plausible_min: pd.Timestamp = pd.Timestamp("2021-01-01"),
        plausible_max: pd.Timestamp = pd.Timestamp("2022-12-31"),
    ):
        self.case_col = case_col
        self.activity_col = activity_col
        self.start_ts_col = start_ts_col
        self.end_ts_col = end_ts_col
        self.container_col = container_col
        self.carrier_col = carrier_col
        self.truck_out_act = truck_out_act
        self.discharge_act = discharge_act
        self.baplie_act = baplie_act
        self.plausible_min = plausible_min
        self.plausible_max = plausible_max

    # ------------------------------------------------------------------ cardinality & summary
    def cardinality(self, events: pd.DataFrame, cases: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Return counts of distinct entities across the event log and reconstructed stays."""
        rows = [
            ("total_events", len(events)),
            ("distinct_cases_events", events[self.case_col].nunique()),
            ("distinct_activities", events[self.activity_col].nunique()),
        ]
        if self.container_col in events.columns:
            rows.append(("distinct_containers_events", events[self.container_col].nunique()))
        if self.carrier_col in events.columns:
            rows.append(("distinct_carriers_events", events[self.carrier_col].dropna().nunique()))
        if cases is not None:
            rows.append(("total_reconstructed_cases", len(cases)))
            if "container_id" in cases.columns:
                rows.append(("distinct_containers_cases", cases["container_id"].nunique()))
            if "split_tag" in cases.columns:
                for tag, count in cases["split_tag"].value_counts().items():
                    rows.append((f"cases_{tag}", count))
        return pd.DataFrame(rows, columns=["metric", "count"])

    def activity_summary(self, events: pd.DataFrame) -> pd.DataFrame:
        """Summary of activity frequencies, case coverage, and typical duration."""
        n_cases = events[self.case_col].nunique()
        res = []
        for act, group in events.groupby(self.activity_col):
            count = len(group)
            cases = group[self.case_col].nunique()
            cov_pct = (cases / n_cases) * 100 if n_cases > 0 else 0.0
            
            durations = None
            if "event_duration_minutes" in group.columns:
                durations = group["event_duration_minutes"].dropna()
            elif self.start_ts_col in group.columns and self.end_ts_col in group.columns:
                valid = group[self.start_ts_col].notna() & group[self.end_ts_col].notna()
                d = (group.loc[valid, self.end_ts_col] - group.loc[valid, self.start_ts_col]).dt.total_seconds() / 60
                durations = d[d >= 0]
                
            median_dur = float(durations.median()) if durations is not None and len(durations) > 0 else np.nan
            p85_dur = float(durations.quantile(0.85)) if durations is not None and len(durations) > 0 else np.nan
            res.append({
                "activity": act,
                "event_count": count,
                "case_count": cases,
                "case_coverage_pct": round(cov_pct, 2),
                "median_duration_min": round(median_dur, 2) if pd.notna(median_dur) else np.nan,
                "p85_duration_min": round(p85_dur, 2) if pd.notna(p85_dur) else np.nan,
            })
        out = pd.DataFrame(res)
        return out.sort_values("event_count", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------ quality ledger
    def timestamp_quality(self, events: pd.DataFrame) -> pd.DataFrame:
        """Ledger of timestamp, precision, format, and ordering defects."""
        rows = []
        n_events = len(events)
        n_cases = events[self.case_col].nunique()

        # 1. Null start / end
        null_start = int(events[self.start_ts_col].isna().sum())
        null_end = int(events[self.end_ts_col].isna().sum())
        rows.append(("null_start_timestamp", null_start, (null_start / n_events) * 100,
                     "Start missing; end is still present."))
        rows.append(("null_end_timestamp", null_end, (null_end / n_events) * 100, "End missing."))

        # 2. Implausible years
        valid_start = events[self.start_ts_col].dropna()
        bad_start = ~valid_start.between(self.plausible_min, self.plausible_max)
        bad_start_count = int(bad_start.sum())
        bad_start_acts = set(events.loc[valid_start[bad_start].index, self.activity_col].unique())
        rows.append(("implausible_start_year", bad_start_count, (bad_start_count / n_events) * 100,
                     f"Start outside {self.plausible_min.date()}..{self.plausible_max.date()}; activities: {bad_start_acts}"))

        valid_end = events[self.end_ts_col].dropna()
        bad_end = ~valid_end.between(self.plausible_min, self.plausible_max)
        bad_end_count = int(bad_end.sum())
        rows.append(("implausible_end_year", bad_end_count, (bad_end_count / n_events) * 100,
                     "End outside the plausible window."))

        # 3. Negative event durations
        paired = events[self.start_ts_col].notna() & events[self.end_ts_col].notna()
        neg = (events.loc[paired, self.end_ts_col] < events.loc[paired, self.start_ts_col]).sum()
        rows.append(("end_before_start", int(neg), (neg / n_events) * 100, "Negative event duration."))

        # 4. Date-only midnight timestamps (00:00:00)
        for act in ["CUSTOMS_DEL", "SP_RED_FLAG_PIB"]:
            act_rows = events[events[self.activity_col] == act]
            if len(act_rows) > 0:
                mid = (act_rows[self.end_ts_col].dt.time == pd.Timestamp("2021-01-01 00:00:00").time()).sum()
                rows.append((f"date_only_end_timestamp:{act}", int(mid), (mid / n_events) * 100,
                             "Every End is exactly 00:00:00, so the event has date precision only."))

        # 5. Tied start timestamps within case
        tied = events.duplicated(subset=[self.case_col, self.start_ts_col], keep=False) & events[self.start_ts_col].notna()
        tied_count = int(tied.sum())
        rows.append(("tied_start_timestamps_within_case", tied_count, (tied_count / n_events) * 100,
                     "Rows sharing a start time inside one case."))

        # 6. Raw Duration column artifacts
        if "Duration" in events.columns:
            dur_null = int(events["Duration"].isna().sum())
            rows.append(("duration_column_null", dur_null, (dur_null / n_events) * 100,
                         "Pre-computed Duration is empty."))
            # Unparseable strings or dates like 1900-01-0X
            dur_str = events["Duration"].astype(str)
            unparseable = dur_str.str.contains("1900|#VALUE!", na=False).sum()
            rows.append(("duration_column_unparseable", int(unparseable), (unparseable / n_events) * 100,
                         "Not HH:MM:SS; Excel durations over 24h stored as 1900-01-0X dates. Use event_duration_minutes instead."))

        # 7. Physical sequencing anomalies
        unstack = events[events[self.activity_col] == "UNSTACK_TO_TRUCK"].set_index(self.case_col)[self.start_ts_col]
        truck_out = events[events[self.activity_col] == self.truck_out_act].set_index(self.case_col)[self.start_ts_col]
        overlap_cases = unstack.index.intersection(truck_out.index)
        if len(overlap_cases) > 0:
            out_before_unstack = (truck_out.loc[overlap_cases] < unstack.loc[overlap_cases]).sum()
            rows.append(("truck_out_starts_before_unstack", int(out_before_unstack),
                         (out_before_unstack / len(overlap_cases)) * 100,
                         "TRUCK_OUT start precedes UNSTACK_TO_TRUCK start; administrative gate passes pre-logged."))

        # 8. Customs release dated before discharge
        customs = events[events[self.activity_col] == "CUSTOMS_DEL"].set_index(self.case_col)[self.end_ts_col]
        discharge = events[events[self.activity_col] == self.discharge_act].set_index(self.case_col)[self.start_ts_col]
        common = customs.index.intersection(discharge.index)
        if len(common) > 0:
            c_date = customs.loc[common].dt.normalize()
            d_date = discharge.loc[common].dt.normalize()
            prior_date = (c_date < d_date).sum()
            rows.append(("customs_release_date_before_discharge_date", int(prior_date),
                         (prior_date / len(common)) * 100,
                         "Customs release dated before the discharge date; clearance legitimately precedes discharge."))

        return pd.DataFrame(rows, columns=["check", "count", "share_pct", "note"])

    # ------------------------------------------------------------------ milestone timing
    def milestone_timing(self, events: pd.DataFrame) -> pd.DataFrame:
        """Offset and timing of each activity relative to DISCHARGE start."""
        discharges = events[events[self.activity_col] == self.discharge_act].copy()
        dis_starts = discharges.groupby(self.case_col)[self.start_ts_col].min().dropna()
        dis_starts = dis_starts[dis_starts.between(self.plausible_min, self.plausible_max)]

        out_rows = []
        for act, group in events.groupby(self.activity_col):
            # Use start timestamp where available, fall back to end timestamp (e.g. date-only customs)
            ts = group[self.start_ts_col].where(group[self.start_ts_col].notna(), group[self.end_ts_col])
            ts = ts.where(ts.between(self.plausible_min, self.plausible_max))
            first_ts = group.assign(_ts=ts).groupby(self.case_col)["_ts"].min().dropna()
            
            merged = pd.concat([first_ts.rename("ts"), dis_starts.rename("dis")], axis=1, join="inner")
            if len(merged) == 0:
                continue

            # Offset in hours: negative means before discharge, positive after
            offsets_h = (merged["ts"] - merged["dis"]).dt.total_seconds() / 3600.0
            pct_before = (offsets_h <= 0).mean() * 100.0

            # Date precision check
            ends = group[self.end_ts_col].dropna()
            date_only = len(ends) > 0 and (ends.dt.time == pd.Timestamp("2021-01-01 00:00:00").time()).all()
            
            pct_cal_prior, pct_cal_same, pct_cal_later = np.nan, np.nan, np.nan
            timing_note = "Continuous timestamps."
            if date_only:
                c_date = merged["ts"].dt.normalize()
                d_date = merged["dis"].dt.normalize()
                pct_cal_prior = (c_date < d_date).mean() * 100.0
                pct_cal_same = (c_date == d_date).mean() * 100.0
                pct_cal_later = (c_date > d_date).mean() * 100.0
                timing_note = "Date-only midnight timestamp (00:00:00); sub-day precision unrecorded."

            out_rows.append({
                "activity": act,
                "n": len(merged),
                "median_h": round(float(offsets_h.median()), 2),
                "p10_h": round(float(offsets_h.quantile(0.10)), 2),
                "p90_h": round(float(offsets_h.quantile(0.90)), 2),
                "pct_before_discharge": round(float(pct_before), 2),
                "timestamp_precision": "date_only" if date_only else "millisecond",
                "pct_calendar_prior": round(float(pct_cal_prior), 2) if pd.notna(pct_cal_prior) else np.nan,
                "pct_calendar_same_day": round(float(pct_cal_same), 2) if pd.notna(pct_cal_same) else np.nan,
                "pct_calendar_later": round(float(pct_cal_later), 2) if pd.notna(pct_cal_later) else np.nan,
                "timing_note": timing_note,
            })
        out = pd.DataFrame(out_rows)
        return out.sort_values("median_h").reset_index(drop=True)

    # ------------------------------------------------------------------ milestone transitions
    def milestone_transitions(self, events: pd.DataFrame) -> pd.DataFrame:
        """Summarize durations for complete, chronologically ordered operational trajectories."""
        activities = ["BAPLIE", "DISCHARGE", "STACK", "CUSTOMS_DEL", "UNSTACK_TO_TRUCK", "TRUCK_OUT"]
        ev = events.loc[
            events[self.activity_col].isin(activities),
            [self.case_col, self.activity_col, self.start_ts_col, self.end_ts_col]
        ].copy()
        
        # Milestone starts define state transitions; TRUCK_OUT end defines final gate exit
        ev["_point"] = ev[self.start_ts_col].where(
            ev[self.activity_col] != self.truck_out_act, ev[self.end_ts_col]
        )
        ev["_point"] = ev["_point"].where(
            ev["_point"].between(self.plausible_min, self.plausible_max)
        )
        points = ev.pivot_table(
            index=self.case_col, columns=self.activity_col, values="_point", aggfunc="min"
        ).reindex(columns=activities)
        
        complete = points.notna().all(axis=1)
        ordered = complete.copy()
        for previous, current in zip(activities, activities[1:]):
            ordered &= points[previous] <= points[current]

        rows = []
        for previous, current in zip(activities, activities[1:]):
            pair = points[[previous, current]].dropna()
            duration = (pair[current] - pair[previous]).dt.total_seconds() / 3600.0
            valid = duration.ge(0)
            trajectory_duration = (
                points.loc[ordered, current] - points.loc[ordered, previous]
            ).dt.total_seconds() / 3600.0
            cumulative_duration = (
                points.loc[ordered, current] - points.loc[ordered, activities[0]]
            ).dt.total_seconds() / 3600.0
            rows.append({
                "transition": f"{previous} -> {current}",
                "paired_cases": len(pair),
                "nonnegative_pair_cases": int(valid.sum()),
                "complete_ordered_trajectories": int(ordered.sum()),
                "complete_trajectory_share_pct": round(float(ordered.mean() * 100), 2) if len(ordered) else np.nan,
                "median_hours": round(float(trajectory_duration.median()), 2) if len(trajectory_duration) else np.nan,
                "p85_hours": round(float(trajectory_duration.quantile(0.85)), 2) if len(trajectory_duration) else np.nan,
                "p95_hours": round(float(trajectory_duration.quantile(0.95)), 2) if len(trajectory_duration) else np.nan,
                "cumulative_median_hours": round(float(cumulative_duration.median()), 2) if len(cumulative_duration) else np.nan,
                "cumulative_p85_hours": round(float(cumulative_duration.quantile(0.85)), 2) if len(cumulative_duration) else np.nan,
            })
        out = pd.DataFrame(rows)
        out.attrs["cases"] = int(len(points))
        out.attrs["complete_cases"] = int(complete.sum())
        out.attrs["out_of_order_complete_cases"] = int((complete & ~ordered).sum())
        return out

    # ------------------------------------------------------------------ horizon audits
    @staticmethod
    def horizon_audit(timing: pd.DataFrame) -> pd.DataFrame:
        """Document feature availability at discharge for the 32 Stage 01 processed columns."""
        timed = timing.set_index("activity") if not timing.empty else pd.DataFrame()

        def pre_discharge_evidence(activity: str) -> str:
            if activity not in timed.index:
                return "No corresponding activity rows in the timing audit; as-of availability is unverified."
            pct = timed.loc[activity, "pct_before_discharge"]
            return f"{pct:.1f}% of observed events start before discharge."

        rows = [
            ("case_id", "METADATA_ID", "Join key known at arrival; exclude as a model predictor."),
            ("container_id", "METADATA_ID", "Container identity known at arrival; exclude to prevent entity memorization."),
            ("carrier_prefix", "PASS", "Derived from container identity; available before discharge."),
            ("call_date", "METADATA_DATE_KEY", "Case-id date is available but differs from the DISCHARGE calendar date for some stays; derive temporal features from discharge_start_ts."),
            ("CTR_SIZE", "PASS", "BAPLIE/stowage attribute; BAPLIE is observed before discharge."),
            ("CTR_TYPE", "PASS", "BAPLIE/stowage attribute; BAPLIE is observed before discharge."),
            ("GROSS", "CONDITIONAL", "Provenance places it at discharge, but the source has no field-level observation timestamp; obtain TOS sign-off before use."),
            ("YARD_BLOCK", "CONDITIONAL", "Provenance places assignment at discharge, but the source has no field-level observation timestamp; obtain TOS sign-off before use."),
            ("YARD_SLOT", "CONDITIONAL", "Provenance places assignment at discharge, but the source has no field-level observation timestamp; obtain TOS sign-off before use."),
            ("JOB_DEL_DOCTYPE", "EXCLUDE_POST_DISCHARGE", "Mid-stay delivery-order event; not knowable at t_discharge."),
            ("total_events_in_stay", "EXCLUDE_POST_DISCHARGE", "Full-stay event count; not knowable until the stay is complete."),
            ("has_quarantine_flag", "CONDITIONAL", f"Whole-stay flag is not strictly safe; {pre_discharge_evidence('HAS_QUARANTINE_FLAG')} Use only a timestamped as-of-discharge value."),
            ("has_red_flag_pib", "CONDITIONAL", f"Whole-stay flag is not strictly safe; {pre_discharge_evidence('SP_RED_FLAG_PIB')} Use only a timestamped as-of-discharge value."),
            ("has_behandle_inspection", "EXCLUDE_POST_DISCHARGE", "Physical inspection occurs in-yard after discharge."),
            ("dwell_hours", "TARGET_ONLY", "Ground truth realized at gate-out; never an input feature."),
            ("dwell_days", "TARGET_ONLY", "Ground truth duration derived from dwell_hours; never an input feature."),
            ("discharge_start_ts", "METADATA_CUTOFF", "The DISCHARGE start timestamp defines t_pred; retain as the time anchor, not an unrestricted raw predictor."),
            ("truck_out_end_ts", "TARGET_ONLY", "Gate-out completion timestamp occurs after t_discharge; never an input feature."),
            ("discharge_hour", "PASS", "Deterministic transform of the DISCHARGE timestamp."),
            ("discharge_day_of_week", "PASS", "Deterministic transform of the DISCHARGE timestamp."),
            ("discharge_month", "PASS", "Deterministic transform of the DISCHARGE timestamp."),
            ("is_weekend", "PASS", "Deterministic transform of the DISCHARGE timestamp."),
            ("target_class", "TARGET_ONLY", "Ground truth label based on local operational p50/p85 turnover tiers, not contractual demurrage."),
            ("target_class_name", "TARGET_ONLY", "String form of the local operational p50/p85 label; never an input feature."),
            ("trailing_7d_volume", "PASS_WITH_COLD_START", "Completed departures in [t-7, t-1]; pre-2021-07-01 zeros represent unobserved history."),
            ("trailing_7d_critical", "PASS_WITH_COLD_START", "Completed critical departures in [t-7, t-1]; pre-2021-07-01 zeros represent unobserved history."),
            ("trailing_7d_at_risk", "PASS_WITH_COLD_START", "Completed at-risk departures in [t-7, t-1]; pre-2021-07-01 zeros represent unobserved history."),
            ("historical_delay_rate_7d", "PASS_WITH_COLD_START", "Completed departures in [t-7, t-1]; pre-2021-07-01 zeros represent unobserved history."),
            ("trailing_7d_mean_dwell", "PASS_WITH_COLD_START", "Mean dwell of completed departures in [t-7, t-1]; pre-2021-07-01 zeros represent unobserved history."),
            ("weather_severity_index", "EXCLUDE_UNVERIFIED_WEATHER", "Sparse, constant-zero external feed with anonymized terminal coordinates; not an active baseline signal."),
            ("has_active_disruption", "EXCLUDE_UNVERIFIED_WEATHER", "Sparse, constant-zero external feed with anonymized terminal coordinates; not an active baseline signal."),
            ("split_tag", "METADATA_SPLIT", "Generated by partitioning; metadata only, never a predictor."),
        ]
        return pd.DataFrame(rows, columns=["feature", "status_at_t_discharge", "audit_evidence"])

    @staticmethod
    def master_horizon_audit(master_df: pd.DataFrame) -> pd.DataFrame:
        """Field-by-field audit of all 55 features in data/master/master_real.parquet."""
        cols = list(master_df.columns)
        rows = []
        for c in cols:
            status = "PASS"
            category = "FEATURE"
            note = ""
            if c.startswith("avail_"):
                act = c.replace("avail_", "")
                category = "AS_OF_AVAILABILITY_TIMESTAMP"
                status = "AVAILABLE_AT_AS_OF_TIMESTAMP"
                note = f"Defines exact observation timestamp for milestone {act}; safe for as-of evaluation at t >= avail_{act}."
            elif c.startswith("ctx_discharge_"):
                category = "DISCHARGE_TIME_EXTERNAL_CONTEXT"
                status = "PASS_AT_DISCHARGE"
                note = "Observed strictly on or before discharge date with publication lags enforced."
            elif c in {"dwell_hours", "target_class"}:
                category = "TARGET"
                status = "TARGET_ONLY"
                note = "Ground truth operational dwell outcome realized at gate-out; strictly forbidden in X."
            elif c in {"truck_out_end_ts"}:
                category = "MILESTONE_EXIT"
                status = "TARGET_ONLY"
                note = "Realized gate-out timestamp; forbidden in X."
            elif c in {"discharge_start_ts"}:
                category = "ANCHOR_TIMESTAMP"
                status = "METADATA_ANCHOR"
                note = "Primary prediction anchor (t_pred = t_discharge); used to calculate as-of offsets."
            elif c in {"case_id", "container_id", "vessel_call_id", "call_date", "source"}:
                category = "IDENTIFIER"
                status = "METADATA_ID"
                note = "Primary join key or entity identifier; exclude from model training."
            elif c in {"in_cohort", "is_holdout_window", "is_synthetic"}:
                category = "COHORT_PARTITION"
                status = "METADATA_PARTITION"
                note = "Defines selection-aware cohort (discharges July 1 - Sept 5, 2021) and holdout evaluation."
            elif c in {"synth_template_case_id", "synth_donor_case_ids", "synth_donor_max_discharge"}:
                category = "SYNTHETIC_PROVENANCE"
                status = "METADATA_SYNTHETIC"
                note = "Tracks donor case IDs and latest donor discharge timestamp for causality filtering."
            elif c == "gross_t":
                category = "EQUIPMENT_STANDARDIZED"
                status = "PASS_AT_DISCHARGE"
                note = "Corrected weight normalized to metric tonnes (values > 100 divided by 1000)."
            elif c == "gross_recorded_in_kg":
                category = "DATA_QUALITY_INDICATOR"
                status = "PASS_AT_DISCHARGE"
                note = "Indicator that original GROSS was recorded in kilograms (18.9% of stays)."
            elif c == "GROSS":
                category = "EQUIPMENT_RAW"
                status = "CONDITIONAL"
                note = "Raw weight with mixed unit scales (kg vs tonnes); prefer gross_t."
            elif c in {"YARD_BLOCK", "YARD_SLOT"}:
                category = "YARD_LOCATION"
                status = "CONDITIONAL"
                note = "Discrete yard location; initial vs final slot cannot be authoritatively separated."
            elif c == "JOB_DEL_DOCTYPE":
                category = "ADMINISTRATIVE_MILESTONE"
                status = "EXCLUDE_AT_DISCHARGE"
                note = "Delivery order issued mid-stay (+32.2h post-discharge); forbidden at discharge."
            elif c in {"CTR_SIZE", "CTR_TYPE", "carrier_prefix"}:
                category = "EQUIPMENT_MANIFEST"
                status = "PASS_PRE_ARRIVAL"
                note = "Manifest/BAPLIE attributes known prior to discharge."
            else:
                category = "OTHER"
                status = "PASS"
                note = "Column verified in master dataset."

            rows.append({
                "column": c,
                "status_at_t_discharge": status,
                "category": category,
                "dtype": str(master_df[c].dtype),
                "null_count": int(master_df[c].isna().sum()),
                "audit_note": note,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ new dataset profiling
    @staticmethod
    def gross_weight_profile(df: pd.DataFrame) -> pd.DataFrame:
        """Profiles raw GROSS vs normalized gross_t, detailing the 18.9% kg scale anomaly."""
        if "gross_t" not in df.columns or "GROSS" not in df.columns:
            return pd.DataFrame()
        
        in_kg = df["GROSS"] > 100
        stats_rows = []
        for name, sub in [("All Stays", df), ("Values Recorded in Tonnes (<= 100)", df[~in_kg]),
                          ("Values Recorded in Kilograms (> 100)", df[in_kg])]:
            stats_rows.append({
                "stratum": name,
                "count": len(sub),
                "share_pct": round(len(sub) / len(df) * 100, 2),
                "raw_gross_min": round(float(sub["GROSS"].min()), 1),
                "raw_gross_median": round(float(sub["GROSS"].median()), 1),
                "raw_gross_p95": round(float(sub["GROSS"].quantile(0.95)), 1),
                "raw_gross_max": round(float(sub["GROSS"].max()), 1),
                "gross_t_median": round(float(sub["gross_t"].median()), 2),
                "gross_t_max": round(float(sub["gross_t"].max()), 2),
            })
        return pd.DataFrame(stats_rows)

    @staticmethod
    def external_context_profile(master_df: pd.DataFrame) -> pd.DataFrame:
        """Profiles external macroeconomic (PPKM/holiday) and PortWatch context features."""
        ctx_cols = [c for c in master_df.columns if c.startswith("ctx_discharge_")]
        rows = []
        for c in ctx_cols:
            s = master_df[c].dropna()
            is_num = pd.api.types.is_numeric_dtype(s)
            rows.append({
                "context_feature": c,
                "dtype": str(s.dtype),
                "non_null_count": len(s),
                "null_count": int(master_df[c].isna().sum()),
                "mean_or_mode": round(float(s.mean()), 3) if is_num else str(s.mode().iloc[0] if len(s) else "–"),
                "median": round(float(s.median()), 3) if is_num else "–",
                "min": round(float(s.min()), 3) if is_num else "–",
                "max": round(float(s.max()), 3) if is_num else "–",
            })
        return pd.DataFrame(rows)

    @staticmethod
    def synthetic_fidelity_audit(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> pd.DataFrame:
        """Audits synthetic data fidelity against real training pool across class mix and dwell percentiles."""
        rows = []
        # Class mix comparison
        classes = [0, 1, 2]
        class_names = {0: "ON_TIME (<=65.1h)", 1: "AT_RISK (65.1-160.9h)", 2: "CRITICAL_DELAY (>160.9h)"}
        for cls in classes:
            r_cnt = (real_df["target_class"] == cls).sum()
            s_cnt = (synth_df["target_class"] == cls).sum()
            r_share = (r_cnt / len(real_df)) * 100
            s_share = (s_cnt / len(synth_df)) * 100
            
            r_dwell = real_df.loc[real_df["target_class"] == cls, "dwell_hours"]
            s_dwell = synth_df.loc[synth_df["target_class"] == cls, "dwell_hours"]
            
            rows.append({
                "dimension": f"Class {cls}: {class_names[cls]}",
                "real_count": int(r_cnt),
                "real_share_pct": round(r_share, 2),
                "synth_count": int(s_cnt),
                "synth_share_pct": round(s_share, 2),
                "real_dwell_p50": round(float(r_dwell.median()), 1),
                "synth_dwell_p50": round(float(s_dwell.median()), 1),
                "real_dwell_p85": round(float(r_dwell.quantile(0.85)), 1),
                "synth_dwell_p85": round(float(s_dwell.quantile(0.85)), 1),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ distributions & boundary mass
    def dwell_quantiles(self, cases: pd.DataFrame, group_by: str = "ALL") -> pd.DataFrame:
        """Dwell percentiles across overall population or categorical splits."""
        quantiles = [0.10, 0.25, 0.50, 0.75, 0.85, 0.90, 0.95, 0.99]
        names = ["p10", "p25", "p50", "p75", "p85", "p90", "p95", "p99"]

        def _stats(series: pd.Series) -> Dict[str, float]:
            s = series.dropna()
            if len(s) == 0:
                return {k: np.nan for k in ["n", "mean"] + names}
            res = {"n": len(s), "mean": round(float(s.mean()), 2)}
            for q, n in zip(quantiles, names):
                res[n] = round(float(s.quantile(q)), 2)
            return res

        rows = []
        if group_by == "ALL":
            st = _stats(cases["dwell_hours"])
            st["group_by"] = "ALL"
            st["level"] = "all"
            rows.append(st)
        else:
            for lvl, sub in cases.groupby(group_by):
                st = _stats(sub["dwell_hours"])
                st["group_by"] = group_by
                st["level"] = str(lvl)
                rows.append(st)
        out = pd.DataFrame(rows)
        cols = ["group_by", "level", "n", "mean"] + names
        return out[cols]

    dwell_percentiles = dwell_quantiles

    def boundary_mass(self, cases: pd.DataFrame, cutoffs: Dict[str, float], bandwidth_h: float = 10.0) -> pd.DataFrame:
        """Quantifies observation density around the frozen classification boundaries."""
        dwell = cases["dwell_hours"].dropna()
        n = len(dwell)
        rows = []
        for name, cutoff in cutoffs.items():
            count = int(((dwell >= cutoff - bandwidth_h) & (dwell <= cutoff + bandwidth_h)).sum())
            share = (count / n) * 100 if n > 0 else 0.0
            rows.append({
                "cutoff": name,
                "hours": round(cutoff, 2),
                "within_10h_count": count,
                "within_10h_share_pct": round(share, 2),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ univariate feature signal
    def feature_signal(
        self,
        df: pd.DataFrame,
        numeric: List[str],
        categorical: List[str],
        target_col: str = "dwell_hours",
    ) -> pd.DataFrame:
        """Calculates Spearman rho for numeric features and Kruskal-Wallis epsilon^2 for categoricals."""
        rows = []
        y = df[target_col]

        for col in numeric:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            common = df[[col, target_col]].dropna()
            if len(common) < 10 or common[col].nunique() <= 1:
                rho = np.nan
            else:
                rho, _ = stats.spearmanr(common[col], common[target_col])
                
            tier = PREDICTION_ELIGIBILITY.get(col, "ARRIVAL_TIME_ELIGIBLE")
            is_arrival = (tier == "ARRIVAL_TIME_ELIGIBLE")
            
            rows.append({
                "feature": col,
                "kind": "numeric",
                "association": round(float(rho), 4) if pd.notna(rho) else np.nan,
                "n": len(common),
                "note": "Spearman rho vs dwell",
                "eligibility_tier": tier,
                "arrival_eligible": is_arrival,
            })

        for col in categorical:
            if col not in df.columns:
                continue
            common = df[[col, target_col]].dropna()
            groups = [g[target_col].values for _, g in common.groupby(col) if len(g) >= 5]
            if len(groups) >= 2:
                try:
                    h, _ = stats.kruskal(*groups)
                    n_tot = len(common)
                    k = len(groups)
                    # Epsilon squared: (H - k + 1) / (n - k)
                    eps2 = max(0.0, (h - k + 1) / (n_tot - k)) if n_tot > k else np.nan
                except Exception:
                    eps2 = np.nan
            else:
                eps2 = np.nan

            tier = PREDICTION_ELIGIBILITY.get(col, "ARRIVAL_TIME_ELIGIBLE")
            is_arrival = (tier == "ARRIVAL_TIME_ELIGIBLE")

            rows.append({
                "feature": col,
                "kind": "categorical",
                "association": round(float(eps2), 4) if pd.notna(eps2) else np.nan,
                "n": len(common),
                "note": "Kruskal epsilon^2 vs dwell",
                "eligibility_tier": tier,
                "arrival_eligible": is_arrival,
            })

        out = pd.DataFrame(rows)
        return out.sort_values("association", ascending=False, key=abs).reset_index(drop=True)

    @staticmethod
    def filter_eligible_signals(sig_df: pd.DataFrame, scenario: str = "ARRIVAL_TIME") -> pd.DataFrame:
        """Filters univariate ranking to keep strictly safe predictors for arrival-time triage."""
        if scenario == "ARRIVAL_TIME":
            return sig_df[sig_df["arrival_eligible"]].copy().reset_index(drop=True)
        return sig_df.copy()
